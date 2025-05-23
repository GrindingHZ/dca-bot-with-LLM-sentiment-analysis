from datetime import datetime
from colorama import Fore
import random

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.entities import Asset

# Additional imports for sentiment analysis
from timedelta import Timedelta
from langchain_ollama import OllamaLLM
from ollama import chat
from pydantic import BaseModel
import json

from llmprompts import get_web_deets, prompt_template

import yfinance as yf

class Response(BaseModel):
    sentiment: str
    score: float

    class Config:
        schema_extra = {
            "example": {
                "sentiment": "positive",
                "score": 0.2
            }
        }

class Mag7SentimentBot(Strategy):
    """
    A trading bot that uses sentiment analysis with LLM that is locally installed with Ollama to trade the Magnificent 7 stocks.
    The concept is to do a DCA strategy on the individual stocks in the Magnificent 7 stocks based on the sentiment of the news.
    """

    def initialize(self, cash_at_risk: float = 0.025):
        """
        Initializes the bot with a list of the Magnificent 7 stocks and sets the needed parameters.
        """
        self.set_market("stock")
        self.sleep_time = "1W"
        self.cash_at_risk = cash_at_risk
        self.last_trade_week = None

        self.mag7 = [
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "GOOGL", # Alphabet
            "AMZN",  # Amazon
            "META",  # Meta Platforms
            "NVDA",  # Nvidia
            "TSLA"   # Tesla
        ]
        self.assets = {symbol: Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK) for symbol in self.mag7}
        self.quote = Asset(symbol="USD", asset_type=Asset.AssetType.FOREX)

    def get_dates(self):
        """
        Returns the current date and the date one day prior.
        """
        today = self.get_datetime()
        day_prior = today - Timedelta(days=7)
        return today.strftime("%Y-%m-%d"), day_prior.strftime("%Y-%m-%d")

    def get_sentiments(self):
        """
        Get the sentiments for the Magnificent 7 stocks using the LLM based on the news data that is fetched
        using the get_web_deets function, which is based on the Serper API.
        """
        today, day_prior = self.get_dates()
        sentiments = {}

        # Iterate through each of the stock symbols in the Magnificent 7 and get the sentiment
        for symbol in self.mag7:
            # Collect news data for the stock
            news = get_web_deets(
                news_start_date=day_prior,
                news_end_date=today,
                stock_name=symbol
            )
            
            # Use the LLM to get the sentiment
            stream = chat(
                model="qwen2.5:14b",
                messages=[
                    {"role": "user", "content": prompt_template(news)}
                ],
                format=Response.model_json_schema()
            )

            result = json.loads(stream["message"]["content"])
            print(Fore.LIGHTBLUE_EX + f"Sentiment for {symbol}: {result}" + Fore.RESET)
            sentiments[symbol] = result["score"]

        return sentiments
        
    def get_position_size(self, stock_symbol):
        # Get the necessary data for the stock
        asset = self.assets[stock_symbol]
        cash = self.get_cash()
        portfolio = self.get_portfolio_value()
        last_price = self.get_last_price(asset, quote=self.quote)

        # Calculate the position size based on the cash at risk and portfolio
        if last_price is None or last_price == 0 or cash < 100: # Let there be a buffer of $100
            quantity = 0
        elif portfolio * self.cash_at_risk > cash:
            quantity = (cash - 100) / last_price
        else:
            quantity = (portfolio * self.cash_at_risk) / last_price

        if quantity < 0:
            quantity = 0
            
        return cash, last_price, quantity

    def get_market_caps(self):
        """
        Fetches the current market capitalization for each of the Mag7 stocks.
        Returns a dictionary with symbols as keys and market caps in billions as values.
        """
        market_caps = {}
        
        # Iterate through each of the stock symbols in the Magnificent 7 and get the market cap
        for symbol in self.mag7:
            # Collect the last price for the stock
            asset = self.assets[symbol]
            price = self.get_last_price(asset, quote=self.quote)

            # If the price is None or 0, skip to the next stock
            if price is None or price == 0:
                continue

            # Try to get the market cap from the API    
            ticker = yf.Ticker(symbol)
            shares_outstanding = ticker.info.get("sharesOutstanding")
            market_cap = price * shares_outstanding / 1_000_000_000  # Convert to billions
            market_caps[symbol] = market_cap
            
        print(Fore.CYAN + f"Market Caps (billions): {market_caps}" + Fore.RESET)
        return market_caps

    def on_trading_iteration(self):
        """
        On the first week of the bot investment journey, it is going to allocate the first 50% of the portfolio to the Mag7 stocks based on 
        their market cap, i.e. we are using the indexing strategy.

        Decision on buying based on sentiments of each stock
        1. Buy the stocks with positive sentiment of lets say more than or equal to 0.5 with the portion of 5% of the value of the whole portfolio
        (if the cash is not enough that is less than 5% of the portfolio, just buy it with all of the available cash).
        2. If there is no cash available, you do not need to buy anything.
        3. If there is no stocks with the sentiment of equal to or more than 0.5, just hold, you don't need to buy anything.

        Decision on selling based on sentiments of each stock
        1. Sell the stocks with negative sentiment of less than -0.8, where we are immediately selling all our position at that stock

        """
        now = self.get_datetime()
        current_week = now.isocalendar()[1]

        # Get the portfolio value and cash
        portfolio_value = self.get_portfolio_value()
        cash = self.get_cash()
        
        # Check if it is the first week
        if self.last_trade_week is None:
            print(Fore.YELLOW + f"[{now.date()}] Portfolio Value: ${portfolio_value:.2f} (Cash: ${cash:.2f})" + Fore.RESET)
            print(Fore.CYAN + f"[{now.date()}] Initial portfolio allocation - buying Mag7 stocks" + Fore.RESET)
            
            # Get the market caps of the stocks
            market_caps = self.get_market_caps()
            total_market_cap = sum(market_caps.values())

            # Allocate 50% of the portfolio to the Mag7 stocks
            total_allocation = portfolio_value * 0.5

            # Iterate through each of the stock symbols in the Magnificent 7 and buy based on market cap
            for symbol, market_cap in market_caps.items():
                weight = market_cap / total_market_cap
                stock_allocation = total_allocation * weight
                
                # Get current price and calculate quantity to buy
                asset = self.assets[symbol]
                last_price = self.get_last_price(asset, quote=self.quote)
                
                if last_price is None or last_price == 0:
                    continue
                    
                quantity = stock_allocation / last_price
                
                # Make buy order
                if quantity > 0:
                    order = self.create_order(
                        asset,
                        quantity,
                        "buy",
                        order_type="market",
                        quote=self.quote,
                    )
                    self.submit_order(order)
                    print(Fore.GREEN + f"INITIAL BUY {symbol}: {quantity} shares @ {last_price:.2f} (Weight: {weight:.2%})" + Fore.RESET)
            
            # Set the last trade week to the current week
            self.last_trade_week = current_week
            return

        # We are iterating through the weeks by stepping day per day, thus we need to check if we are in the same week
        if current_week == self.last_trade_week:
            return

        print(Fore.YELLOW + f"[{now.date()}] Portfolio Value: ${portfolio_value:.2f} (Cash: ${cash:.2f})" + Fore.RESET)

        self.last_trade_week = current_week
        traded_indicator = False

        # Get the sentiments for the stocks
        sentiments = self.get_sentiments()
        best_buy = [sentiment for sentiment in sentiments.items() if sentiment[1] >= 0.5]
        best_sell = [sentiment for sentiment in sentiments.items() if sentiment[1] <= -0.8]
        buy_symbols, buy_sentiments = [sentiment[0] for sentiment in best_buy], [sentiment[1] for sentiment in best_buy]
        sell_symbols, sell_sentiments = [sentiment[0] for sentiment in best_sell], [sentiment[1] for sentiment in best_sell]

        print(Fore.CYAN + f"[{now.date()}] Sentiments: {sentiments}" + Fore.RESET)

        # Try to BUY
        for buy_symbol, buy_sentiment in zip(buy_symbols, buy_sentiments):
            # Get the position size for the stock
            cash, last_price, quantity = self.get_position_size(buy_symbol)

            if quantity > 0 and cash >= quantity * last_price:
                order = self.create_order(
                    self.assets[buy_symbol],
                    quantity,
                    "buy",
                    order_type="market",
                    quote=self.quote,
                )
                self.submit_order(order)
                print(Fore.GREEN + f"BUY {buy_symbol} @ {last_price:.2f} (Sentiment: {buy_sentiment:.2f})" + Fore.RESET)
                traded_indicator = True
            else:
                print(Fore.YELLOW + f"Not enough cash to buy {buy_symbol}" + Fore.RESET)
                break

        # Try to SELL
        for sell_symbol, sell_sentiment in zip(sell_symbols, sell_sentiments):
            # Get the position size for the stock
            position = self.get_position(self.assets[sell_symbol])
            quantity_owned = position.quantity if position else 0
            
            if quantity_owned > 0:
                order = self.create_order(
                    self.assets[sell_symbol],
                    quantity_owned,
                    "sell",
                    order_type="market",
                    quote=self.quote,
                )
                self.submit_order(order)
                print(Fore.RED + f"SELL {sell_symbol} (Sentiment: {sell_sentiment:.2f})" + Fore.RESET)
                traded_indicator = True
        
        if not traded_indicator:
            print(Fore.YELLOW + f"No trades executed this week." + Fore.RESET)

if __name__ == "__main__":
    start_date = datetime(2024, 5, 1)
    end_date = datetime(2025, 5, 1)

    Mag7SentimentBot.backtest(
        datasource_class=YahooDataBacktesting,
        backtesting_start=start_date,
        backtesting_end=end_date,
        parameters={
            "cash_at_risk": 0.025,
        }
    )