from datetime import datetime
from colorama import Fore
import random

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.entities import Asset


class Mag7SentimentBot(Strategy):

    def initialize(self, cash_at_risk: float = 0.2):
        self.set_market("stock")
        self.sleep_time = "1D"
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

    def get_sentiments(self):
        return {symbol: random.uniform(-1, 1) for symbol in self.mag7}

    def get_position_size(self, stock_symbol):
        asset = self.assets[stock_symbol]
        cash = self.get_cash()
        last_price = self.get_last_price(asset, quote=self.quote)
        if last_price is None or last_price == 0:
            return cash, last_price, 0
        quantity = (cash * self.cash_at_risk) / last_price
        return cash, last_price, quantity

    def on_trading_iteration(self):
        now = self.get_datetime()
        current_week = now.isocalendar()[1]

        if current_week == self.last_trade_week:
            return

        self.last_trade_week = current_week

        sentiments = self.get_sentiments()

        best_buy = max(sentiments.items(), key=lambda x: x[1])
        best_sell = min(sentiments.items(), key=lambda x: x[1])

        buy_symbol, buy_sentiment = best_buy
        sell_symbol, sell_sentiment = best_sell

        print(Fore.CYAN + f"[{now.date()}] Sentiments: {sentiments}" + Fore.RESET)

        # Try to BUY
        if buy_sentiment > 0:
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

        # Try to SELL
        if sell_sentiment < 0:
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


if __name__ == "__main__":
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 1, 1)

    Mag7SentimentBot.backtest(
        datasource_class=YahooDataBacktesting,
        backtesting_start=start_date,
        backtesting_end=end_date,
        parameters={
            "cash_at_risk": 0.2,
        }
    )
