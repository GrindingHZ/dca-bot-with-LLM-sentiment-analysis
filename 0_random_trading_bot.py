from datetime import datetime
from colorama import Fore
import random

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.entities import Asset


class WeeklyRandomBot(Strategy):

    def initialize(self, cash_at_risk: float = 0.2, stock: str = "SPY"):
        self.set_market("stock")
        self.sleep_time = "1D"  # Still use 1 day to simulate real market days
        self.cash_at_risk = cash_at_risk
        self.stock = stock
        self.asset = Asset(symbol=self.stock, asset_type=Asset.AssetType.STOCK)
        self.quote = Asset(symbol="USD", asset_type=Asset.AssetType.FOREX)
        self.last_trade_week = None  # Keep track of last week we traded

    def get_position_size(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.asset, quote=self.quote)
        if last_price is None or last_price == 0:
            return cash, last_price, 0
        quantity = (cash * self.cash_at_risk) / last_price
        return cash, last_price, quantity

    def decide_action(self):
        return random.choice(["buy", "sell", "hold"])

    def on_trading_iteration(self):
        now = self.get_datetime()
        current_week = now.isocalendar()[1]  # ISO week number

        if current_week == self.last_trade_week:
            return  # Skip if we already traded this week

        # Proceed with a trade
        self.last_trade_week = current_week

        cash, last_price, quantity = self.get_position_size()
        position = self.get_position(self.asset)
        quantity_owned = position.quantity if position else 0

        if last_price is None:
            print(Fore.YELLOW + "Price unavailable. Skipping." + Fore.RESET)
            return

        decision = self.decide_action()
        print(Fore.CYAN + f"[{now.date()}] Decision this week: {decision.upper()}" + Fore.RESET)

        if decision == "hold":
            return

        elif decision == "buy" and cash >= quantity * last_price:
            order = self.create_order(
                self.asset,
                quantity,
                "buy",
                order_type="market",
                quote=self.quote,
            )
            self.submit_order(order)
            print(Fore.GREEN + f"BUY order submitted: {order}" + Fore.RESET)

        elif decision == "sell" and quantity_owned > 0:
            order = self.create_order(
                self.asset,
                quantity_owned,
                "sell",
                order_type="market",
                quote=self.quote,
            )
            self.submit_order(order)
            print(Fore.RED + f"SELL order submitted: {order}" + Fore.RESET)


if __name__ == "__main__":
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 5, 1)

    WeeklyRandomBot.backtest(
        datasource_class=YahooDataBacktesting,
        backtesting_start=start_date,
        backtesting_end=end_date,
        parameters={
            "cash_at_risk": 0.2,
            "stock": "SPY",
        }
    )
