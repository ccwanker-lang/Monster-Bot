# core/portfolio.py

FEE_RATE = 0.0025  # 0.25% fee

class Portfolio:
    def __init__(self):
        self.initial_capital = 1000
        self.USDT = self.initial_capital
        self.BTC = 0
        self.average_entry_price = 0  
        self.max_portfolio_value = self.initial_capital
        self.highest_price_seen = 0  # NIEUW: Houdt bij hoe hoog de prijs is gekomen sinds aankoop

    def value(self, price):
        return self.USDT + (self.BTC * price)

    def get_drawdown(self, current_price):
        current_val = self.value(current_price)
        if current_val > self.max_portfolio_value:
            self.max_portfolio_value = current_val
        drawdown = (self.max_portfolio_value - current_val) / self.max_portfolio_value
        return drawdown

    def buy(self, price, amount_usdt):
        if self.USDT >= amount_usdt and amount_usdt > 0:
            fee = amount_usdt * FEE_RATE
            amount_after_fee = amount_usdt - fee
            btc_bought = amount_after_fee / price
            
            total_cost_basis = (self.BTC * self.average_entry_price) + amount_after_fee
            self.BTC += btc_bought
            self.USDT -= amount_usdt
            self.average_entry_price = total_cost_basis / self.BTC
            
            # Zet de hoogste prijs op de huidige koopprijs
            self.highest_price_seen = price

    def sell_all(self, price):
        if self.BTC > 0:
            proceeds = self.BTC * price
            fee = proceeds * FEE_RATE
            self.USDT += proceeds - fee
            self.BTC = 0
            self.average_entry_price = 0
            self.highest_price_seen = 0

    def check_stop_take(self, price, hard_stop_loss=0.015, trailing_distance=0.02):
        """
        Checkt of we moeten verkopen op basis van:
        1. Een harde stop-loss (voor als de trade direct fout gaat).
        2. Een Trailing stop-loss (om winst vast te klikken als we stijgen).
        """
        if self.BTC > 0 and self.average_entry_price > 0:
            
            # Update de hoogste prijs als we verder stijgen
            if price > self.highest_price_seen:
                self.highest_price_seen = price

            # 1. Harde Stop-Loss (Als het direct misgaat vanaf entry, 1.5% verlies)
            if price <= self.average_entry_price * (1 - hard_stop_loss):
                self.sell_all(price)
                return "STOPLOSS"

            # 2. Trailing Stop-Loss (Klikt winst vast)
            # Pas actief als we de trailing afstand in de plus staan om direct uitstoten te voorkomen
            if self.highest_price_seen > self.average_entry_price * (1 + trailing_distance):
                # Als we vanaf de hoogste piek met 2% zakken, verkoop!
                if price <= self.highest_price_seen * (1 - trailing_distance):
                    self.sell_all(price)
                    return "TAKEPROFIT" # We noemen dit takeprofit omdat we in de plus zijn uitgestapt
                    
        return None

