#!/usr/bin/env python3
# backtest.py

import pandas as pd
import numpy as np
import time
import os

from core.portfolio import Portfolio
from core.dqn import DQNAgent
from indicators.moving_average import moving_average
from indicators.rsi import rsi
from indicators.macd import macd

DATA_FILE = "data/historical.csv"
MODEL_FILE = "models/dqn_model.pt"
STATE_SIZE = 16
ACTION_SIZE = 3
BATCH_SIZE = 64

def train_offline():
    if not os.path.exists(DATA_FILE):
        print(f"Fout: Kan {DATA_FILE} niet vinden. Draai eerst download_data.py!")
        return

    print("Data inladen en Smart Money Concepts berekenen...")
    df = pd.read_csv(DATA_FILE)

    df["MA5"] = moving_average(df["close"], 5)
    df["MA20"] = moving_average(df["close"], 20)
    df["MA50"] = moving_average(df["close"], 50)
    df["RSI"] = rsi(df)
    df["MACD"], df["SIGNAL"] = macd(df)
    
    df["cum_volume"] = df["volume"].cumsum()
    df["cum_pv"] = (df["close"] * df["volume"]).cumsum()
    df["VWAP"] = df["cum_pv"] / (df["cum_volume"] + 1e-8)
    
    df["FVG_Bull"] = ((df["low"] > df["high"].shift(2)) & (df["close"] > df["open"])).astype(int)
    df["FVG_Bear"] = ((df["high"] < df["low"].shift(2)) & (df["close"] < df["open"])).astype(int)
    
    df["Swing_High"] = df["high"].rolling(window=20, min_periods=1).max()
    df["Swing_Low"] = df["low"].rolling(window=20, min_periods=1).min()
    
    df["MA150"] = moving_average(df["close"], 150)
    df["MTF_Trend_Up"] = (df["MA20"] > df["MA150"]).astype(int)
    df = df.bfill() 

    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    if os.path.exists(MODEL_FILE):
        print("Vorige AI status ingeladen!")
        agent.load(MODEL_FILE)
    
    portfolio = Portfolio()
    last_state, last_action = None, None
    
    print(f"Start simulatie van {len(df)} candles...")
    start_time = time.time()

    for i in range(150, len(df)):
        row = df.iloc[i]
        
        volatility = df["close"].iloc[i-15:i].std() / row["close"]
        if np.isnan(volatility): volatility = 0
        
        price_dist_ma20 = (row["close"] - row["MA20"]) / row["MA20"]
        macd_delta_norm = (row["MACD"] - row["SIGNAL"]) / (row["close"] * 0.01 + 1e-8)
        price_dist_vwap = (row["close"] - row["VWAP"]) / (row["VWAP"] + 1e-8)
        dist_swing_high = (row["Swing_High"] - row["close"]) / row["close"]
        dist_swing_low = (row["close"] - row["Swing_Low"]) / row["close"]

        current_state = [
            int(row["RSI"] > 50), int(row["close"] > row["MA20"]), int(row["MA20"] > row["MA50"]), int(row["MACD"] > row["SIGNAL"]),
            int(portfolio.BTC > 0), row["RSI"] / 100.0, price_dist_ma20 * 100, int(row["MA5"] > row["MA20"]),
            macd_delta_norm, volatility * 1000, price_dist_vwap * 100, row["FVG_Bull"], row["FVG_Bear"],
            dist_swing_high * 100, dist_swing_low * 100, row["MTF_Trend_Up"]
        ]
        
        prev_value = portfolio.value(row["close"])
        action, confidence = agent.act(current_state)

        special_action = portfolio.check_stop_take(row["close"])
        if special_action: action = special_action

        if action == "BUY" and portfolio.USDT > 15: 
            bet_size_pct = max(0.2, min(0.8, confidence)) 
            portfolio.buy(row["close"], portfolio.USDT * bet_size_pct) 
        elif action == "SELL" and portfolio.BTC > 0:
            portfolio.sell_all(row["close"])

        new_value = portfolio.value(row["close"])
        reward = ((new_value - prev_value) / prev_value) * 100

        if action == "HOLD" and portfolio.BTC == 0 and row["MTF_Trend_Up"] == 1 and row["FVG_Bull"] == 1:
            reward -= 0.005 
        if special_action == "TAKEPROFIT": reward += 2.0
        elif special_action == "STOPLOSS": reward -= 2.0
        if portfolio.get_drawdown(row["close"]) > 0.02: reward -= 0.5

        if last_state is not None:
            agent.remember(last_state, last_action, reward, current_state, done=False)
            agent.replay(BATCH_SIZE)

        last_state = current_state
        last_action = action

    agent.save(MODEL_FILE)
    print(f"\n✅ Training compleet in {time.time() - start_time:.1f} sec!")
    print(f"Eindwaarde: {portfolio.value(row['close']):.2f} USDT")
    print("Het getrainde brein is opgeslagen. Start nu main.py!")

if __name__ == "__main__":
    train_offline()
