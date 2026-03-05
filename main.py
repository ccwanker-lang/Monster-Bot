#!/usr/bin/env python3
# main.py

import websocket
import json
import pandas as pd
import numpy as np
import time
from threading import Thread
from collections import deque
import os
import csv

from core.portfolio import Portfolio
from core.dqn import DQNAgent
from indicators.moving_average import moving_average
from indicators.rsi import rsi
from indicators.macd import macd

# ===============================
# Config
# ===============================
KRAKEN_PAIR = "XBT/USD"
CANDLE_LIMIT = 300 
LOG_FILE = "logs/trades.csv"
MODEL_FILE = "models/dqn_model.pt"
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===============================
# Portfolio & AI (MET AUTO-LOAD!)
# ===============================
portfolio = Portfolio()
STATE_SIZE = 16  
ACTION_SIZE = 3  
agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

if os.path.exists(MODEL_FILE):
    print(f"✅ Pre-getraind model gevonden! Brein inladen...")
    agent.load(MODEL_FILE)
    # Zet epsilon heel laag, we willen traden, niet meer random gokken!
    agent.epsilon = 0.05 

# ===============================
# Opslag & Logging
# ===============================
ohlcv_deque = deque(maxlen=CANDLE_LIMIT)
last_state = None
last_action = None

def log_trade(action, price, btc, usdt, confidence=0.0):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), action, price, btc, usdt, confidence])

# ===============================
# Process tick (Live Data)
# ===============================
def process_tick(price, volume):
    global last_state, last_action
    
    timestamp = int(time.time())
    ohlcv_deque.append({"timestamp": timestamp, "open": price, "high": price, "low": price, "close": price, "volume": volume})

    if len(ohlcv_deque) < CANDLE_LIMIT:
        return

    df = pd.DataFrame(list(ohlcv_deque))
    
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
    row = df.iloc[-1]

    price_dist_ma20 = (row["close"] - row["MA20"]) / row["MA20"]
    macd_delta_norm = (row["MACD"] - row["SIGNAL"]) / (row["close"] * 0.01 + 1e-8)
    volatility = np.std(df["close"][-15:]) / row["close"]
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
    elif action in ["STOPLOSS", "TAKEPROFIT"]: pass  
    else: action = "HOLD"

    new_value = portfolio.value(row["close"])
    reward = ((new_value - prev_value) / prev_value) * 100

    if action == "HOLD" and portfolio.BTC == 0 and row["MTF_Trend_Up"] == 1 and row["FVG_Bull"] == 1:
        reward -= 0.005 
    if special_action == "TAKEPROFIT": reward += 2.0
    elif special_action == "STOPLOSS": reward -= 2.0
    if portfolio.get_drawdown(row["close"]) > 0.02: reward -= 0.5

    if last_state is not None:
        agent.remember(last_state, last_action, reward, current_state, done=False)
        agent.replay()
        # Save live model every now and then
        if int(time.time()) % 300 == 0: 
            agent.save(MODEL_FILE)

    last_state = current_state
    last_action = action

    position = "LONG" if portfolio.BTC > 0 else "NONE"
    print(f"{action} (Conf:{confidence:.2f}) | P={row['close']:.1f} | MTF={'UP' if row['MTF_Trend_Up'] else 'DN'} | FVG={'BULL' if row['FVG_Bull'] else ('BEAR' if row['FVG_Bear'] else '-')} | Val={new_value:.2f}")
    
    if action in ["BUY", "SELL", "STOPLOSS", "TAKEPROFIT"]:
        log_trade(action, row["close"], portfolio.BTC, portfolio.USDT, confidence)

# ===============================
# WebSocket & Startup
# ===============================
def on_message(ws, message):
    try:
        data = json.loads(message)
        if isinstance(data, list) and len(data) > 1:
            ticker = data[1]
            if "c" in ticker:
                process_tick(float(ticker["c"][0]), float(ticker["c"][1]))
    except Exception as e: pass 

def on_open(ws):
    ws.send(json.dumps({"event": "subscribe", "pair": [KRAKEN_PAIR], "subscription": {"name": "ticker"}}))
    print(f"[{time.strftime('%H:%M:%S')}] Monster Bot LIVE. Analyzing 16 Features (SMC, MTF, Liquidity)")

def start_ws():
    while True:
        try:
            ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_message=on_message, on_open=on_open)
            ws.run_forever()
        except Exception: time.sleep(5)

if __name__ == "__main__":
    ws_thread = Thread(target=start_ws)
    ws_thread.start()
    while True: time.sleep(1)

