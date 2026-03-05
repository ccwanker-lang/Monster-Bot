#!/usr/bin/env python3
# download_data.py

import requests
import pandas as pd
import os
import time

def fetch_kraken_data():
    print("Downloaden van historische Kraken data (1-minuut candles)...")
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        "pair": "XXBTZUSD", # Kraken interne naam voor BTC/USD
        "interval": 5       # 1 minuut
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data["error"]:
        print("Kraken API Error:", data["error"])
        return

    # Extraheer data
    candles = data["result"]["XXBTZUSD"]
    
    # Maak er een net DataFrame van
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    
    # Converteer strings naar floats
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
        
    df["timestamp"] = df["timestamp"].astype(int)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/historical.csv", index=False)
    print(f"✅ Succesvol {len(df)} candles opgeslagen in data/historical.csv")

if __name__ == "__main__":
    fetch_kraken_data()

