#!/usr/bin/env python3
# stats.py

import pandas as pd
import numpy as np
import os

def analyze_trades():
    if not os.path.exists("logs/trades.csv"):
        print("Nog geen logboek gevonden! Laat de bot eerst wat trades maken.")
        return

    # 1. Lees de ruwe data in (de kolommen die we in main.py wegschrijven)
    df = pd.read_csv("logs/trades.csv", names=["timestamp", "action", "price", "btc", "usdt", "confidence"])

    if len(df) < 2:
        print("De bot heeft nog niet genoeg trades gesloten om statistieken te berekenen.")
        return

    # 2. Bereken de actuele portfolio waarde op elk moment
    df['portfolio_value'] = df['usdt'] + (df['btc'] * df['price'])

    # 3. Bereken de MAXIMALE DRAWDOWN (Hoe diep was de hardste val?)
    df['max_waarde_tot_nu_toe'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['max_waarde_tot_nu_toe'] - df['portfolio_value']) / df['max_waarde_tot_nu_toe']
    max_drawdown = df['drawdown'].max() * 100

    # 4. Bereken de WIN RATE (Hoeveel trades sloten we af met winst?)
    wins = 0
    losses = 0
    last_buy_price = 0

    for index, row in df.iterrows():
        if row['action'] == 'BUY':
            last_buy_price = row['price']
        elif row['action'] in ['SELL', 'TAKEPROFIT', 'STOPLOSS'] and last_buy_price > 0:
            # We moeten de fee meerekenen! 0.25% in en 0.25% uit = 0.5% in totaal om break-even te spelen.
            break_even_price = last_buy_price * 1.005 
            if row['price'] > break_even_price:
                wins += 1
            else:
                losses += 1
            last_buy_price = 0 # Reset voor de volgende trade

    totaal_trades = wins + losses
    win_rate = (wins / totaal_trades * 100) if totaal_trades > 0 else 0
    start_waarde = df['portfolio_value'].iloc[0]
    eind_waarde = df['portfolio_value'].iloc[-1]
    roi = ((eind_waarde - start_waarde) / start_waarde) * 100

    # 5. Print het strakke rapport
    print("\n" + "="*45)
    print("📊 JOUW BOT STATISTIEKEN (REAL-TIME)")
    print("="*45)
    print(f"Totaal aantal gesloten trades : {totaal_trades}")
    print(f"Winstgevende trades         : {wins}")
    print(f"Verliesgevende trades       : {losses}")
    print(f"Win Rate                    : {win_rate:.1f}%")
    print("-" * 45)
    print(f"Start Kapitaal              : {start_waarde:.2f} USDT")
    print(f"Huidige Waarde              : {eind_waarde:.2f} USDT")
    print(f"Netto Winst/Verlies (ROI)   : {roi:.2f}%")
    print(f"Maximale Drawdown (Dip)     : -{max_drawdown:.2f}%")
    print("="*45 + "\n")

if __name__ == "__main__":
    analyze_trades()
