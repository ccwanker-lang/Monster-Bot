# Monster-Bot
Als je nu met een compleet schone lei begint, hoef je letterlijk maar 3 stappen uit te voeren in je terminal:

python3 download_data.py

Wat gebeurt er? Het script maakt automatisch de map data/ aan en zet daar historical.csv in.

python3 backtest.py

Wat gebeurt er? Het script leest de data, traint de AI, maakt automatisch de map models/ aan en slaat het getrainde brein op als dqn_model.pt.

python3 main.py

Wat gebeurt er? Het script start op, laadt het brein in, maakt automatisch de map logs/ aan en begint je live trades weg te schrijven in trades.csv.
