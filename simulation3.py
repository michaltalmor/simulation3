import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

portfolio_composition = [('VMC', 0.5), ('AAPL', 0.2), ('GOOG', 0.3)]
returns = pd.DataFrame({})
for t in portfolio_composition:
    name = t[0]
    ticker = yfinance.Ticker(name)
    data = ticker.history(interval="1d",
                          start="2010-01-01", end="2019-12-31")

    data['return_%s' % (name)] = data['Close'].pct_change(1)
    returns = returns.join(data[['return_%s' % (name)]],
                           how="outer").dropna()
print(returns)