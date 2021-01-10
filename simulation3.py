import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class simulation:

    def get_portfolio_composition(self):
        portfolio_composition = [('VMC', 0.25), ('EMR', 0.25), ('CSX', 0.25), ('UNP', 0.25)]
        returns = pd.DataFrame({})
        for t in portfolio_composition:
            name = t[0]
            ticker = yfinance.Ticker(name)
            data = ticker.history(interval="1d", start="1980-01-01", end="2020-12-31")

            data['return_%s' % (name)] = data['Close'].pct_change(1)
            returns = returns.join(data[['return_%s' % (name)]], how="outer").dropna()
        return returns

    def get_mean_std(self, data):
        print('Mean: \n', data.mean())
        print('STD: \n', data.std())


if __name__ == "__main__":
    SM = simulation()
    # data = SM.get_portfolio_composition()
    # data.to_csv("data.csv")
    data = pd.read_csv("data.csv")
    SM.get_mean_std(data)
