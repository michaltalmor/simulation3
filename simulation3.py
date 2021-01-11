import seaborn as sns
import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class simulation:
    portfolio_composition = [('VMC', 0.25), ('EMR', 0.25), ('CSX', 0.25), ('UNP', 0.25)]

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

    def plot_hist(self, data):
        for col in data:
            if col == "Date":
                continue
            fig = plt.figure()
            ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            data[col].plot.hist(bins=60)
            ax1.set_xlabel("Daily returns %")
            ax1.set_ylabel("Number of days")
            ax1.set_title(f'{col} hist plot')
            ax1.text(-0.35, 200, "Extreme Low\nreturns")
            ax1.text(0.25, 200, "Extreme High\nreturns")
            plt.show()


    def get_mean_std(self, data):
        print('Mean: \n', data.mean())
        print('STD: \n', data.std())



    def get_covariance_correlation(self, data):
        d = data.drop("Date", axis=1)
        columns = d.columns
        d = d.values.T

        covmat = np.cov(d)
        # print('covariance matrix: \n', covmat)
        ax = plt.axes()
        sns.heatmap(covmat, annot=True, xticklabels=columns, yticklabels=columns, ax=ax)
        ax.set_title('covariance matrix')
        plt.show()

        ax2 = plt.axes()
        correlation_mat = data.corr()
        sns.heatmap(correlation_mat, annot=True, ax=ax2)
        ax2.set_title('correlation matrix')
        plt.show()

    def get_correlation(self, data):
        for col in data:
            if col == "Date":
                continue
            df = pd.DataFrame()
            df['T'] = data[col]
            df['T+1'] = data[col].shift(1)
            correlation_mat = df.corr()
            ax2 = plt.axes()
            sns.heatmap(correlation_mat, annot=True, ax=ax2)
            ax2.set_title(f'{col} correlation matrix')
            plt.show()

    #For single stock
    def simulate_returns(self, historical_returns, forecast_days):
        records=pd.DataFrame()
        while forecast_days>0:
            record=historical_returns.sample(n=1, replace=True)
            index=record.index[0]

            if forecast_days-10>0:
                samples=historical_returns[index:index+10]
                records=pd.concat([records, samples])
                forecast_days=forecast_days-10
            else:
                samples=historical_returns[index:index+forecast_days]
                records=pd.concat([records, samples])
                forecast_days=0
        records=records.reset_index(drop=True)
        return records

    #For portfilio - only name and weight
    def simulate_portfolio(self, historical_returns, composition, forecast_days):
        result = 0
        for t in composition:
            name, weight = t[0], t[1]
            s = self.simulate_returns(historical_returns['return_%s' % (name)], forecast_days)
            result = result + s * weight
        return (result)

    def simulation(self, historical_returns, composition, forecast_days, n_iterations):
        simulated_portfolios = None
        for i in range(n_iterations):
            sim = self.simulate_portfolio(historical_returns, composition, forecast_days)
            sim_port = pd.DataFrame({'returns_%d' % (i): sim})
            if simulated_portfolios is None:
                simulated_portfolios = sim_port
            else:
                simulated_portfolios = simulated_portfolios.join(sim_port)
        return simulated_portfolios

    def get_simulation_mean(self, simulated_portfolios):
        mean_data = simulated_portfolios.mean()*885
        return mean_data

if __name__ == "__main__":
    SM = simulation()
    portfolio_composition = SM.portfolio_composition
    # data = SM.get_portfolio_composition()
    # data.to_csv("data.csv")
    data = pd.read_csv("data.csv")
    # SM.get_mean_std(data)
    # SM.plot_hist(data)
    # SM.get_covariance_correlation(data)
    # SM.get_correlation(data)
    simulated_portfolios = SM.simulation(data, portfolio_composition, int(253*3.5), 100)
    print(simulated_portfolios)
    simulation_mean = SM.get_simulation_mean(simulated_portfolios)

    target_return = 0.02
    target_prob_port = simulated_portfolios.cumsum().apply(
        lambda x: np.mean(x > target_return)
        , axis=1)
    print(target_prob_port)
    percentile_90th = simulated_portfolios.cumsum().apply(lambda x: np.percentile(x, 90), axis=1)
    average_port = simulated_portfolios.cumsum().apply(lambda x: np.mean(x), axis=1)
    print(f'Average port: {average_port}\n percentile_90th: {percentile_90th}')
