import seaborn as sns
import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

class simulation:
    portfolio_composition = [('VMC', 0.25), ('EMR', 0.25), ('CSX', 0.25), ('UNP', 0.25)]

    def get_portfolio_composition(self):
        portfolio_composition = [('VMC', 0.25), ('EMR', 0.25), ('CSX', 0.25), ('UNP', 0.25)]
        returns = pd.DataFrame({})
        self.days_profit = pd.DataFrame({})
        for t in portfolio_composition:
            name = t[0]
            ticker = yfinance.Ticker(name)
            data = ticker.history(interval="1d", start="1980-01-01", end="2020-12-31")

            data['day_profit_%s' % (name)] = data['Close'].pct_change(1)
            data['return_%s' % (name)] = data['day_profit_%s' % (name)].copy()
            # data['return_%s' % (name)] = np.exp(np.log1p(data['return_%s' % (name)]).cumsum())-1
            # returns = returns.join(data[['return_%s' % (name)]], how="outer").dropna()
            returns = returns.join(data[['return_%s' % (name)]], how="outer").dropna()
            self.days_profit = self.days_profit.join(data[['day_profit_%s' % (name)]], how="outer").dropna()
        return returns.reset_index(drop=True)

    def plot_hist(self):
        for col in self.days_profit:
            if col == "Date":
                continue
            fig = plt.figure()
            ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            self.days_profit[col].plot.hist(bins=60)
            ax1.set_xlabel("Daily returns %")
            ax1.set_ylabel("Number of days")
            ax1.set_title(f'{col} hist plot')
            ax1.text(-0.35, 200, "Extreme Low\nreturns")
            ax1.text(0.25, 200, "Extreme High\nreturns")
            plt.show()


    def get_mean_std(self):
        print('Mean: \n', self.days_profit.mean())
        print('STD: \n', self.days_profit.std())



    def get_covariance(self):
        # d = data.drop("Date", axis=1)
        d = self.days_profit.copy()
        columns = d.columns
        d = d.values.T

        covmat = np.cov(d)
        # print('covariance matrix: \n', covmat)
        ax = plt.axes()
        sns.heatmap(covmat, annot=True, xticklabels=columns, yticklabels=columns, ax=ax)
        ax.set_title('covariance matrix')
        plt.show()

        # ax2 = plt.axes()
        # correlation_mat = data.corr()
        # sns.heatmap(correlation_mat, annot=True, ax=ax2)
        # ax2.set_title('correlation matrix')
        # plt.show()

    def get_correlation(self):
        data = self.days_profit.copy()
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
        days = forecast_days
        num= int(days/10)
        while days>0:
            record=historical_returns.sample(n=1, replace=True)
            index=record.index[0]
            if days-num>0:#Receive 10 consecutive days
                while len(historical_returns)-index<=num:
                    record = historical_returns.sample(n=1, replace=True)
                    index = record.index[0]
                samples=historical_returns[index:index+num]
                records=pd.concat([records, samples])
                days=days-num
            else:#Get the rest of the days
                while len(historical_returns)-index<=days:
                    record = historical_returns.sample(n=1, replace=True)
                    index = record.index[0]
                samples=historical_returns[index:index+days]
                records=pd.concat([records, samples])
                days=0
        records=records.reset_index(drop=True)
        records_accu = np.exp(np.log1p(records).cumsum()) - 1

        return pd.Series(records_accu[0])
        # return historical_returns.sample(n=forecast_days, replace=True).reset_index(drop=True)

    def check_threshold(self, data):
        # col = data.cumsum()
        col = data.copy()
        for prof in col:
            if prof > 0.36:
                col = col.apply(lambda x: 0)
                col.iloc[-1] = 0.02
                return col
        return col

    #For portfilio - only name and weight
    def simulate_portfolio(self, historical_returns, composition, forecast_days, flag=False):
        result = 0
        for t in composition:
            name, weight = t[0], t[1]
            s = self.simulate_returns(historical_returns['return_%s' % (name)], forecast_days)
            if(flag):
                s = self.check_threshold(s)
            result = result + s * weight
        return (result)

    def simulation(self, historical_returns, composition, forecast_days, n_iterations, flag=False):
        simulated_portfolios = None
        for i in range(n_iterations):
            sim = self.simulate_portfolio(historical_returns, composition, forecast_days, flag)
            sim_port = pd.DataFrame({'returns_%d' % (i): sim})
            if simulated_portfolios is None:
                simulated_portfolios = sim_port
            else:
                simulated_portfolios = simulated_portfolios.join(sim_port)
        return simulated_portfolios

    def get_simulation_mean(self, simulated_portfolios):
        mean_data = simulated_portfolios.mean()*885
        return mean_data

    def total_profit_for_case_2(self, simulation):
        total_profit=[]
        for iter in simulation:
            col=simulation[iter]
            col= simulation[iter].cumsum()
            profit = col[-1]
            for prof in col:
                if prof>0.36:
                    profit=0.02
            total_profit.append(profit)

    def nullify_negative_profit(self, simulated_portfolios):
        sim = simulated_portfolios.copy()
        for col in sim:
            if sim[col][-1:].values[0] < 0:
                sim[col] = sim[col].apply(lambda x: 0)
        return sim



if __name__ == "__main__":
    Q2_2 = False # For Question 2.2
    SM = simulation()
    portfolio_composition = SM.portfolio_composition
    data = SM.get_portfolio_composition()
    # data.to_csv("data.csv")
    # data = pd.read_csv("data.csv")
    """Q1"""
    SM.get_mean_std()
    SM.plot_hist()
    SM.get_covariance()
    SM.get_correlation()
    """Q2"""
    simulated_portfolios = SM.simulation(data, portfolio_composition, int(253*3.5), 100, flag=Q2_2)

    if(Q2_2): #put 0 instead of negative profit (for Q2.2)
        simulated_portfolios = SM.nullify_negative_profit(simulated_portfolios)
    # print(simulated_portfolios)

    """Q2.a (0% profit)"""
    target_return = 0.0
    target_prob_port = simulated_portfolios.apply(lambda x: np.mean(x == target_return), axis=1)
    probability = target_prob_port[-1:].values[0] * 100
    print(f"The probability for final profit = {target_return} is: {probability}")

    """Q2.b (2% profit)"""
    target_return = 0.02
    target_prob_port = simulated_portfolios.apply(lambda x: np.mean(x == target_return), axis=1)
    probability = target_prob_port[-1:].values[0] * 100
    print(f"The probability for final profit = {target_return} is: {probability}")

    """Q2.c (2%,20%]"""
    target_return = (0.02, 0.2)
    target_prob_port = simulated_portfolios.apply(lambda x: np.mean((target_return[0] < x) & (x <= target_return[1])), axis=1)
    probability = target_prob_port[-1:].values[0] * 100
    print(f"The probability for final profit is ({target_return[0]},{target_return[1]}]: {probability}")

    """Q2.d (20%,36%)"""
    target_return = (0.2, 0.36)
    target_prob_port = simulated_portfolios.apply(lambda x: np.mean((target_return[0] < x) & (x <= target_return[1])), axis=1)
    probability = target_prob_port[-1:].values[0] * 100
    print(f"The probability for final profit is ({target_return[0]},{target_return[1]}): {probability}")

    # percentile_10th = simulated_portfolios.cumsum().apply(lambda x: np.percentile(x, 10), axis=1)
    # percentile_90th = simulated_portfolios.cumsum().apply(lambda x: np.percentile(x, 90), axis=1)

    """confidence interval"""
    d = simulated_portfolios.iloc[-1]
    print(f"The average mean is: {np.mean(d)}")
    interval = st.norm.interval(alpha=0.90, loc=np.mean(d), scale=st.sem(d))
    print(f"The 90% confidence interval is : {interval}")
    # print(f"{percentile_10th[-1:].values[0]} to {percentile_90th[-1:].values[0]}")
    # average_port = simulated_portfolios.cumsum().apply(lambda x: np.mean(x), axis=1)
    # print(f'Average port: {average_port}\n percentile_90th: {percentile_90th}')
