import pandas as pd
# pd.set_option('display.max_columns', None)
import numpy as np


class Indicator:
    '''A class for calculating various technical indicators on a given DataFrame.

    Attributes:
        df (DataFrame): The input DataFrame containing the financial data.
        td (bool): A flag indicating whether the indicator is trend deterministic or not.
        imp_features (None): Placeholder for storing important features.
        k (None): Placeholder for storing stochastic k% values.
        inter (DataFrame): Intermediate DataFrame for storing temporary calculations.

    Methods:
        view_data(n=10): Prints the first n rows of the DataFrame.
        drop_cols(colname): Drops the specified column from the DataFrame.
        view_cols(): Returns the column names of the DataFrame.
        set_colnames(col_names): Sets the column names of the DataFrame.
        make_up_down(): Creates a column 'movement' indicating the up or down signals from the 'Close' column.
        get_df(): Returns the DataFrame.
        index_reset(): Resets the index of the DataFrame.
        sma(n=5): Calculates the simple moving averages of the 'Close' column and adds a new column to the DataFrame.
        wma(n=5): Calculates the weighted moving averages of the 'Close' column and adds a new column to the DataFrame.
        momentum(n=7): Calculates the momentum column for a given period and adds a new column to the DataFrame.
        stochastic_k(period=14): Calculates the stochastic k% for a given period and adds a new column to the DataFrame.
        stochastic_d(period=3): Calculates the stochastic d% using the previously calculated k% and adds a new column to the DataFrame.
        rsi(period=14): Calculates the relative strength index (RSI) for a given period and adds a new column to the DataFrame.
        stochatic_r(period=14): Calculates Larry Williams' R% oscillator for a given period and adds a new column to the DataFrame.
        ad(): Calculates the accumulation/distribution oscillator and adds a new column to the DataFrame.
        cci(period=20): Calculates the commodity channel index (CCI) for a given period and adds a new column to the DataFrame.
        '''

    def __init__(self, df, td=False):
        self.df = df
        self.td = td
        self.imp_features = None
        self.k = None
        self.inter = pd.DataFrame()

    def view_data(self, n=10):
        print(self.df.head(n))

    def drop_cols(self,colname):
        self.df.drop(colname,axis=1,inplace=True)

    def view_cols(self):
        return self.df.columns

    def set_colnames(self, col_names):
        self.df.columns = col_names

    def make_up_down(self):
        """
        creates column movement which contains the up or down signals from Close column
        :return: None
        """
        self.df['movement'] = np.sign(self.df['Close'].diff().fillna(0))

    def get_df(self):
        return self.df

    def index_reset(self):
        self.df.reset_index(inplace=True)

    # ////////////////////////////////// Methods for calculating the indicators ///////////////////////////////////////

    def sma(self, n=5):
        """
        method adds a column to self containing simple moving averages of Close column
        :param n: window length
        :return: None
        """
        if not self.td:
            self.df[f'sma_{n}'] = self.df['Close'].rolling(window=n, min_periods=1).mean()
        else:
            self.df[f'sma_{n}_td'] = np.sign(self.df['Close'] - self.df['Close'].rolling(window=n, min_periods=1).mean())

    def wma(self, n=5):
        denom = n*(n+1)/2
        closes = list(self.df['Close'])
        means = []
        for i in range(len(closes)):
            if n < i < len(closes)-n:
                mean = closes[i]
            else:
                mean = sum(closes[i-n:i])/denom
            means.append(mean)
        if not self.td:
            self.df[f'wma_{n}'] = means
        else:
            means.insert(0, 0)
            means = np.array(means)
            self.df[f'wma_{n}_td'] = np.sign(np.diff(means))

    def momentum(self, n=7):
        """
        Calculates the momentum column for a period n for the Close column
        :param n: period length
        :return: None
        """
        if not self.td:
            self.df[f'momentum_{n}'] = self.df['Close'] - self.df['Close'].shift(n).fillna(0)
        else:
            temp = (self.df['Close'] - self.df['Close'].shift(n).fillna(0))
            self.df[f'momentum_{n}_td'] = [1 if i > 0 else -1 if i < 0 else 0 for i in temp]
            del temp

    def stochastic_k(self, period=14):
        """
        Calculates stochastic k% for period 14; creates duplicate attribute self.k for d%
        :param period: the period to take LL and HH
        :return: None
        """
        Lowest_Low = self.df['Low'].rolling(window=period, min_periods=7).min()
        Highest_High = self.df['High'].rolling(window=period, min_periods=7).max()
        self.k = 100 * ((self.df['Close'] - Lowest_Low) / (Highest_High - Lowest_Low))
        if not self.td:
            self.df[f'k_{period}'] = self.k.fillna(0)
        else:
            self.df[f'k_{period}_td'] = np.sign(self.k.diff().fillna(0))
        del Highest_High, Lowest_Low

    def stochastic_d(self, period=3):
        """
        Calculates stochastic d% using self.k; Always run after k%; deletes self.k
        :param period: int. period to take MA
        :return: None
        """
        if not self.td:
            self.df[f'd_{period}'] = self.k.rolling(window=period).mean().fillna(0)
        else:
            self.df[f'd_{period}_td'] = np.sign(self.k.rolling(window=period).mean().diff().fillna(0))

        del self.k

    def rsi(self, period=14):
        """
        Calculates RSI for period 14; deletes local vars
        :param period: int, time period for gains, losses
        :return: None
        """
        delta = self.df['Close'].diff()
        rs = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean() / (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        if not self.td:
            self.df[f'rsi_{period}'] = 100 - (100/(1 + rs))
            self.df[f'rsi_{period}'] = self.df[f'rsi_{period}'].fillna(0)
        else:
            rsi = (100 - (100/(1 + rs)))
            vals = []
            for i, value in enumerate(rsi):
                if value <= 30:
                    vals.append(1)
                elif value >= 70:
                    vals.append(-1)
                else:
                    if i > 0:
                        if value > rsi[i-1]:
                            vals.append(-1)
                        elif value < rsi[i-1]:
                            vals.append(1)
                        elif value == rsi[i-1]:
                            vals.append(0)
                    else:
                        vals.append(1)
            self.df[f'rsi_{period}_td'] = vals
            self.df[f'rsi_{period}_td'] = self.df[f'rsi_{period}_td'].fillna(0)
        del rs, delta

    def stochatic_r(self, period=14):
        """
        Calculates Larry Williams' R% oscillator
        :param period:
        :return:
        """
        Lowest_Low = self.df['Low'].rolling(window=period,min_periods=7).min()
        Highest_High = self.df['High'].rolling(window=period,min_periods=7).max()
        if not self.td:
            self.df[f'r_{period}'] = (100*(Highest_High - self.df['Close'])/(Highest_High-Lowest_Low)).fillna(0)
        else:
            self.df[f'r_{period}_td'] = np.sign(((Highest_High - self.df['Close'])/(Highest_High-Lowest_Low)).diff().fillna(0))

    def ad(self):
        """
        calculates the accumulation/distribution oscillator
        :return: None
        """
        mfv = self.df['Volume'] * (((2*self.df['Close']) - self.df['High'] - self.df['Low']) / ((self.df['High'] - self.df['Low'])))
        if not self.td:
            self.df['ad'] = mfv.cumsum()
        else:
            self.df['ad_td'] = np.sign(mfv.cumsum().diff().fillna(0))

    def cci(self, period=20):
        """
        calculates the commodity channel index
        :return: None
        """
        temp = np.array([(x+y+z)/3 for x,y,z in zip(self.df['Close'], self.df['High'], self.df['Low'])])
        tempdf = pd.DataFrame(temp,columns=['tempvar'])
        sma = tempdf['tempvar'].rolling(window=period, min_periods=1).mean()
        dev = [abs(x-y) for x,y in zip(temp,sma)]
        means = []
        for i in range(len(dev)):
            if 10 <= i <= len(dev)-10:
                mean = sum(dev[i-10:i+11])/21
            else:
                mean = dev[i]
            means.append(mean)
        ccis = [(x-y)/(0.015*z) for x,y,z in zip(temp,sma,means)]
        if not self.td:
            self.df[f'cci_{period}'] = ccis
        else:
            vals = []
            for i, values in enumerate(ccis):
                if values >= 200:
                    vals.append(-1)
                elif values <= -200:
                    vals.append(1)
                else:
                    if i > 0:
                        if values > ccis[i - 1]:
                            vals.append(-1)
                        elif values < ccis[i - 1]:
                            vals.append(1)
                        else:
                            vals.append(0)
                    else:
                        vals.append(1)
            self.df[f'cci_{period}_td'] = vals
            del vals
        del temp, tempdf, sma, dev, means, ccis









