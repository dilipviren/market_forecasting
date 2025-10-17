import json
import pandas as pd
import numpy as np
import os
import sys
import yaml
import certifi
import tools

try:
    from urllib.request import urlopen
except ImportError:
    print('Error: Could not import urlopen from urllib.request')

class DataFetch:
    """
    A class to fetch historical market data for a given symbol using configuration parameters.
    Attributes
    ----------
    config_path : str
        Path to the YAML configuration file containing API keys and request URLs.
    symbol : str
        The market symbol (e.g., stock ticker) to fetch data for.
    is_stock : bool
        Flag indicating whether the symbol is a stock (default: True).
    from_date : str
        Start date for fetching historical data in 'YYYY-MM-DD' format.
    to_date : str
        End date for fetching historical data in 'YYYY-MM-DD' format.
    Methods
    -------
    fetch_data():
        Fetches historical data for the specified symbol using the configuration file.
        Returns a tuple containing the symbol and a pandas DataFrame of historical data.
        If an error occurs or the symbol is not provided, returns an empty string and empty DataFrame.
    """

    def __init__(self, config_path: str=None, symbol: str = None, is_stock: bool = True, from_date: str = '2021-01-01', to_date: str = '2024-12-31'):
        self.config_path = config_path
        self.symbol = symbol
        self.is_stock = is_stock
        self.from_date = from_date
        self.to_date = to_date

    def fetch_data(self):

        try:
            if self.config_path:
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
            else:
                config = tools.GetConfig.get_config()
        except Exception:
            config = tools.GetConfig.get_config()

        from_date = self.from_date
        to_date = self.to_date

        if self.is_stock and self.symbol!=None:
            api_key = config['keys']['stock_key']
            url = config['requests']['eod_url']
            url = url.replace('{symbol}', self.symbol).replace('{api_key}', api_key).replace('{fromdate}', from_date).replace('{todate}', to_date)
        else:
            print("Error: Currently only stock data fetching is supported with a valid symbol.")
            return '', pd.DataFrame()

        try:
            response = urlopen(url, cafile=certifi.where())
            data = response.read().decode("utf-8")
            json_data = json.loads(data)

            if isinstance(json_data, dict):
                symbol = json_data['symbol']
                data_frame = pd.DataFrame(json_data['historical'])
                return symbol, data_frame
            
            if isinstance(json_data,list):
                return '', pd.DataFrame(json_data)
            
        except Exception as e:
            print(f"An error occurred: {e}, url {url} may be invalid")
            return '', pd.DataFrame()
    

if __name__ == "__main__":

    symbol, data = DataFetch(symbol='AAPL', is_stock=True).fetch_data()

    print(f"Fetched data for symbol: {symbol}")
    print(data)
