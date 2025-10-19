import json
import pandas as pd
import numpy as np
import os
import sys
import yaml
import certifi
import tools
from dotenv import load_dotenv

try:
    from urllib.request import urlopen
except ImportError:
    print('Error: Could not import urlopen from urllib.request')


class DataImporter:
    """
    A class for importing financial market data from external APIs.
    This class provides functionality to import historical market data for a given portfolio
    of symbols between specified date ranges using configured API endpoints.
    Attributes
    ----------
    None
    Methods
    -------
    import_data(portfolio_df: pd.DataFrame, data_type: str = 'eod_url') -> pd.DataFrame
        Imports historical market data for symbols specified in the portfolio dataframe.
    Notes
    -----
    - Requires environment variable 'API_KEY' to be set
    - Uses configuration file for API endpoints
    - Handles both dictionary and list response formats from APIs
    """

    @staticmethod
    def import_data(portfolio_df: pd.DataFrame, data_type:str='eod_url') -> pd.DataFrame:
        """
        Imports historical market data for a given portfolio of symbols.
        Args:
            portfolio_df (pd.DataFrame): A DataFrame containing the portfolio information with columns 
                                          'symbol', 'start_date', and 'end_date'.
            data_type (str): The type of data to fetch, default is 'eod_url'. This should correspond 
                             to a key in the configuration for the API request.
        Returns:
            pd.DataFrame: A DataFrame containing the concatenated historical data for all symbols 
                          in the portfolio.
        Raises:
            Exception: If an error occurs while fetching data for any symbol, an error message will 
                       be printed, but the function will continue to attempt to fetch data for 
                       remaining symbols.
        """

        final_data = pd.DataFrame()
        config = tools.GetConfig.get_config()
        load_dotenv()
        api_key = os.getenv('API_KEY')
        for index, row in portfolio_df.iterrows():
            symbol = row['symbol']
            from_date = row['start_date']
            to_date = row['end_date']
            url = config['requests'][data_type]
            # print(f"Fetching data for {symbol} from {from_date} to {to_date}, api_key {api_key}")
            url = url.replace('{symbol}', symbol).replace('{api_key}', api_key).replace('{fromdate}', from_date).replace('{todate}', to_date)
            try:
                response = urlopen(url, cafile=certifi.where())
                data = response.read().decode("utf-8")
                json_data = json.loads(data)
                if isinstance(json_data, dict):
                    data_frame = pd.DataFrame(json_data['historical'])
                    # print(f"Data for {symbol}:")
                    # print(data_frame.head())
                elif isinstance(json_data, list):
                    data_frame = pd.DataFrame(json_data)
                    # print(f"Data for {symbol}:")
                    # print(data_frame.head())
            except Exception as e:
                print(f"An error occurred while fetching data for {symbol}: {e}")
            final_data = pd.concat([final_data, data_frame], ignore_index=True)

        return final_data


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

        load_dotenv()
        api_key = os.getenv('API_KEY')

        if self.is_stock and self.symbol!=None:
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

    def fetch_data_multiple(self, portfolio_df: pd.DataFrame):
        """
        Fetches financial data for multiple symbols defined in a portfolio DataFrame.
        This method iterates through a portfolio DataFrame and fetches historical data for each symbol,
        storing the results in a dictionary.
        Parameters
        ----------
        portfolio_df : pd.DataFrame
            DataFrame containing portfolio information with at least a 'symbol' column and optionally
            a 'type' column to specify if the symbol is a stock (default) or other security type.
        Returns
        -------
        dict
            Dictionary where keys are symbols and values are pandas DataFrames containing
            the historical financial data for each symbol. Empty DataFrames are excluded.
        Notes
        -----
        The method uses the instance's fetch_data() method for each symbol and updates
        the instance's symbol and is_stock attributes during iteration.
        """

        all_data = {}
        for index, row in portfolio_df.iterrows():
            symbol = row['symbol']
            is_stock = True if row.get('type','stock').lower() == 'stock' else False
            self.symbol = symbol
            self.is_stock = is_stock
            self.from_date = row.get('start_date', self.from_date)
            self.to_date = row.get('end_date', self.to_date)
            fetched_symbol, data_frame = self.fetch_data(symbol=self.symbol, is_stock=self.is_stock, from_date=self.from_date, to_date=self.to_date)
            if not data_frame.empty:
                all_data[fetched_symbol] = data_frame
        return all_data

if __name__ == "__main__":

    symbol, data = DataFetch(symbol='AAPL', is_stock=True).fetch_data()

    print(f"Fetched data for symbol: {symbol}")
    print(data)
