import json
import pandas as pd
import numpy as np
import os
import sys
import yaml
import certifi
from system_check import check_system

try:
    from urllib.request import urlopen
except ImportError:
    print('Error: Could not import urlopen from urllib.request')   

def query_data(config_path: str=None, key_name: str=None, data_freq: str=None, fromdate: str=None, todate: str=None, symbol: str = None):
    """
    Queries financial market data from a specified API using configuration parameters.
    Parameters:
        config_path (str): Path to the YAML configuration file containing API keys and request URLs.
        key_name (str): The key in the config file to retrieve the API key.
        data_freq (str): The type of data to request (e.g., 'forex', 'forex_list', 'forex_light').
        fromdate (str, optional): The start date for the data query (format: 'YYYY-MM-DD').
        todate (str, optional): The end date for the data query (format: 'YYYY-MM-DD').
        symbol (str, optional): The symbol or currency pair to query.
    Returns:
        tuple:
            - symbol (str): The symbol for which data was retrieved (empty string if not applicable).
            - data_frame (pd.DataFrame): DataFrame containing the historical or list data.
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If required keys are missing in the configuration.
        urllib.error.URLError: If the API request fails.
        json.JSONDecodeError: If the response cannot be parsed as JSON.
    Notes:
        - The function expects the configuration file to be in YAML format with specific structure for keys and requests.
        - The returned DataFrame structure depends on the API response and the requested data frequency.
    """

    with open(config_path,'r') as file:
        config = yaml.safe_load(file)

    final_urls = []
    api_key = config['keys'][key_name]

    base_url = config['requests'][data_freq]


    if data_freq == 'forex_list':
        # base_url = config['requests'][data_freq]
        final_url = base_url.format(key=api_key)

    elif data_freq == 'forex':
        # base_url = config['requests'][data_freq]
        final_url = base_url.format(currencies=symbol, key=api_key)

    elif data_freq == 'forex_light':
        # base_url = config['requests'][data_freq]
        final_url = base_url.format(currencies=symbol, from_date=fromdate, to_date=todate, key=api_key)

    else:
        final_url = base_url.format(symbol=symbol,key=api_key,from_date=fromdate,to_date=todate)

        # if type(ticker_name)==list:
        #     for i in ticker_name:
        #         # symbol = config['tickers'][i]
        #         final_url = base_url.format(symbol=symbol,key=api_key,from_date=fromdate,to_date=todate)
        #         final_urls.append(final_url)

    final_urls.append(final_url)

    if len(final_urls)==1:
        final_url = final_urls[0]
    
    response = urlopen(final_url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    json_data = json.loads(data)

    if isinstance(json_data, dict):
        symbol = json_data['symbol']
        data_frame = pd.DataFrame(json_data['historical'])
        return symbol, data_frame
    
    if isinstance(json_data,list):
        return '', pd.DataFrame(json_data)


if __name__ == "__main__":

    # Check the system type to ensure correct paths
    if check_system() == 'macOS':
        config_path = '/Users/macbook/Mirror/trading_algorithm/config/config_file.yml'

    else:
        config_path = 'C:\\Users\\viren\\DSML projects\\trading_alg\\config\\config_file.yml'

    # Define parameters
    fromdate = '2024-11-04'
    todate = '2025-02-05'

    final_url = query_data(config_path=config_path, key_name='stock_key',data_freq='forex_light', ticker_name=['apple'], fromdate=fromdate, todate=todate, symbol='EURUSD')

