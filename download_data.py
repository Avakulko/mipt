from tqdm import tqdm
import quandl
quandl.ApiConfig.api_key = "SR7LxL3vRbxu8zibz6Sy"

# Download data from Quandl

# DESCRIPTION
# Historical Futures Prices: Crude Oil Futures, Continuous Contract #1.
# Non-adjusted price based on spot-month continuous contract calculations.
# Raw data from CME.
# https://www.quandl.com/data/CHRIS/CME_CL1-Crude-Oil-Futures-Continuous-Contract-1-CL1-Front-Month

def download_data():
    print('Downloading data from Quandl...')
    for i in tqdm(range(1, 39)):
        symbol = f'CME_CL{i}'
        quandl.get(f"CHRIS/{symbol}").to_csv(f"Data/{symbol}.csv")
    print('Done')
