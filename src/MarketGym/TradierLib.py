import datetime
import os
import requests
import pandas as pd
import time
import random
from tqdm import tqdm
from multiprocessing import Pool
from dotenv import load_dotenv
from typing import Union, List
from . import Types
load_dotenv()


tradier_url = 'https://api.tradier.com'


def load_env_file(path: str):
    load_dotenv(path)


def _get_historical(symbol: Types.Symbol, start: Union[None, Types.Datetime] = None, end: Union[None, Types.Datetime] = None, rate_limit: bool = False):
    def process_frame(sf: Types.StockFrame, symbol: Types.Symbol):
        sf = sf.dropna()
        sf = sf.rename(columns={'date': 'timestamp'})
        sf['symbol'] = symbol.upper()
        sf['timestamp'] = pd.to_datetime(sf['timestamp'], utc=True)
        sf['timestamp'] = sf['timestamp'].dt.tz_convert('US/Eastern')
        sf['timestamp'] = sf['timestamp'].apply(lambda x: (
            x + datetime.timedelta(hours=12)).replace(hour=16, minute=30))
        sf = sf.drop_duplicates(subset=['timestamp'])
        sf = sf.set_index('timestamp')
        return sf

    if rate_limit:
        time.sleep(random.randrange(1, 20) / 10.0)

    url = '%s/v1/markets/history' % tradier_url
    params = {
        'symbol': symbol.upper(),
        'interval': 'daily',
    }
    if start is not None:
        params['start'] = start.strftime('%Y-%m-%d')
    if end is not None:
        params['end'] = end.strftime('%Y-%m-%d')

    headers = {
        'Authorization': 'Bearer %s' % os.getenv('TRADIER_TOKEN'),
        'Accept': 'application/json'
    }

    response = requests.get(url, params=params, headers=headers)
    try:
        json_response = response.json()
        sf: Types.StockFrame = pd.DataFrame(json_response['history']['day'])
        sf = process_frame(sf, symbol)
        return sf
    except:
        print('%s from %s to %s failed' %
              (symbol, params['start'], params['end']))
        return None


def get_historical(symbols: List[Types.Symbol], start: Union[None, Types.Datetime] = None, end: Union[None, Types.Datetime] = None):
    jobs = []
    pool = Pool(processes=10)

    for symbol in symbols:
        jobs.append(pool.apply_async(func=_get_historical,
                    args=(symbol, start, end, True)))
    pool.close()
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    return [i for i in results if i is not None]
