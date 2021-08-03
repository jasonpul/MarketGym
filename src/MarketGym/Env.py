from . import StonkDataLib as SDL
from . import utils
from . import Types
from . import Constants
import pandas as pd
import random
from typing import Union
import numpy as np
import datetime


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class Env:
    def __init__(self, start_balance: float) -> None:
        self.lookback_period = 90
        self.episode_period = 90
        self.transaction_fee = 10
        self.transaction_threshold = 1e-4
        self.done_treashold = 100
        self.start_balance = start_balance
        self.start_date = datetime.datetime.now()

        self.reset()

    def reset(self):
        self.balance = self.start_balance * 1
        self.last_balance = self.balance * 1
        self.set_start_date()
        self.set_stock_frame()

    def set_start_date(self) -> None:
        date_list = SDL.get_historical_date_list()
        idx = random.choice(list(range(len(date_list)))[
                            self.lookback_period:-self.episode_period])

        self.date = date_list[idx]
        self.start_date = date_list[idx - self.lookback_period]
        self.end_date = date_list[idx + self.episode_period]

        self.date_list = date_list[(
            idx - self.lookback_period):(idx + self.episode_period + 1)]
        self.index = self.lookback_period

    def process_stock_frame(self, sf: Types.StockFrame) -> Types.StockFrame:

        def group_processing(subframe):
            close = subframe['close']
            low = subframe['low']
            high = subframe['high']
            subframe = subframe.sort_index()
            subframe['rsi'] = utils.rsi(close, 14)
            subframe['macd'] = utils.macd(close)
            subframe['cci'] = utils.cci(close, high, low, 20)
            subframe['adx'] = utils.cci(close, high, low, 14)

            return subframe

        sf = sf.groupby('symbol').apply(group_processing)
        sf = sf.sort_values(by=['timestamp', 'symbol'])
        sf = sf.fillna(0)

        return sf

    def initialize_weights(self) -> None:
        initial_weight = 1.0 / len(self.symbols)

        def set_shares(subframe: Types.StockFrame):
            subframe = subframe.sort_index()
            subframe['shares'] = initial_weight / subframe['close'][0]
            subframe['value'] = subframe['shares'] * subframe['close']
            return subframe

        def set_weight(subframe: Types.StockFrame):
            subframe['weight'] = subframe['value'] / subframe['value'].sum()
            return subframe

        sf = self.sf
        sf = sf.groupby('symbol').apply(set_shares)
        sf = sf.sort_values(by=['timestamp', 'symbol'])
        # sf = sf.droplevel('symbol').sort_values(by=['timestamp', 'symbol'])

        sf = sf.groupby('day').apply(set_weight)
        sf = sf.sort_values(by=['timestamp', 'symbol'])
        sf.loc[sf.index > self.date, 'weight'] = 0
        self.sf = sf.drop(columns=['shares', 'value'])

    def set_stock_frame(self) -> None:
        self.sf = SDL.get_historical(
            start=self.start_date, end=self.end_date)

        self.sf = self.sf.sort_values(by=['timestamp', 'symbol'])
        self.sf['day'] = self.sf.index.floor('d')
        self.symbols = self.sf['symbol'].unique()
        self.initialize_weights()
        assert len(self.sf) == len(self.symbols) * len(self.date_list)

    def get_viewable_stock_frame(self) -> Types.StockFrame:
        dates = self.date_list[(
            self.index - self.lookback_period): (self.index + 1)]
        sf = self.sf[self.sf.index.isin(dates)]
        return sf

    def get_state(self):
        '''
        returns a (m, n, b) sized array
        m dimension: symbol
        n dimension: lookback period
        b dimension: symbol attributs ['weight', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'cci', 'adx']
        '''
        sf = self.get_viewable_stock_frame()
        sf = self.process_stock_frame(sf)
        sf = sf.sort_values(['symbol', 'day'])

        group = sf.reset_index(drop=True).groupby(['symbol'])
        data_columns = ['weight'] + Constants.sf_data_columns + \
            ['rsi', 'macd', 'cci', 'adx']
        raw_data = sf[data_columns].values
        data = np.array([raw_data[i.values, :]
                        for k, i in group.groups.items()])

        for i in range(len(self.symbols)):
            data[i, :, 1:] = data[i, :, 1:] / data[i, -1, 1:]
        return data

    def step(self, action: np.ndarray):

        old_weights = self.sf.loc[self.sf.index == self.date, 'weight'].values
        old_price = self.sf.loc[self.sf.index == self.date, 'close'].values
        shares = old_weights / old_price

        # minimize transactions with a threashold
        idx = abs(action - old_weights) <= self.transaction_threshold
        transaction_fee = sum(idx == False) * self.transaction_fee
        action[idx] = old_weights[idx]
        max_weight = action[idx == False].max()
        action[action == max_weight] += 1 - action.sum()

        self.index += 1
        self.date = self.date_list[self.index]
        idx = self.sf.index == self.date

        new_price = self.sf.loc[idx, 'close'].values
        new_balance = self.balance * \
            (shares * new_price).sum() - transaction_fee
        self.sf.loc[idx, 'weight'] = action

        reward = new_balance - self.balance
        self.balance = new_balance

        state = self.get_state()
        done = False
        if self.index == (len(self.date_list) - 1) or self.balance <= self.done_treashold:
            done = True

        print(self.balance)
        return state, reward, done

    # def backfill_stock_frame(self) -> None:
    #     self.sf['day'] = self.sf.index.floor('d')
    #     group = self.sf.groupby(['symbol'])
    #     counts = group.agg(['count'])
    #     n = counts['close'].max()

    #     idx = (counts['close'] != n).to_numpy().squeeze()
    #     short_symbols = counts.index[idx]
    #     full_symbols = counts.index[idx == False]
    #     full_set = self.sf[self.sf['symbol'] == full_symbols[0]]

    #     short_sets = []
    #     for symbol in short_symbols:
    #         sf = self.sf[self.sf['symbol'] == symbol].iloc[0]
    #         short_set = full_set[full_set['day'] < sf.day].copy()
    #         short_set['symbol'] = symbol
    #         short_set['open'] = sf['open']
    #         short_set['high'] = sf['high']
    #         short_set['low'] = sf['low']
    #         short_set['close'] = sf['close']
    #         short_set['volume'] = sf['volume']
    #         short_sets.append(short_set)
    #     self.sf = pd.concat([self.sf] + short_sets)
    #     self.sf = self.sf.drop(columns=['day'])
    #     self.sf = self.sf.sort_values(by=['timestamp', 'symbol'])
