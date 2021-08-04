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
    def __init__(self, start_balance: float, min_date: Union[datetime.datetime, None] = None, max_date: Union[datetime.datetime, None] = None) -> None:
        self.lookback_period = 90
        self.episode_period = 90
        # self.lookback_period = 25
        # self.episode_period = 22
        self.trade_period = 3
        self.transaction_fee = 0
        self.transaction_threshold = 1e-4
        self.done_treashold = 100
        self.start_balance = start_balance
        self.min_date, self.max_date = min_date, max_date

        self.reset()

    def reset(self):
        self.balance = self.start_balance * 1
        self.last_balance = self.balance * 1
        self.set_start_date()
        self.set_stock_frame()
        state = self.get_state()
        return state

    def set_start_date(self) -> None:
        date_list = SDL.get_historical_date_list()
        if self.min_date is not None:
            date_list = [i for i in date_list if i.date() >=
                         self.min_date.date()]

        if self.max_date is not None:
            date_list = [i for i in date_list if i.date() <=
                         self.max_date.date()]

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

            return subframe

        sf = sf.groupby('symbol').apply(group_processing)
        sf = sf.sort_values(by=['timestamp', 'symbol'])
        sf = sf.fillna(0)

        return sf

    def initialize_shares(self) -> None:
        initial_weight = 1.0 / len(self.symbols)

        def set_shares(subframe: Types.StockFrame):
            subframe = subframe.sort_index()
            subframe['shares'] = initial_weight / subframe['close'][0]
            subframe['value'] = subframe['shares'] * subframe['close']
            return subframe

        sf = self.sf
        sf = sf.groupby('symbol').apply(set_shares)
        sf = sf.sort_values(by=['timestamp', 'symbol'])

        sf.loc[sf.index > self.date, ['shares']] = 0
        self.sf = sf.drop(columns=['value'])

    def set_stock_frame(self) -> None:
        self.sf = SDL.get_historical(
            start=self.start_date, end=self.end_date)

        self.sf = self.sf.sort_values(by=['timestamp', 'symbol'])
        self.sf['day'] = self.sf.index.floor('d')
        self.symbols = self.sf['symbol'].unique()

        self.initialize_shares()
        self.sf = self.process_stock_frame(self.sf)
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
        sf = sf.sort_values(['symbol', 'day'])

        group = sf.reset_index(drop=True).groupby(['symbol'])
        data_columns = ['shares'] + Constants.sf_data_columns + \
            ['rsi', 'macd', 'cci']
        raw_data = sf[data_columns].values
        data = np.array([raw_data[i.values, :]
                        for k, i in group.groups.items()])

        # normalize data
        data_min = np.repeat(
            data.min(axis=1)[:, :, np.newaxis], data.shape[1], axis=2).swapaxes(2, 1)
        data_max = np.repeat(
            data.max(axis=1)[:, :, np.newaxis], data.shape[1], axis=2).swapaxes(2, 1)
        scaled = (data - data_min) / (data_max - data_min + 1e-16)

        return scaled

    def step(self, action: np.ndarray):
        def backfill_shares(subframe):
            subframe = subframe.sort_index()
            subframe_slice = subframe[subframe.index <= self.date]
            idx = subframe_slice['shares'] == 0
            last_shares = subframe_slice.loc[idx == False, 'shares'][-1]
            subframe_slice.loc[idx, 'shares'] = last_shares
            subframe[subframe.index <= self.date] = subframe_slice
            return subframe

        # get old values
        old_price = self.sf.loc[self.sf.index == self.date, 'close'].values
        old_shares = self.sf.loc[self.sf.index == self.date, 'shares'].values

        # advance date
        done = False
        if (self.index + self.trade_period * 2) >= len(self.date_list):
            self.index = len(self.date_list) - 1
            done = True
        else:
            self.index += self.trade_period
        self.date = self.date_list[self.index]

        # calculate new shares from given weights (action)
        new_price = self.sf.loc[self.sf.index == self.date, 'close'].values
        new_shares = action / new_price
        self.sf.loc[self.sf.index == self.date, 'shares'] = new_shares

        # calculate delta-shares for transaction free
        idx = abs(old_shares - new_shares) <= self.transaction_threshold
        transaction_fee = sum(idx == False) * self.transaction_fee

        # calculate reward using new balance
        new_balance = self.balance * \
            (1 + (old_shares * (new_price - old_price)).sum()) - transaction_fee
        reward = new_balance - self.balance
        self.balance = new_balance

        # backfill shares if necessary
        if self.trade_period > 1:
            sf = self.sf
            sf = sf.groupby('symbol').apply(backfill_shares)
            sf = sf.sort_values(by=['timestamp', 'symbol'])
            self.sf = sf

        # get new state
        state = self.get_state()

        # check if done
        if self.balance <= self.done_treashold:
            done = True
            reward = -1e4

        return state, reward, done
