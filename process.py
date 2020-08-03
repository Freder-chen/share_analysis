# -*- coding: utf-8 -*-
"""
process data
Created on 2020/06/22
@author: Freder Chen
@contact: freder-chen@qq.com
"""

import os
import logging
import pandas as pd
from tqdm import tqdm
from .utils import TMP_PATH
from .data import (
    fetch_stock_base, fetch_trade_date, fetch_history,
    fetch_daily, fetch_em_feature, fetch_xq_feature
)


logger = logging.getLogger(__name__)


def create_frame_by_dates(dates, symbols=None, tmp=TMP_PATH, update=False):
    ret = []
    for date in dates:
        features_filename = os.path.join(tmp, 'features', '{}.csv'.format(date))
        label_filename = os.path.join(tmp, 'labels', '{}.csv'.format(date))
        features_frame = create_features_frame_by_date(date, symbols=symbols, filename=features_filename, update=update)
        label_frame = create_label_frame_by_date(date, symbols=symbols, filename=label_filename, update=update)
        ret.append(pd.merge(features_frame, label_frame, how='left', on='symbol').dropna())
    return ret


def create_label_frame_by_date(date, symbols=None, filename=None, update=False):
    trade_dates = fetch_trade_date(start=date, n=3)
    if (not update) and filename and os.path.exists(filename):
        df = pd.read_csv(filename, dtype={'symbol': 'str'})
    else:
        try:
            base_frame = _create_base_frame(fetch_stock_base()['symbol'].tolist())
            history_frame = pd.merge(base_frame, fetch_history(trade_dates[0])[['symbol', 'close']].rename(
                columns={'close':'close_0'}), how='left', on='symbol').dropna()
            history_frame = pd.merge(history_frame, fetch_history(trade_dates[2])[['symbol', 'close']].rename(
                columns={'close':'close_2'}), how='left', on='symbol').dropna()
            history_frame['label_change'] = (history_frame['close_2'] - history_frame['close_0']) / history_frame['close_0'] * 100
            history_frame['label'] = history_frame.apply(lambda s: (1 if s['label_change'] > 0 else 0), axis=1)
            df = history_frame[['symbol', 'label_change', 'label']]
            if filename:
                dirname = os.path.dirname(filename)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                df.to_csv(filename, index=False)
        except IndexError:
            logger.warning('There have no label on \'{}\''.format(date))
            raise
    return df.loc[df['symbol'].isin(symbols.tolist()), :] if symbols is not None else df


def create_features_frame_by_date(date, symbols=None, filename=None, tmp=TMP_PATH, update=False):
    if (not update) and filename and os.path.exists(filename):
        df = pd.read_csv(filename, dtype={'symbol': 'str'})
    else:
        origin_features_filename = os.path.join(tmp, 'origin_features', '{}.csv'.format(date))
        pattern_features_filename = os.path.join(tmp, 'pattern_features', '{}.csv'.format(date))
        base_frame = _create_base_frame(fetch_stock_base()['symbol'].tolist())
        origin_features_frame = create_origin_features_frame_by_date(date, filename=origin_features_filename)
        pattern_features_frame = create_pattern_features_frame_by_date(date, filename=pattern_features_filename)
        df = pd.merge(origin_features_frame, pattern_features_frame, how='left', on='symbol').dropna()
        if filename:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            df.to_csv(filename, index=False)
    return df.loc[df['symbol'].isin(symbols.tolist()), :] if symbols is not None else df


def create_origin_features_frame_by_date(date, symbols=None, filename=None, update=False):
    if (not update) and filename and os.path.exists(filename):
        df = pd.read_csv(filename, dtype={'symbol': 'str'})
    else:
        base_frame = _create_base_frame(fetch_stock_base()['symbol'].tolist())
        em_frame = fetch_em_feature(date)
        xq_frame = fetch_xq_feature(date)[['symbol', 'current', 'xq_followers']]
        feature_frame = pd.merge(em_frame, xq_frame, how='left', on='symbol').dropna()
        df = pd.merge(base_frame, feature_frame, how='left', on='symbol').dropna()
        if filename:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            df.to_csv(filename, index=False)
    return df.loc[df['symbol'].isin(symbols.tolist()), :] if symbols is not None else df


def create_pattern_features_frame_by_date(date, ns=[5, 20, 60, 120], symbols=None, filename=None, update=False):
    def _apply_ma(row, ma_u, ma_l):
        try:
            return (row[ma_u] - row[ma_l]) / row[ma_u]
        except ZeroDivisionError:
            return 0

    def _aplly_ma20_5(row):
        return _apply_ma(row, 'ma20', 'ma5')

    def _aplly_ma60_20(row):
        return _apply_ma(row, 'ma60', 'ma20')

    def _aplly_ma120_60(row):
        return _apply_ma(row, 'ma120', 'ma60')

    if (not update) and filename and os.path.exists(filename):
        df = pd.read_csv(filename, dtype={'symbol': 'str'})
    else:
        ma_frame = _create_ma_features_frame_by_date(date, ns)
        ma_frame['ma20-5'] = ma_frame.apply(_aplly_ma20_5, axis=1)
        ma_frame['ma60-20'] = ma_frame.apply(_aplly_ma60_20, axis=1)
        ma_frame['ma120-60'] = ma_frame.apply(_aplly_ma120_60, axis=1)
        good_k_frame = _create_goodk_features_frame_by_date(date)
        df = pd.merge(ma_frame, good_k_frame, how='left', on='symbol').dropna()
        if filename:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            df.to_csv(filename, index=False)
    return df.loc[df['symbol'].isin(symbols.tolist()), :] if symbols is not None else df


def _create_ma_features_frame_by_date(date, ns, symbols=None):
    """
        return frame format 'ma{n}' columns
    """
    ma_date = '{}-{}-{}'.format(date[:4], date[4:6], date[6:]) # request need

    ma_l = []
    for s in tqdm(fetch_stock_base()['symbol'].tolist(), desc=f'MA {date}', leave=False, mininterval=1):
        try:
            daily_frame = fetch_daily(s)
            for n in ns: daily_frame[f'ma{n}'] = daily_frame['close'].rolling(n).mean() # compute ma
            mas = daily_frame.loc[(daily_frame['trade_date'] == ma_date), [f'ma{n}' for n in ns]].values[0].tolist()
            ma_l.append([s, *mas])
        except (KeyError, IndexError):
            # print(daily_frame)
            ma_l.append([s, *[0 for n in ns]])
    df = pd.DataFrame(ma_l, columns=['symbol', *[f'ma{n}' for n in ns]])
    return df.loc[df['symbol'].isin(symbols.tolist()), :] if symbols is not None else df


def _create_goodk_features_frame_by_date(date, symbols=None):
    def _aplly_line(row, col):
        try:
            return abs(row[col] - max(row['open'], row['close'])) / (row['close'] - row['open'])
        except ZeroDivisionError:
            return 0

    def _apply_up_line(row):
        return _aplly_line(row, col='high')

    def _apply_bottem_line(row):
        return _aplly_line(row, col='low')

    def _apply_line_rate(row):
        try:
            return row['up_line_rate'] / row['bottem_line_rate']
        except ZeroDivisionError:
            return 0
    
    history_frame = fetch_history(date)
    history_frame['up_line_rate'] = history_frame.apply(_apply_up_line, axis=1)
    history_frame['bottem_line_rate'] = history_frame.apply(_apply_bottem_line, axis=1)
    history_frame['line_rate'] = history_frame.apply(_apply_line_rate, axis=1)
    cols = ['symbol', 'up_line_rate', 'bottem_line_rate', 'line_rate', 'change']
    return history_frame.loc[history_frame['symbol'].isin(symbols.tolist()), cols] if symbols is not None else history_frame[cols]


def _create_base_frame(symbols):
    '''
        return pd.DataFrame
    '''
    return pd.DataFrame({'symbol': symbols})
