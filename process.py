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
from .utils import TMP_PATH
from .data import fetch_stock_base, fetch_trade_date, fetch_history, fetch_em_feature, fetch_xq_feature


logger = logging.getLogger(__name__)


def create_frame_by_dates(dates, symbols=None, tmp=TMP_PATH, update=False):
    ret = []
    for date in dates:
        feature_filename = os.path.join(tmp, 'features', '{}.csv'.format(date))
        feature_frame = create_feature_frame_by_date(date, symbols=symbols, filename=feature_filename, update=update)
        label_filename = os.path.join(tmp, 'labels', '{}.csv'.format(date))
        label_frame = create_label_frame_by_date(date, symbols=symbols, filename=label_filename, update=update)
        ret.append(pd.merge(feature_frame, label_frame, how='left', on='symbol').dropna())
    return ret


def create_feature_frame_by_date(date, symbols=None, filename=None, update=False):
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
    return df.loc[df['symbol'].isin(symbols.tolist()), :] if not symbols.empty else df


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
            history_frame['change'] = (history_frame['close_2'] - history_frame['close_0']) / history_frame['close_0'] * 100
            history_frame['label'] = history_frame.apply(lambda s: (1 if s['change'] > 0 else 0), axis=1)
            df = history_frame[['symbol', 'change', 'label']]
            if filename:
                dirname = os.path.dirname(filename)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                df.to_csv(filename, index=False)
        except IndexError:
            logger.warning('There have no label on \'{}\''.format(date))
            raise
    return df.loc[df['symbol'].isin(symbols)] if not symbols.empty else df


def _create_base_frame(symbols):
    '''
        return pd.DataFrame
    '''
    return pd.DataFrame({'symbol': symbols})
