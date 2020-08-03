# -*- coding: utf-8 -*-

import os
import sys
import fire
import logging
import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from .data import fetch_stock_base, fetch_trade_date, fetch_history
from .process import create_frame_by_dates, create_features_frame_by_date
from .model import pred as model_pred
from .utils import check_date, TMP_PATH


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s : [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
logger.info('Buffer data storage path: {}'.format(TMP_PATH))


N = 30
TODAY = datetime.today().strftime('%Y%m%d')
DELETE_STOCK_AFTER_DATE = '20200101'

# FEATURES
ORIGIN_FEATURES = [
    # east money
    'next_day_sample_num', 'next_day_rose_probability', 'next_day_chg_avg', 
    'next_five_day_sample_num', 'next_five_day_rose_probability', 'next_five_day_chg_avg', 
    'total_score', 'capital_score', 'value_score', 'market_heat_score', 'focus_score', 
    'msg_count', 'stock_focus', 'stock_focus_chg', 'people_num_chg', 'avg_pos_rate',
    # xieqiu
    'xq_followers'
]
PATTERN_FEATURES = ['ma20-5', 'ma60-20', 'ma120-60', 'up_line_rate', 'bottem_line_rate', 'line_rate', 'change']
FEATURES = ORIGIN_FEATURES + PATTERN_FEATURES


def _get_symbols(delete_stock_after_date, date):
    base_frame = fetch_stock_base()[['symbol', 'list_date']]
    base_frame = base_frame[base_frame['list_date'] < delete_stock_after_date].drop(['list_date'], axis=1)
    history_frame = fetch_history(date)
    history_frame = history_frame[history_frame['close'] > 10]
    base_frame = pd.merge(base_frame, history_frame, how='left', on='symbol').dropna()
    return base_frame['symbol']


def _get_trade_dates(n, end):
    end = check_date(end)
    if not isinstance(n, int): raise TypeError('n must be intenger: {}'.format(n))
    trade_dates = fetch_trade_date(n=n, end=end)
    logger.info('trade date: {}, len: {}'.format(trade_dates, len(trade_dates)))
    return trade_dates


def draw_scatter(x, y):
    plt.scatter(x, y, alpha=0.7)
    plt.show()


def test(n=N, end=TODAY, delete_stock_after_date=DELETE_STOCK_AFTER_DATE, features=FEATURES):
    '''
        n: train days
        end: labels' end date
    '''
    trade_dates = _get_trade_dates(n + 4, end)
    symbols = _get_symbols(delete_stock_after_date, trade_dates[-1])
    data = create_frame_by_dates(trade_dates[:-2], symbols=symbols, update=False)
    logger.info('train len: {}, test len: {}'.format(len(data[:-2]), len(data[-1:])))

    train = pd.concat(data[:-2], sort=False)
    test  = pd.concat(data[-1:], sort=False)
    logger.info('train shape: {}, test shape: {}'.format(train.shape, test.shape))
    test['pred'] = model_pred(train, test, label_name='label', features=features)

    base_frame = fetch_stock_base()[['symbol', 'name', 'industry']]
    test = test.merge(base_frame, how='left', on='symbol')
    test = test.sort_values(by=['pred'], ascending=False).reset_index(drop=True)
    show_list = ['symbol', 'current', 'total_score', 'xq_followers', 'avg_pos_rate', 'label_change', 'pred', 'name', 'industry']
    print(test.loc[:40, show_list])
    for i in [10, 20, 30, 40, 50]:
        print(i, test.loc[(i-10):i, 'label_change'].mean())
    draw_scatter(test['pred'], test['label_change'])


def pred(n=N, end=TODAY, delete_stock_after_date=DELETE_STOCK_AFTER_DATE, features=FEATURES):
    '''
        n: train days
        end: features' end date
    '''
    trade_dates = _get_trade_dates(n + 2, end)
    symbols = _get_symbols(delete_stock_after_date, trade_dates[-1])
    data = create_frame_by_dates(trade_dates[:-2], symbols=symbols, update=False)
    logger.info('train len: {}, test date: {}'.format(len(data), trade_dates[-1]))

    train = pd.concat(data, sort=False)
    test  = create_features_frame_by_date(trade_dates[-1], symbols=symbols)
    logger.info('train shape: {}, test shape: {}'.format(train.shape, test.shape))
    test['pred'] = model_pred(train, test, label_name='label', features=features)

    base_frame = fetch_stock_base()[['symbol', 'name', 'industry']]
    test = test.merge(base_frame, how='left', on='symbol')
    test = test.sort_values(by=['pred'], ascending=False).reset_index(drop=True)
    show_list = ['symbol', 'current', 'total_score', 'next_day_rose_probability',
        'next_five_day_rose_probability', 'pred', 'name', 'industry']
    print(test.loc[:40, show_list])


def delete_cache():
    import shutil
    shutil.rmtree(TMP_PATH)


if __name__ == '__main__':
    fire.Fire({
        'test': test,
        'pred': pred,
        'delete': delete_cache,
    })
