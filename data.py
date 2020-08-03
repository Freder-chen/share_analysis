# -*- coding: utf-8 -*-
"""
fetch data
Created on 2020/06/22
@author: Freder Chen
@contact: freder-chen@qq.com
"""

import os
import time
import json
import logging
import pandas as pd
from datetime import datetime
from .utils import fetch
from .utils import check_date
from .utils import HOST, TMP_PATH


logger = logging.getLogger(__name__)

_SCRET_DATE = '20190101' # begin date of fetch function


_STOCK_BASE = '{host}/api/tushareprobase'
_TRADE_DATE = '{host}/api/workday?start={start}&end={end}'
_HISTORY    = '{host}/api/history?symbol={symbol}&start={start}&end={end}'
_EM_FEATURE = '{host}/api/eastmoney?start={start}&end={end}'
_XQ_FEATURE = '{host}/api/xueqiu?start={start}&end={end}'


def fetch_stock_base(host=HOST, tmp=TMP_PATH):
    url = _STOCK_BASE.format(host=host)
    filename = os.path.join(tmp, 'stock_base.csv') if tmp else None
    return _fetch_data_frame(url, filename=filename).astype({'list_date': 'str'})


def fetch_history(date, host=HOST, tmp=TMP_PATH):
    url = _HISTORY.format(host=host, symbol='', start=date, end=date)
    filename = os.path.join(tmp, 'history', '{}.csv'.format(date)) if tmp else None
    return _fetch_data_frame(url, filename=filename)


def fetch_daily(symbol, host=HOST, tmp=TMP_PATH):
    """
    Describe:
        request symbol: need
                start: '20190101'
    """
    url = _HISTORY.format(host=host, symbol=symbol, start=_SCRET_DATE, end='')
    filename = os.path.join(tmp, 'daily', '{}.csv'.format(symbol)) if tmp else None
    update = False
    if os.path.exists(filename) and time.strftime('%Y%m%d', \
            time.gmtime(os.path.getmtime(filename))) != datetime.today().strftime('%Y%m%d'):
        update = True
    return _fetch_data_frame(url, filename=filename, update=update)


def fetch_em_feature(date, host=HOST, tmp=TMP_PATH):
    url = _EM_FEATURE.format(host=host, start=date, end=date)
    filename = os.path.join(tmp, 'em_feature', '{}.csv'.format(date)) if tmp else None
    return _fetch_data_frame(url, filename=filename)


def fetch_xq_feature(date, host=HOST, tmp=TMP_PATH):
    url = _XQ_FEATURE.format(host=host, start=date, end=date)
    filename = os.path.join(tmp, 'xq_feature', '{}.csv'.format(date)) if tmp else None
    return _fetch_data_frame(url, filename=filename)


def fetch_trade_date(start=None, end=None, n=None, rs=_SCRET_DATE, re=datetime.today().strftime('%Y%m%d'), host=HOST, tmp=TMP_PATH):
    '''
    Describe:
        request to update once a day.
        request start: '20190101'
                end  : today
    '''
    def _ftd(dates, start, end, n):
        dates = [check_date(d) for d in dates]; dates.sort()
        if start is not None and end is not None:
            start, end = check_date(start), check_date(end)
            dates = [(d) for d in dates if d >= start and d <= end]
        elif start is not None and n is not None:
            start = check_date(start)
            if not isinstance(n, int):
                raise TypeError('n must be intenger: {}'.format(n))
            dates = [(d) for d in dates if d >= start][:n]
        elif end is not None and n is not None:
            end = check_date(end)
            if not isinstance(n, int):
                raise TypeError('n must be intenger: {}'.format(n))
            dates = [(d) for d in dates if d <= end][-n:]
        else:
            raise ValueError('start, end and n must be filled with two.')
        return dates

    fetch_url = _TRADE_DATE.format(host=host, start=rs, end=re)
    filename = os.path.join(tmp, 'trade_date.csv') if tmp else None
    try:
        with open(filename) as f:
            data = json.load(f)
        if data['url'] != fetch_url:
            raise ValueError('Data need to update.')
    except (ValueError, FileNotFoundError):
        data  = {'url': fetch_url, 'data': _fetch_json(fetch_url)}
        if filename:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(filename, 'w') as f:
                json.dump(data, f)
    finally:
        dates = _ftd(data['data'], start, end, n)
    return dates


def _fetch_data_frame(url, filename=None, update=False):
    if (not update) and filename and os.path.exists(filename):
        df = pd.read_csv(filename, dtype={'symbol': 'str'})
    else:
        df = pd.DataFrame(json.loads(fetch(url).content.decode('utf-8')))
        if (not df.empty) and filename:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            df.to_csv(filename, index=False)
    return df


def _fetch_json(url, filename=None, update=False):
    if (not update) and filename and os.path.exists(filename):
        with open(filename) as f:
            ret = json.load(f)
    else:
        # fetch(url).json()
        ret = json.loads(fetch(url).content.decode('utf-8'))
        if ret != {} and filename:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(filename, 'w') as f:
                json.dump(ret, f)
    return ret
