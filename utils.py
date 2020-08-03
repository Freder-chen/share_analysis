# -*- coding: utf-8 -*-

import os
import logging
import requests
import datetime
import tempfile

logger = logging.getLogger(__name__)


# HOST = 'http://123.57.60.215:8000'
HOST = 'http://134.175.216.8'
TMP_PATH = os.path.join(tempfile.gettempdir(), 'share_analysis')


def fetch(url):
    logger.info('fetch data - \'{}\''.format(url))
    return requests.get(url=url)


def _is_date(str_):
    try:
        datetime.datetime.strptime(str_, '%Y%m%d')
    except ValueError:
        return False
    return True


def check_date(date):
    def _check_str(date):
        if len(date) != 8 or (not _is_date(date)):
            raise ValueError('Illegal date: {}, useful format: \'%Y%m%d\''.format(date))
        return date

    if isinstance(date, int):
        return _check_str(str(date))
    elif isinstance(date, str):
        return _check_str(date)
    elif isinstance(date, datetime.date):
        return date.strftime('%Y%m%d')
    else:
        raise TypeError('Date is not datetime.date or a valid string: {}'.format(date))


# def sub_str_date(str_date, n_days):
#     if not isinstance(str_date, str) or len(str_date) != 8 or not _is_date(str_date):
#         raise ValueError('Illegal date: {}, useful format: \'%Y%m%d\''.format(str_date))
#     date = datetime.datetime.strptime(str_date, '%Y%m%d')
#     return date - datetime.timedelta(days=n_days)
