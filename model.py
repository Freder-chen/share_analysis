# -*- coding: utf-8 -*-
"""
modeling
Created on 2020/06/22
@author: Freder Chen
@contact: freder-chen@qq.com
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from .data import fetch_stock_base, fetch_history, fetch_em_feature, fetch_xq_feature


logger = logging.getLogger(__name__)


_params = {
    'boosting_type': 'gbdt',
    'num_leaves': 32,
    'reg_alpha': 1,
    'reg_lambda': 10,
    'max_depth': -1,
    'n_estimators': 3000,
    'objective': 'binary',
    'max_bin': 155,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'subsample_freq': 1,
    'learning_rate': 0.04,
    'random_state': 2020,
    'n_jobs': -1    
}


def pred(train_frame, pred_frame, label_name='label', features=None, k=2, params=_params):
    ''' pred
        train_frame: pd.DataFrame
        pred_frame: pd.DataFrame
        label_name: str
        features: [str, str, ...]
        k: int
        params: dict(lgb_params)

        return: pd.Series
    '''
    pred_frame['pred'] = 0
    train_x = train_frame[features].values if features else train_frame.values
    train_y = train_frame[label_name].values
    pred_x = pred_frame[features].values if features else pred_frame.values
    for lgb_model in _train(train_x, train_y, k=k, params=_params):
        pred_frame['pred'] += lgb_model.predict_proba(pred_x, num_iteration=lgb_model.best_iteration_)[:, 1]
    pred_frame['pred'] /= k
    return pred_frame['pred']


def _train(x, y, k=2, params=_params):
    ''' _train
        >>> _train(x, y)
        [lgb_model, lgb_model, ...]
    '''
    lgb_model = lgb.LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=k, random_state=2020, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(x, y)):
        lgb_model.fit(x[train_index, :], y[train_index],
            eval_set=[(x[train_index, :], y[train_index]), (x[test_index, :], y[test_index])],
            early_stopping_rounds=900, verbose=False)
        logger.info('model {} best binary_logloss: {}'.format(index, lgb_model.best_score_['valid_1']['binary_logloss']))
        yield lgb_model
