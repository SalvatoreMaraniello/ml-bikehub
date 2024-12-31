'''
This library contains a few utils for ML training
'''

import copy
import pandas as pd
import numpy as np

# from sklearn.svm import LinearSVC
from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error,
                             r2_score, median_absolute_error)

from sklearn.model_selection import GridSearchCV


def get_scores(y_true, y_pred, model_id: str = None) -> pd.Series:
    """Produce a set of scores suitable for regression models."""
    return pd.DataFrame([dict(
        id=model_id,
        # same unit as target variable, plus less sensitive to outliers than MSE.
        mae=mean_absolute_error(y_true, y_pred),
        # same unit as target variable, plus robust to outliers
        medae=median_absolute_error(y_true, y_pred),
        mse=mean_squared_error(y_true, y_pred),
        rmse=root_mean_squared_error(y_true, y_pred),
        r2=r2_score(y_true, y_pred)
    )]).set_index('id')


def get_Xy(dt, cols_features, col_out):
    '''
    Utility function to convert a pandas DataFrame to numpy arrays.
    '''
    # return dt[cols_features].values, dt[col_out].values
    return dt[cols_features], dt[col_out]


# def make_learning_curves(
#     train_sizes_norm, estimator, X, y, cv, random_state,
#     scoring=make_scorer(log_loss, needs_proba=True, needs_threshold=False)
# ):
#     '''
#     Wrapper for `sklearn.model_selection.learning_curve` with default log-loss scorer.

#     Params
#     ------
#     Refer to `sklearn.model_selection.learning_curve`.

#     Returns
#     -------
#     {train,test}_scores_avg: array-like
#         Average of training and test score (in this order) per each size in train_sizes_norm.
#     {train,test}_scores_std: array-like
#         Standard deviation of training and test score (in this order) per each size in train_sizes_norm.
#     '''

#     train_sizes_abs, train_scores, test_scores = learning_curve(
#         estimator, X=X, y=y, train_sizes=train_sizes_norm, cv=cv, scoring=scoring,
#         n_jobs=-1,
#         verbose=0,
#         shuffle=True,
#         random_state=random_state,
#         error_score='raise',
#     )

#     # get avg/std
#     train_scores_avg, test_scores_avg = train_scores.mean(axis=1), test_scores.mean(axis=1)
#     train_scores_std, test_scores_std = train_scores.std(axis=1), test_scores.std(axis=1)

#     return train_scores_avg, test_scores_avg, train_scores_std, test_scores_std


def from_gridsearch_to_df(g_search: GridSearchCV, scoring_dict: dict):
    '''
    Given an instance of GridSearchCV, `g_search`, This function extracts the score for each set
    of parameters run, and dumps them into a `pandas.DataFrame`.
    '''

    N = len(g_search.cv_results_['params'])
    outs = []

    for ii in range(N):
        # get model params
        params = copy.deepcopy(g_search.cv_results_['params'][ii])
        # extract scores...
        for score_lab in scoring_dict.keys():
            params['%s_avg' % score_lab] = g_search.cv_results_['mean_test_%s' % score_lab][ii]
            params['%s_std' % score_lab] = g_search.cv_results_['std_test_%s' % score_lab][ii]
        outs.append(params)
    return pd.DataFrame(outs)
