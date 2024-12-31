from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

import xgboost as xg

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error,
                             r2_score, median_absolute_error, make_scorer)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer


# Local
PATH_TO_SRC = Path('../src').resolve()
sys.path.append(PATH_TO_SRC.as_posix())
import lib_model  # nopep8

PATH_TO_DATA = Path('../data').resolve()
PATH_TO_RES = Path('./results').resolve()
PATH_TO_FIGS = Path('./figures').resolve()

os.makedirs(PATH_TO_RES, exist_ok=True)
os.makedirs(PATH_TO_FIGS, exist_ok=True)

RANDOM_STATE = 78239


# --------------------------------------------------------------------------------------- Load data
# Preprocess dataset
df = pd.read_csv(PATH_TO_DATA / 'training-data.csv')

# categorise kpis
cols_features = [
    'avg_duration_prev_7days', 'HPCP',
    'is_registered', 'dow_sin',
    'dow_cos', 'has_trace',
]
col_target = 'duration_min'


# ---------------------------------------------------------------------------------------- Clean Up

# remove data points collected when weather data were not available
df = df[df['has_precip_data'] == 1]

# Some trips have unrealisticly long durations (either negative, or too large). These records are
# removed
duration_min_up_bound = df['duration_min'].quantile(.95)
mask_duration = (df['duration_min'] > 0) & (df['duration_min'] <= duration_min_up_bound)
print(f'Removing {sum(~mask_duration)}/{len(df)} records due to abnormal trip duration time above '
      f'{duration_min_up_bound: .2f}.')
df = df[mask_duration].reset_index(drop=True)


# Overview for notebook.
df[cols_features].describe()


# # - ~10% of records do not have a `avg_duration_prev_7days`. These is for trips that took place within 7 days from opening a new hub. These records can be removed because as:
# #     - There is plenty of data
# #     - trips happening just after a hub was opened (or moved) may not be longer (as users need to learn new routes)
# # -
mask_na = df['avg_duration_prev_7days'].isna()
df = df[~mask_na].reset_index()


# -------------------------------------------------------------------------------- Scaling Pipeline

# add scaler/column transformer
scaler = ColumnTransformer([
    ('standard_sc', StandardScaler(), ['avg_duration_prev_7days',]),
    ('log_sc', FunctionTransformer(np.log1p, validate=True), ['HPCP',]),
    ('no_scaling', 'passthrough', ['is_registered', 'dow_sin', 'dow_cos', 'has_trace',])
], remainder='passthrough')

scaler.fit(df[cols_features])

df_transf = pd.DataFrame(
    scaler.transform(df[cols_features]),
    columns=cols_features
)

cols_plot = cols_features
fig = plt.figure('Impact of Transformations', (4*len(cols_plot), 6))
Ax = fig.subplots(2, len(cols_plot), squeeze=False)
for ii, col in enumerate(cols_plot):
    # before
    Ax[0, ii].set_title(col + ' (before)')
    df[col].hist(ax=Ax[0, ii], alpha=.4)
    # after
    Ax[1, ii].set_title(col + ' (after)')
    df_transf[col].hist(ax=Ax[1, ii], alpha=.4)


# ---------------------------------------------------------------------------------- Baseline Model
# This model does not require training, as it simply uses the average time. We only need evaluating it.
# The model has a marginally positive R2 (i.e., it performs just marginally better than a a model that
# always predicts the mean trip duration time). This makes sense, as in this model the average is
# computed for each station.

print(f'''Median duration: {df['duration_min'].median():.2f}''')
print(f'''Mean duration: {df['duration_min'].mean():.2f} (std: {df['duration_min'].std():.2f})''')

scores = lib_model.get_scores(df[col_target], df['avg_duration_prev_7days'], 'baseline')
scores


# ---------------------------------------------------------------------------- Prepare for training
# Split data into training and testing sets
df_train, df_test, y_train, y_test = train_test_split(
    df[cols_features], df[col_target], test_size=0.2, random_state=RANDOM_STATE)

# For testing, we also prepare a scaled version of the target variable
scaler_target = StandardScaler()
scaler_target.fit(df[col_target].values.reshape(-1, 1))
y_train_scaled = scaler_target.transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_target.transform(y_test.values.reshape(-1, 1))


# -------------------------------------------------------------------------- Linear regressor model

regressor_lin = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE)
model_lin = Pipeline([('scaler', scaler), ('regressor', regressor_lin)])

# train/evaluate
model_lin.fit(df_train, y_train)
y_pred = model_lin.predict(df_test)
scores_lin = lib_model.get_scores(y_test, y_pred)

# # train/evaluate (using scaled target variable)
# model_lin.fit(df_train, y_train_scaled)
# y_pred = scaler_target.inverse_transform(model_lin.predict(df_test).reshape(-1,1))
# scores_lin = lib_model.get_scores(y_test, y_pred)

scores = pd.concat([scores, scores_lin]).drop_duplicates()
scores


# ----------------------------------------------------------------------------------------- XGBoost
# XGBoost should capture nonlinearities, and it may be less sensitive to scaling of the HPCP variable.

regressor_xgb = xg.XGBRegressor(
    objective='reg:squarederror',
    # objective = 'reg:absoluteerror'
    n_estimators=50,
    max_depth=5,
    learning_rate=.5,         # Values in range [0.1,1] produced best results here.
    subsample=0.75,           # Data subsampling.
    gamma=1,                  # Min. split loss (fight overfitting).
    min_child_weight=0,       # Min. sum of weights required in child node.
    # colsample_bytree=0.8,   # Feature subsampling
    # reg_alpha=0.1,          # L1 regularization
    # reg_lambda=1.0,         # L2 regularization
    seed=RANDOM_STATE,
)

model_xgb = Pipeline([('scaler', scaler), ('regressor', regressor_xgb)])
model_xgb.fit(df_train, y_train)
y_pred = model_xgb.predict(df_test[cols_features])
scores_xgb = lib_model.get_scores(y_test, y_pred, 'xgb')

scores = pd.concat([scores, scores_xgb]).drop_duplicates()
scores

print(scores_xgb)

# ------------------------------------------------------------------------------------- Grid Search

param_grid = dict(
    regressor__n_estimators=[10, 20, 50, 100],
    regressor__max_depth=[5, 10, 20],
    regressor__learning_rate=[0.01, 0.1, 0.3, 0.5],
    regressor__subsample=[0.75, 1.0],
    regressor__gamma=[0, 1, 5],
    regressor__min_child_weight=[1, 3, 5],
)
scoring_dict = dict(
    mae=make_scorer(mean_absolute_error),
    medae=make_scorer(median_absolute_error),
    mse=make_scorer(mean_squared_error),
    rmse=make_scorer(root_mean_squared_error),
    r2=make_scorer(r2_score),
)
gs = RandomizedSearchCV(
    estimator=model_xgb,
    param_distributions=param_grid,
    scoring=scoring_dict,
    refit='mae',
    n_iter=60,
    cv=4,
    n_jobs=-1,
    verbose=1,
)


RUN_GRID_SEARCH = True

if RUN_GRID_SEARCH:
    _ = gs.fit(df[cols_features], df[col_target].values)
    df_gs_res = lib_model.from_gridsearch_to_df(gs, scoring_dict)
    df_gs_res.to_csv(PATH_TO_RES/'grid-search-xgb.csv', index=False)
else:
    df_gs_res = pd.read_csv(PATH_TO_RES/'grid-search-xgb.csv')

df_gs_res.sort_values('mae_avg', ascending=False)


1/0
# ---------------------------------------------------------------------------------- Neural network

regressor_nnet = MLPRegressor(
    hidden_layer_sizes=(32, 16),  # Two hidden layers with 64 and 32 neurons
    activation='relu',  # Activation function (e.g., 'relu', 'tanh', 'logistic')
    solver='adam',      # Optimization algorithm ('adam', 'sgd', etc.)
    max_iter=1000,       # Maximum number of iterations
)

model_nnet = Pipeline([('scaler', scaler), ('regressor', regressor_nnet)])

# # train/evaluate
# model_nnet.fit(df_train, y_train)
# y_pred = model_nnet.predict(df_test)
# scores_nnet = lib_model.get_scores(y_test, y_pred)

# {'mae': 6.717339454815039,
#  'medae': np.float64(5.032088090813949),
#  'mse': 94.47408823339603,
#  'rmse': 9.719778198775733,
#  'r2': 0.21761861167337815}


# ----------------------------------------------------------------------- Gradient Boost Tree Model
# A boosting tree model should capture nonlinearities, while being less sensitive to scaling of the
# HPCP variable.

regressor_btree = GradientBoostingRegressor(
    n_estimators=10,         # Number of boosting stages
    max_depth=10,            # max. depth of trees
    learning_rate=0.01,      # learning rate step size
    subsample=0.75,           # Data subsampling
    random_state=RANDOM_STATE,
)

model_btree = Pipeline([('scaler', scaler), ('regressor', regressor_btree)])
model_btree.fit(df_train, y_train)
y_pred = model_btree.predict(df_test[cols_features])
scores_btree = lib_model.get_scores(y_test, y_pred)
print(scores_btree)
