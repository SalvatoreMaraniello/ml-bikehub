import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Local
PATH_TO_SRC = Path('../src').resolve()
sys.path.append(PATH_TO_SRC.as_posix())
import lib_data_prep  # nopep8


PATH_TO_DATA = Path('../data').resolve()


dt = lib_data_prep.prepare_trip_data(pd.read_csv(PATH_TO_DATA / 'hubway_trips.csv'))
dt['start_date_trunc'] = dt['start_date'].dt.floor('D')

# Compute avg trip time from station
# - depends on regular users/routes/active stations nearby.
# dt = dt[dt['start_date_trunc'] >= datetime(2012, 7, 1)]
# dt = dt[dt['start_date_trunc'] <= datetime(2012, 7, 6)]
# dt = dt[dt['strt_statn'].isin([44, 46, 68])]

dt = dt.sort_values('start_date')
dt['strt_statn'] = dt['strt_statn'].astype(int)

# get total trips duration (and count) for each station for each date
avg_duration_prev_7days_from_statn = dt.groupby(['strt_statn', 'start_date_trunc'])['duration_min']\
    .agg(['sum', 'count'])\
    .rename(columns={'sum': 'daily_trips_total_duration', 'count': 'daily_trips_count'})\
    .reset_index()

# ... next, for each station, compute average over previous `window_avg` days
# (note: shift required to exclude current date from average)
window_avg = 4
total_in_window = avg_duration_prev_7days_from_statn\
    .groupby('strt_statn')\
    .shift()\
    .rolling(window=window_avg, on='start_date_trunc')\
    .agg({'daily_trips_total_duration': 'sum', 'daily_trips_count': 'sum'})

total_in_window['avg_duration_prev_7days_prev_days'] = total_in_window[
    'daily_trips_total_duration'] / total_in_window['daily_trips_count']


# join back on index (note: calc verified)
avg_duration_prev_7days_from_statn = avg_duration_prev_7days_from_statn[[
    'strt_statn', 'start_date_trunc']].join(total_in_window[['avg_duration_prev_7days_prev_days']])


# # for each station, we only have `window_avg` nulls.
# dd = avg_duration_prev_7days_from_statn
# mask_null = dd['avg_duration_prev_7days_prev_days'].isna()
# dd = dd[mask_null]
# dd.groupby('strt_statn').count().max()

dt = dt.merge(
    avg_duration_prev_7days_from_statn,
    on=['strt_statn', 'start_date_trunc'],
    how='left'
)
