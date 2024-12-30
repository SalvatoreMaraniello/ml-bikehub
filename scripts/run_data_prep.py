"""Run data preparation. See `src/lib_data_prep` for more info.

Data Preparation.
Trip data.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import warnings


# Local
PATH_TO_SRC = Path('../src').resolve()
sys.path.append(PATH_TO_SRC.as_posix())
import lib_data_prep  # nopep8

# settings
PATH_TO_DATA = Path('../data').resolve()

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


# Preprocess dataset
dw = lib_data_prep.prepare_weather_data(pd.read_csv(PATH_TO_DATA / 'weather.csv'))
dw.head()

dt = lib_data_prep.prepare_trip_data(pd.read_csv(PATH_TO_DATA / 'hubway_trips.csv'))
print(dt.head())

# dh = pd.read_csv(PATH_TO_DATA / 'hubway_stations.csv')


# Merge data
# Precipitation data are an aggregate of previous hour. Therefore, for best accuracy,
# we may want to ceil the trip start date to the next hour and join on the weather data
# to retrieve the previpitation during the trip.
# E.g. if trip starts at 14:23, we want to use weather precipitation data at 15:00,
# which are the cumulative sum of precipitation from 14 to 15.
# The issue with this approach is that:
# 1. At prediction time, we would need to feed a deployed model with real time
# precipitation data.
# 2. At train time, we are feeding to the model info from the future. E.g., if
# the trip started at 12:12 but rain only started minutes after, during training
# we would be feeding this info to the model. As this would not be possible when
# the model is deployed, the deployed model would perform worse than expected.
# dt['date_to_merge_weather_data'] = dt['start_date'].dt.ceil('H')
# (dt['date_to_merge_weather_data'] - dt['start_date']).min()


# We will use as feature the HPCP during the previous hour. This will always be available.
# We assume a 1 minute latency for the precipitation data to update.
# Namely, if a trip started at 12:00, we will assume that the data for the hour ending at 12:00 are
# not yet available, and use data for the period 10:00 to 11:00.
dt['date_to_merge_weather_data'] = (dt['start_date'] - timedelta(minutes=1)).dt.floor('h')
#  dt[['date_to_merge_weather_data', 'start_date']]

df = dt.merge(dw, left_on='date_to_merge_weather_data', right_on='DATE', how='left')

# Add indicator of missingness for period when no weather data were collected.
mask_no_weather_data = (
    (df['date_to_merge_weather_data'] < dw['DATE'].min()) |
    (df['date_to_merge_weather_data'] > dw['DATE'].max())
)
df['has_precip_data'] = 1
df.loc[mask_no_weather_data, 'has_precip_data'] = 0


# Set HPCP and trace to zero when weather data not available
# Note: this can be anything. In theory, a ML model should learn to discard wehether info when the
# 'has_precip_data' is zero.
df.loc[mask_no_weather_data, ['HPCP', 'has_trace']] = 0


# Fill NA
# In the period when whether data were collected, we assume no precipitation whenever HPCP data are
# missing.
df.loc[~mask_no_weather_data, 'HPCP'] = df.loc[~mask_no_weather_data, 'HPCP'].fillna(0.0)
df.loc[~mask_no_weather_data, 'has_trace'] = df.loc[~mask_no_weather_data, 'has_trace'].fillna(0)

# Override types and drop unwanted columns
df['has_trace'] = df['has_trace'].astype(int)
df = df.drop(columns=['date_to_merge_weather_data', 'DATE',])
