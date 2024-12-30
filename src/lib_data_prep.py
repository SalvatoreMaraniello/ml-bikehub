import pandas as pd
import typing
import pathlib
from datetime import datetime
import numpy as np


def prepare_weather_data(
        data: pd.DataFrame,
        drop_unnecessary_columns: bool = True) -> pd.DataFrame:
    """Prepare weather data. The following steps are executed:
    - Add `has_trace` column
    - Convert dates to datetime.

    Args:
        data (pd.DataFrame): data from `weather.csv`.
        drop_unnecessary_columns (bool, optional): If true, drops columns that can not be used
        for training/merging to other dataset. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe with prepared data.
    """

    # Feature engineering
    data['has_trace'] = (data['Measurement_Flag'] == 'T').astype(int)

    data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m%d %H:%M')

    cols_drop = [
        'STATION', 'STATION_NAME', 'ELEVATION', 'LATITUDE', 'LONGITUDE', 'Quality_Flag',
        'Measurement_Flag',
    ]

    if drop_unnecessary_columns:
        data = data.drop(columns=cols_drop)

    return data.sort_values('DATE')


def prepare_trip_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare trip data. The following steps are executed:
    - Convert trip duration to minutes.
    - Remove trips without start station id.
    - Build 'is_registered' feature.
    - Added average trip time from start station over the 7 days ahead of the trip.
    - Add cyclic variables for day of the week (`dow_sin` and `dow_cos`).
    - Drop unnecessary/unusable columns.

    Args:
        data (pd.DataFrame): Data from `hubway_trips.csv`.

    Returns:
        pd.DataFrame: A dataframe with extra features/preprocessed data. Imputing and encoding
            will still be required (but these should be taken care of in a preprocessing pipeline
            before feeding the data to any ML model).
    """

    # convert duration in minutes (for easier interpretation)
    data['duration_min'] = data['duration'] / 60.0

    # Remove trips without start station id.
    # - Trips without end station id should be removed for training (but can be included for
    # model evaluation and at prediction time - the end station id is unknown and not used
    # as a feature).
    data = data[~data[['strt_statn']].isnull().any(axis=1)]
    data.loc[:, 'strt_statn'] = data['strt_statn'].astype(int)

    # convert and truncate the trip start_date
    def _to_datetime(s: str):
        start_date, time = s.split(' ')
        m, d, y = start_date.split('/')
        H, M, _ = time.split(':')
        return datetime(year=int(y), month=int(m), day=int(d), hour=int(H), minute=int(M))
    data['start_date'] = data['start_date'].apply(_to_datetime)
    data.loc[:, 'start_date_trunc'] = data.start_date.dt.floor('D')

    # TODO: use for quality check
    data['end_date'] = data['end_date'].apply(_to_datetime)

    data['is_registered'] = (data['subsc_type'] == 'Registered').astype(int)

    # Add average trip duration (per station, per date)
    # Date dependency is based on average over previous 7 days.
    avg_duration_prev_7days_from_statn = compute_average_trip_time_from_station(data, window_days=7)
    data = data.merge(
        avg_duration_prev_7days_from_statn,
        on=['strt_statn', 'start_date_trunc'],
        how='left'
    )

    # Add cyclic variable for day of the week
    data['dow'] = data['start_date'].dt.day_of_week
    data['dow_sin'] = np.sin(2 * np.pi * data['dow'] / 7.0)
    data['dow_cos'] = np.cos(2 * np.pi * data['dow'] / 7.0)

    # Drop unnecessary KPI / including those replaced by new features
    data = data.drop(columns=[
        'duration', 'zip_code', 'bike_nr', 'birth_date', 'gender', 'status',
        'subsc_type', 'hubway_id', 'end_date', 'start_date_trunc',
        'dow',
    ])

    return data


def compute_average_trip_time_from_station(
    trip_data: pd.DataFrame,
    window_days: int = 7
) -> pd.DataFrame:
    """Compute the average trip time per starting hub station, per date. The dependence on the date
    allows accounting for changes in average trip time due to changes in users behavior and typical
    routes (e.g. new routes becoming popular as new stations are added to the network).

    Args:
        trip_data (pd.DataFrame): data from trips dataset.
        window_days (int, optional): Window over which the average trip time is computed. Defaults 
        to 7.

    Returns:
        pd.DataFrame: A dataframe with columns 'strt_statn', 'start_date_trunc', 'avg_duration_prev_7days'.
    """

    trip_data = trip_data.sort_values('start_date')

    # get total trips duration (and count) for each station for each date
    avg_duration_prev_7days_from_statn = trip_data.groupby(
        ['strt_statn', 'start_date_trunc'])['duration_min'].agg(
        ['sum', 'count']).rename(
        columns={'sum': 'daily_trips_total_duration', 'count': 'daily_trips_count'}).reset_index()

    # ... next, for each station, compute average over previous `window_avg` days
    # (note: shift required to exclude current date from average)
    window_avg = 7
    total_in_window = avg_duration_prev_7days_from_statn\
        .groupby('strt_statn')\
        .shift()\
        .rolling(window=window_avg, on='start_date_trunc')\
        .agg({'daily_trips_total_duration': 'sum', 'daily_trips_count': 'sum'})

    total_in_window['avg_duration_prev_7days'] = total_in_window[
        'daily_trips_total_duration'] / total_in_window['daily_trips_count']

    # join back on index (note: calc verified)
    avg_duration_prev_7days_from_statn = avg_duration_prev_7days_from_statn[[
        'strt_statn', 'start_date_trunc']].join(total_in_window[['avg_duration_prev_7days']])

    return avg_duration_prev_7days_from_statn
