import pandas as pd
import numpy as np
from os.path import abspath, dirname
from core.weather_extraction import (
    create_region_filter, extract_regional_weather)
from core.extract_all_weather import extract_rainfall, generate_hdf_day_keys
from collections import defaultdict

import pytest

PARENT_DIR = dirname(dirname(abspath(__file__)))
CUR_DIR = dirname(abspath(__file__))


@pytest.fixture
def lat_grid():
    return np.array([
        [50.0, 50.15, 50.3],
        [51.0, 51.15, 51.3],
        [52.0, 52.15, 52.3]
    ])


@pytest.fixture
def lng_grid():
    return np.array([
        [1.0, 1.15, 1.3],
        [2.0, 2.15, 2.3],
        [3.0, 3.15, 3.3]
    ])


def test_create_region_filter(lat_grid, lng_grid):
    region_filter = create_region_filter(
        lat_grid, lng_grid, 50.0, 1.0, 0.25)
    expected = np.array([
        [True, True, False],
        [False, False, False],
        [False, False, False]
    ])
    assert np.array_equal(region_filter, expected)


@pytest.fixture
def weather_grid():
    return np.array([
        [4.0, 6.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])


def test_extract_regional_weather(weather_grid):
    region_filter = np.array([
        [True, True, False],
        [False, False, False],
        [False, False, False]
    ])
    weather = extract_regional_weather(weather_grid, region_filter)
    assert weather == 5.0


@pytest.fixture
def hdf_rainfall(lat_grid, lng_grid):
    return {
        'Latitude_grid': lat_grid,
        'Longitude_grid': lng_grid,
        '2005_010_0001': {
            'daily_grid': np.array([
                [21.0, 14.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
        },
        '2005_010_0002': {
            'daily_grid': np.array([
                [18.0, 11.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
        }
    }


@pytest.fixture
def all_cultivars_df():
    return pd.DataFrame.from_records([
        {
            'Cultivar': 'Alchemy',
            'Lat': 50.0,
            'Long': 1.0,
            'Year': 2006,
            'Ripe Time': 1,
            'Yield': 9.75,
            'Sow Month': '10',
            'Sow Day': 1,
            'Sow Year': 2005,
        },
        {
            'Cultivar': 'Ambrosia',
            'Lat': 50.0,
            'Long': 1.0,
            'Year': 2006,
            'Ripe Time': 1,
            'Yield': 11.21,
            'Sow Month': '10',
            'Sow Day': 1,
            'Sow Year': 2005,
        }
    ])


def test_extract_rainfall(hdf_rainfall, all_cultivars_df):
    rainfall = extract_rainfall(all_cultivars_df, hdf_rainfall, 0.25)
    # get an array of mean weather for each region for each grow
    # day (17.5 average for grow day 1, 14.5 grow day 2, only for one location)
    e_data = defaultdict(
        list,
        {'Alchemy': [[17.5, 14.5]], 'Ambrosia': [[17.5, 14.5]]})
    assert rainfall == e_data


def test_generate_hdf_day_keys(all_cultivars_df):
    row_0 = all_cultivars_df.iloc[0]
    hdf_day_keys = list(generate_hdf_day_keys(row_0))
    e_day_keys = ['2005_010_0001', '2005_010_0002']
    assert e_day_keys == hdf_day_keys
