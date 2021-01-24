from core.utilities import extract_data
from core.weather_extraction import read_or_create

import numpy as np
from os.path import abspath, dirname, join
import os

import pytest


PARENT_DIR = dirname(dirname(abspath(__file__)))


# Extracted from the eample_data/Test_Data.csv file ------------
expected_lats = np.array([
    52.0834, 52.0834, 52.0834, 52.5181, 52.5181, 52.5181, 57.466,
    57.466, 56.1843, 51.8599, 51.8533, 51.8533, 54.0264, 54.0264,
    53.2523, 53.2523, 53.2523, 53.2523, 52.3342, 52.3342, 52.3342,
    53.429, 55.6729, 55.6729, 55.6729, 55.6729, 53.1635, 53.1635,
    53.1635, 54.0116, 54.0116, 52.2429, 52.2429
])

expected_longs = np.array([
    -1.4545, -1.4545, -1.4545, 1.0155, 1.0155, 1.0155, -2.117,
    -2.117, -3.1244, 0.4633, 0.3865,  0.3865, -0.8172, -0.8172,
    -0.1713, -0.1713, -0.1713, -0.1713, -1.6477, -1.6477, -1.6477,
    -0.1807, -2.0135, -2.0135, -2.0135, -2.0135, -2.9972, -2.9972,
    -2.9972, -0.8236, -0.8236, 0.7105, 0.7105
])

expected_years = np.array([
    2013, 2014, 2015, 2006, 2008, 2012, 2014, 2015, 2015, 2010, 2005,
    2007, 2005, 2011, 2005, 2006, 2008, 2009, 2007, 2008, 2009, 2008,
    2005, 2006, 2007, 2008, 2012, 2014, 2015, 2008, 2010, 2006, 2008
])

expected_ripe_time = np.array([
    323, 311, 310, 318, 318, 282, 340, 362, 347, 318, 284, 318, 324,
    318, 318, 318, 318, 318, 318, 318, 305, 325, 332, 316, 324, 314,
    318, 318, 318, 313, 318, 318, 294
])

expected_yields = np.array([
    8.76, 14.8, 12.03, 6.82, 12.45, 8.85, 10.66, 9.81, 11.0,
    9.01, 11.97, 10.89, 11.15, 10.35, 8.65, 9.43, 12.03, 7.01,
    10.19,  8.55, 11.12, 11.68, 11.17, 9.75, 9.96, 8.97, 8.49,
    12.15, 11.23, 10.42, 7.11, 10.28, 8.9
])

expected_sow_day = np.array([
    10, 10, 22, 10, 10, 10, 28, 24, 27, 10, 10, 10, 10, 27, 10, 10, 21,
    21, 10, 13, 13, 10, 10, 10, 10, 10, 10, 24, 22, 10, 29, 10, 10
])

expected_sow_month = np.array([
    '09', '08', '10', '01', '01', '01', '09', '09', '09', '09', '01',
    '01', '01', '09', '01', '01', '10', '10', '01', '10', '10', '01',
    '01', '01', '01', '01', '01', '10', '10', '01', '09', '01', '01'
])
# ----------------------------------------------------------------------


def rounded_equal(array_a, array_b):
    return np.array_equal(np.round(array_a, 3), np.round(array_b, 3))


def test_extract_data():
    data = extract_data(join(PARENT_DIR, "example_data", "Test_Data.csv"))
    lats, longs, years, ripe_time, yields, sow_day, sow_month = data
    assert rounded_equal(lats, expected_lats)
    assert rounded_equal(longs, expected_longs)
    assert np.array_equal(years, expected_years)
    assert np.array_equal(ripe_time, expected_ripe_time)
    assert rounded_equal(yields, expected_yields)
    assert np.array_equal(sow_day, expected_sow_day)
    assert np.array_equal(sow_month, expected_sow_month)


@pytest.fixture()
def small_dataset():
    cult = "Test_Single"
    hdf5_filepath = join(
        PARENT_DIR, "Climate_Data", "Region_Climate_" + cult + ".hdf5")
    csv_filepath = join(
        PARENT_DIR, "example_data", cult + "_Data.csv")
    yield cult, csv_filepath
    os.remove(hdf5_filepath)


def test_create(small_dataset):
    cultivar, data_filepath = small_dataset
    cultivar_data = extract_data(data_filepath)

    # create a new regional hdf5 weather file for the cultivar
    created_weather_data = read_or_create(cultivar, cultivar_data)
    tmi, tma, pre, pre_anom, sun, sun_anom = created_weather_data

    expected_temp_min = np.array([14.55400152, 10.37660339])
    expected_temp_max = np.array([19.80653811, 16.42561807])
    expected_rainfall = np.array([1.73540998, 0.05580953])
    expected_rainfall_anom = np.array([-3.08042272, -4.63849409])
    expected_sun = 173.4835060087431
    expected_sun_anom = 47.197628129593326

    assert rounded_equal(tmi[0, 0:2], expected_temp_min)
    assert rounded_equal(tma[0, 0:2], expected_temp_max)
    assert rounded_equal(pre[0, 0:2], expected_rainfall)
    assert rounded_equal(pre_anom[0, 0:2], expected_rainfall_anom)
    assert sun[0, 0] == expected_sun
    assert sun_anom[0, 0] == expected_sun_anom

    # Extract the weather data from the previously created hdf5 file
    # Data should be consistent in both cases
    extracted_weather_data = read_or_create(cultivar, cultivar_data)
    tmi, tma, pre, pre_anom, sun, sun_anom = extracted_weather_data
    assert rounded_equal(tmi[0, 0:2], expected_temp_min)
    assert rounded_equal(tma[0, 0:2], expected_temp_max)
    assert rounded_equal(pre[0, 0:2], expected_rainfall)
    assert rounded_equal(pre_anom[0, 0:2], expected_rainfall_anom)
    assert sun[0, 0] == expected_sun
    assert sun_anom[0, 0] == expected_sun_anom
