from collections import defaultdict
from contextlib import contextmanager
from datetime import date, timedelta
from functools import lru_cache
import h5py
from os.path import abspath, dirname, join
import numpy as np
from itertools import zip_longest
from tqdm import tqdm

from utilities import extract_data
from weather_extraction import create_region_filter, extract_regional_weather

PARENT_DIR = dirname(dirname(abspath(__file__)))

RAINFALL_HDF = join(PARENT_DIR, "SimFarm2030_rainfall.hdf5")
TEMP_MIN_HDF = join(PARENT_DIR, "SimFarm2030_tempmin.hdf5")
TEMP_MAX_HDF = join(PARENT_DIR, "SimFarm2030_tempmax.hdf5")
SUNSHINE_HDF = join(PARENT_DIR, "SimFarm2030_sunshine.hdf5")

WEATHER_OUTPUT_HDF = join(
    PARENT_DIR, "Climate_Data", "all_cultivars_weather.hdf")
EXTRACTED_WEATHER_HDF = WEATHER_OUTPUT_HDF


def extract_rainfall(all_cultivars_df, hdf, tol):
    dataset = defaultdict(lambda: defaultdict(list))
    # Get the mean weather data for each month of the year
    uk_monthly_mean = hdf["all_years_mean"][...]
    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    latlng_groups = all_cultivars_df.groupby(["Lat", "Long"])
    num_groups = len(latlng_groups.size())
    for (lat, lng), group, in tqdm(
            latlng_groups, total=num_groups,
            desc="Rainfall", colour="Blue"):
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # cache_size is the closest power of 2 to
        # 3 months worth of sow days (Sep-Nov) + 400 grow days)
        @lru_cache(maxsize=512)
        def fetch_regional_weather(grow_day):
            weather_grid = hdf[grow_day]["daily_grid"]
            return extract_regional_weather(weather_grid, region_filter)

        for i, row in group.iterrows():
            cultivar_data = dataset[row.Cultivar]
            rainfall = []
            anomaly = []
            for grow_date in generate_hdf_day_keys(row):
                rain = fetch_regional_weather(grow_date)
                rainfall.append(rain)
                _, _, day = grow_date.split("_")
                anom = rain - uk_monthly_mean[int(day) - 1]
                anomaly.append(anom)
            cultivar_data['Rainfall_Anomaly'].append(anomaly)
            cultivar_data['Rainfall'].append(rainfall)

    return dataset


def extract_sunshine(all_cultivars_df, hdf, tol):
    dataset = defaultdict(lambda: defaultdict(list))
    # Get the mean weather data for each month of the year
    uk_monthly_mean = hdf["all_years_mean"][...]
    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    latlng_groups = all_cultivars_df.groupby(["Lat", "Long"])
    num_groups = len(latlng_groups.size())
    for (lat, lng), group, in tqdm(
            latlng_groups, total=num_groups,
            desc="Sunshine", colour="Yellow"):
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # cache_size is the closest power of 2 to
        # 3 months worth of sow days (Sep-Nov) + 400 grow days)
        @lru_cache(maxsize=512)
        def fetch_regional_weather(grow_month):
            weather_grid = hdf[grow_month]["monthly_grid"]
            return extract_regional_weather(weather_grid, region_filter)

        for i, row in group.iterrows():
            cultivar_data = dataset[row.Cultivar]
            sunshine = []
            anomaly = []
            for grow_date in generate_hdf_month_keys(row):
                sun = fetch_regional_weather(grow_date)
                sunshine.append(sun)
                _, month = grow_date.split("_")
                anom = sun - uk_monthly_mean[int(month) - 1]
                anomaly.append(anom)
            cultivar_data['Sunshine_Anomaly'].append(anomaly)
            cultivar_data['Sunshine'].append(sunshine)

    return dataset


def extract_temp(all_cultivars_df, hdf, temp_type, tol):
    dataset = defaultdict(lambda: defaultdict(list))
    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    latlng_groups = all_cultivars_df.groupby(["Lat", "Long"])
    num_groups = len(latlng_groups.size())
    for (lat, lng), group, in tqdm(
            latlng_groups, total=num_groups,
            desc="Regions", colour="Red"):
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # cache_size is the closest power of 2 to
        # 3 months worth of sow days (Sep-Nov) + 400 grow days)
        @lru_cache(maxsize=512)
        def fetch_regional_weather(grow_day):
            weather_grid = hdf[grow_day]["daily_grid"]
            return extract_regional_weather(weather_grid, region_filter)

        for i, row in group.iterrows():
            cultivar_data = dataset[row.Cultivar]
            temperature = []
            for grow_date in generate_hdf_day_keys(row):
                temp = fetch_regional_weather(grow_date)
                temperature.append(temp)
            if temp_type == 'min':
                cultivar_data['Temperature_Minimum'].append(temperature)
            else:
                cultivar_data['Temperature_Maximum'].append(temperature)

    return dataset


def generate_hdf_day_keys(cultivar_row):
    sow_date = date(
        year=cultivar_row["Sow Year"],
        month=int(cultivar_row["Sow Month"]),
        day=int(cultivar_row["Sow Day"]))
    ripe_time = cultivar_row["Ripe Time"]
    for nday in range(ripe_time + 1):
        grow_day = sow_date + timedelta(days=nday)
        yield f"{grow_day.year}_{grow_day.month:03}_{grow_day.day:04}"


def generate_hdf_month_keys(cultivar_row):
    sow_date = date(
        year=cultivar_row["Sow Year"],
        month=int(cultivar_row["Sow Month"]),
        day=int(cultivar_row["Sow Day"]))
    ripe_time = cultivar_row["Ripe Time"]

    yield f"{sow_date.year}_{sow_date.month:03}"
    prev_month = sow_date.month
    for nday in range(1, ripe_time + 1):
        grow_day = sow_date + timedelta(days=nday)
        if grow_day.month != prev_month:
            yield f"{grow_day.year}_{grow_day.month:03}"
            prev_month = grow_day.month


def to_np_array(array):
    return np.array(list(zip_longest(*array, fillvalue=0))).T


def map_dict(f, dictionary):
    return {key: f(value) for key, value in dictionary.items()}


def nested_to_np_array(dictionary):
    return map_dict(to_np_array, dictionary)


@contextmanager
def hdf_open(filename, access="r"):
    hdf = h5py.File(filename, access)
    yield hdf
    hdf.close()


def get_or_add_group(hdf_file, group_name):
    try:
        return hdf_file[group_name]
    except KeyError:
        return hdf_file.create_group(group_name)


def write_weather_to_hdf(output_hdf, cultivars_weather_data):
    for cultivar, weather_data in cultivars_weather_data.items():
        cultivar_group = get_or_add_group(output_hdf, cultivar)
        for name, data in weather_data.items():
            cultivar_group.create_dataset(
                name,
                shape=data.shape,
                dtype=data.dtype,
                data=data, compression="gzip")


def extract_all_weather(all_cultivars_df, tol=0.25):
    with hdf_open(WEATHER_OUTPUT_HDF, access="a") as outfile:

        with hdf_open(TEMP_MAX_HDF) as f:
            temp_max = extract_temp(all_cultivars_df, f, "max", tol)
        temp_max = map_dict(nested_to_np_array, temp_max)
        write_weather_to_hdf(outfile, temp_max)

        with hdf_open(TEMP_MIN_HDF) as f:
            temp_min = extract_temp(all_cultivars_df, f, "min", tol)
        temp_min = map_dict(nested_to_np_array, temp_min)
        write_weather_to_hdf(outfile, temp_min)

        with hdf_open(RAINFALL_HDF) as f:
            rainfall = extract_rainfall(all_cultivars_df, f, tol)
        rainfall = map_dict(nested_to_np_array, rainfall)
        write_weather_to_hdf(outfile, rainfall)

        with hdf_open(SUNSHINE_HDF) as f:
            sunshine = extract_sunshine(all_cultivars_df, f, tol)
        sunshine = map_dict(nested_to_np_array, sunshine)
        write_weather_to_hdf(outfile, sunshine)


def fetch_weather(cultivar, extractor_f):
    with hdf_open(EXTRACTED_WEATHER_HDF) as f:
        cultivar_data = extractor_f(f[cultivar])
    return cultivar_data


if __name__ == '__main__':
    all_cultivars_df = extract_data(
        join(
            PARENT_DIR,
            "All_Cultivars_Spreadsheets",
            "all_cultivars.csv"))
    extract_all_weather(all_cultivars_df)
