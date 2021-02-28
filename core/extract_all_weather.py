from collections import defaultdict
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
            desc="Regions", colour="Yellow"):
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # cache_size is the closest power of 2 to
        # 3 months worth of sow days (Sep-Nov) + 400 grow days)
        @lru_cache(maxsize=512)
        def fetch_regional_weather(grow_day):
            weather_grid = hdf[grow_day]["daily_grid"]
            return extract_regional_weather(weather_grid, region_filter)

        for i, row in tqdm(
                group.iterrows(), total=group.shape[0],
                desc="Cultivar Years"):
            cultivar_data = dataset[row.Cultivar]
            rainfall = []
            anomaly = []
            for grow_date in generate_hdf_day_keys(row):
                rain = fetch_regional_weather(grow_date)
                rainfall.append(rain)
                _, _, day = grow_date.split("_")
                anom = rain - uk_monthly_mean[int(day) - 1]
                anomaly.append(anom)
            cultivar_data['rainfall_anomaly'].append(anomaly)
            cultivar_data['rainfall'].append(rainfall)

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
            desc="Regions", colour="Yellow"):
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # cache_size is the closest power of 2 to
        # 3 months worth of sow days (Sep-Nov) + 400 grow days)
        @lru_cache(maxsize=512)
        def fetch_regional_weather(grow_day):
            weather_grid = hdf[grow_day]["daily_grid"]
            return extract_regional_weather(weather_grid, region_filter)

        for i, row in tqdm(
                group.iterrows(), total=group.shape[0],
                desc="Cultivar Years"):
            cultivar_data = dataset[row.Cultivar]
            sunshine = []
            anomaly = []
            for grow_date in generate_hdf_month_keys(row):
                sun = fetch_regional_weather(grow_date)
                sunshine.append(sun)
                _, month = grow_date.split("_")
                anom = sun - uk_monthly_mean[int(month) - 1]
                anomaly.append(anom)
            cultivar_data['sunshine_anomaly'].append(anomaly)
            cultivar_data['sunshine'].append(sunshine)

    return dataset


def extract_temp(all_cultivars_df, hdf, temp_type, tol):
    dataset = defaultdict(lambda: defaultdict(list))
    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    latlng_groups = all_cultivars_df.groupby(["Lat", "Long"])
    num_groups = len(latlng_groups.size())
    for (lat, lng), group, in tqdm(
            latlng_groups, total=num_groups,
            desc="Regions", colour="Yellow"):
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # cache_size is the closest power of 2 to
        # 3 months worth of sow days (Sep-Nov) + 400 grow days)
        @lru_cache(maxsize=512)
        def fetch_regional_weather(grow_day):
            weather_grid = hdf[grow_day]["daily_grid"]
            return extract_regional_weather(weather_grid, region_filter)

        for i, row in tqdm(
                group.iterrows(), total=group.shape[0],
                desc="Cultivar Years"):
            cultivar_data = dataset[row.Cultivar]
            temperature = []
            for grow_date in generate_hdf_day_keys(row):
                temp = fetch_regional_weather(grow_date)
                temperature.append(temp)
            if temp_type == 'min':
                cultivar_data['temp_min'].append(temperature)
            else:
                cultivar_data['temp_max'].append(temperature)

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


if __name__ == '__main__':
    all_cultivars_df = extract_data(
        join(
            PARENT_DIR,
            "All_Cultivars_Spreadsheets",
            "all_cultivars.csv"))
    hdf = h5py.File(
        join(PARENT_DIR, "SimFarm2030_rainfall.hdf5"), "r")
    extract_rainfall(
        all_cultivars_df.sort_values(["Lat", "Long", "Year"]),
        hdf, tol=0.25)
