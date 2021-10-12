from simfarm.utils.hdf5 import hdf_open, write_weather_to_hdf
from simfarm.utils.pandas import extract_data

from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from os.path import abspath, dirname, join
import numpy as np
from itertools import zip_longest
from tqdm import tqdm


PARENT_DIR = dirname(dirname(abspath(__file__)))

WEATHER_OUTPUT_HDF = join(
    PARENT_DIR, "Climate_Data", "all_cultivars_weather.hdf")
EXTRACTED_WEATHER_HDF = WEATHER_OUTPUT_HDF


def create_region_filter(lat_grid, lng_grid, lat, lng, tolerance):
    """Create a filter grid (or boolean mask) for the region surrounding
    the lat,lng of interest that fits within the tolerance.

    The boolean mask is then used to extract regional weather from a 2D weather
    array. For example:
    boolean_mask = np.array([[True, True], [False, True]])
    Weather_array = np.array([1, 2], [3, 4])
    weather_array[boolean_mask] = np.array([1, 2, 4])
    """
    return np.logical_and(
        np.abs(lat_grid - lat) < tolerance, np.abs(lng_grid - lng) < tolerance)


def extract_regional_weather(weather, region_filter):
    # Get the extracted region
    ex_reg = weather[region_filter]
    # Remove any nan and set them to 0 these correspond to ocean
    ex_reg = ex_reg[ex_reg < 1e8]

    if ex_reg.size == 0:
        # FIXME:
        # can't warn about this now:
        # print(f"Region not in coords: {region_lat} {region_long}")
        return np.nan
    else:
        return np.mean(ex_reg)


def extract_rainfall(all_cultivars_df, hdf, tol):
    # End structure
    # {"Cultivar": {"Rainfall": [for each (Lat,Lng,Year)[rainfall-on-day]]}
    #              {"Rainfall_Anomoly": [[]]}}
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


def extract_all_weather(
        sunshine_datafile, tempmin_datafile,
        tempmax_datafile, rainfall_datafile,
        all_cultivars_csv, output_file, tol=0.25):
    all_cultivars_df = extract_data(all_cultivars_csv)
    with hdf_open(output_file, access="a") as outfile:

        with hdf_open(tempmax_datafile) as f:
            temp_max = extract_temp(all_cultivars_df, f, "max", tol)
        temp_max = map_dict(nested_to_np_array, temp_max)
        write_weather_to_hdf(outfile, temp_max)

        with hdf_open(tempmin_datafile) as f:
            temp_min = extract_temp(all_cultivars_df, f, "min", tol)
        temp_min = map_dict(nested_to_np_array, temp_min)
        write_weather_to_hdf(outfile, temp_min)

        with hdf_open(rainfall_datafile) as f:
            rainfall = extract_rainfall(all_cultivars_df, f, tol)
        rainfall = map_dict(nested_to_np_array, rainfall)
        write_weather_to_hdf(outfile, rainfall)

        with hdf_open(sunshine_datafile) as f:
            sunshine = extract_sunshine(all_cultivars_df, f, tol)
        sunshine = map_dict(nested_to_np_array, sunshine)
        write_weather_to_hdf(outfile, sunshine)


def read_from_existing_file(hdf):
    temp_max = hdf["Temperature_Maximum"][...]
    temp_min = hdf["Temperature_Minimum"][...]
    precip_anom = hdf["Rainfall_Anomaly"][...]
    precip = hdf["Rainfall"][...]
    sun_anom = hdf["Sunshine_Anomaly"][...]
    sun = hdf["Sunshine"][...]
    return temp_min, temp_max, precip, precip_anom, sun, sun_anom


def fetch_weather(cultivar, weather_datafile):
    with hdf_open(weather_datafile) as f:
        cultivar_data = read_from_existing_file(f[cultivar])
    return cultivar_data
