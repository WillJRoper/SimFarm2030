from os.path import abspath, dirname, join
import h5py
import numpy as np
import datetime
from collections import defaultdict
from ordered_set import OrderedSet
from tqdm import tqdm

PARENT_DIR = dirname(dirname(abspath(__file__)))


# General Idea for weather extraction:
# Trying to build a 2d array of shape(lat-lon, num_day|num_months)
# i.e. each row corresponds to a "region (lat,lon)" and each
# column represents a mean weather value for day or month.
# It appears that max days = 400 and max months = 15

def read_from_existing_file(hdf):
    temp_max = hdf["Temperature_Maximum"][...]
    temp_min = hdf["Temperature_Minimum"][...]
    precip_anom = hdf["Rainfall_Anomaly"][...]
    precip = hdf["Rainfall"][...]
    sun_anom = hdf["Sunshine_Anomaly"][...]
    sun = hdf["Sunshine"][...]
    return temp_min, temp_max, precip, precip_anom, sun, sun_anom


def read_or_create(
        cult, regional_data, tol=0.25, extract_flag=False):
    """
    If a regional hdf5 file exists, read in the data.
    Otherwise extract regional data from the weather hdf5 files
    and create a new regional hdf5 file.
    """
    filename = join(
        PARENT_DIR, "Climate_Data", "Region_Climate_" + cult + ".hdf5")
    try:
        hdf = h5py.File(filename, "r")
    except OSError:
        extract_flag = True
        print(f"File {filename} not found")
    else:
        try:
            data = read_from_existing_file(hdf)
        except KeyError:
            extract_flag = True
            print("Key not found, previous extraction incomplete")
        finally:
            hdf.close()

    if extract_flag:
        day_keys, month_keys = generate_hdf_keys(regional_data)
        data = extract_weather_data(
            filename, cult, regional_data,
            day_keys, month_keys, tol)
        write_dataset(filename, data)

    return data


def extract_weather_data(
        filename, cult, regional_data,
        reg_keys, reg_mth_keys, tol):
    print(f"Extracting meterological data for {cult}")
    temp_max = get_temp(
        "tempmax", cult, regional_data, reg_keys, tol)
    temp_min = get_temp(
        "tempmin", cult, regional_data, reg_keys, tol)

    # Apply conditions from
    # https://ndawn.ndsu.nodak.edu/help-wheat-growing-degree-days.html
    temp_max[temp_max < 0] = 0
    temp_min[temp_min < 0] = 0

    precip_anom, precip = get_weather_anomaly_daily(
        "rainfall", cult, regional_data, reg_keys, tol)

    sun_anom, sun = get_weather_anomaly_monthly(
        "sunshine", cult, regional_data, reg_mth_keys, tol)

    return temp_min, temp_max, precip, precip_anom, sun, sun_anom


def write_dataset(filename, data):
    temp_min, temp_max, precip, precip_anom, sun, sun_anom = data
    hdf = h5py.File(filename, "w")

    hdf.create_dataset(
        "Temperature_Maximum",
        shape=temp_max.shape,
        dtype=temp_max.dtype,
        data=temp_max, compression="gzip")
    hdf.create_dataset(
        "Temperature_Minimum",
        shape=temp_min.shape,
        dtype=temp_min.dtype,
        data=temp_min, compression="gzip")
    hdf.create_dataset(
        "Rainfall", shape=precip.shape,
        dtype=precip.dtype,
        data=precip, compression="gzip")
    hdf.create_dataset(
        "Rainfall_Anomaly",
        shape=precip_anom.shape,
        dtype=precip_anom.dtype,
        data=precip_anom, compression="gzip")
    hdf.create_dataset(
        "Sunshine", shape=sun.shape,
        dtype=sun.dtype,
        data=sun, compression="gzip")
    hdf.create_dataset(
        "Sunshine_Anomaly",
        shape=sun_anom.shape,
        dtype=sun_anom.dtype,
        data=sun_anom, compression="gzip")

    hdf.close()


def generate_hdf_keys(regional_df):
    # reg_lats, reg_longs, sow_year, sow_days, sow_months, ripe_days):

    # Initialise the dictionary to hold keys
    day_keys_collection = defaultdict(dict)
    month_keys_collection = defaultdict(dict)

    # Loop over regions
    for _, row in regional_df.iterrows():
        sow_yr = row["Sow Year"]
        sow_month = row["Sow Month"]
        sow_day = row["Sow Day"]
        ripe_time = row["Ripe Time"]
        lat = row["Lat"]
        long = row["Long"]

        # Extract the sow day for this year
        sow_date = datetime.date(
            year=sow_yr, month=int(sow_month), day=int(sow_day))

        # Initialise this region"s day and month keys
        day_keys = OrderedSet()
        month_keys = OrderedSet()

        # Loop over the days between sowing and ripening
        for nday in range(ripe_time + 1):
            # Compute the grow day since sowing
            grow_day = sow_date + datetime.timedelta(days=nday)

            day_key = f"{grow_day.year}_{grow_day.month:03}_{grow_day.day:04}"
            month_key = f"{grow_day.year}_{grow_day.month:03}"

            # Append keys
            day_keys.add(day_key)
            month_keys.add(month_key)

        # Assign keys to dictionary
        day_keys_collection[f"{lat}.{long}"][f"{sow_yr}"] = day_keys
        month_keys_collection[f"{lat}.{long}"][f"{sow_yr}"] = month_keys

    return day_keys_collection, month_keys_collection


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


def get_temp(temp, cult, regional_df, day_keys, tol):
    hdf = h5py.File(
        join(PARENT_DIR, "SimFarm2030_" + temp + ".hdf5"),
        "r")

    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    # Loop over regions
    num_records = regional_df.shape[0]
    temps = np.zeros((num_records, 400))
    for llind, row in tqdm(
            regional_df.iterrows(), total=num_records,
            desc=f"Extracting {temp}", colour='red'):
        lat = row["Lat"]
        lng = row["Long"]
        year = row["Sow Year"]

        grow_days = day_keys[f"{lat}.{lng}"][f"{year}"]
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # Initialise arrays to hold results
        for grow_day_idx, grow_date in enumerate(grow_days):
            _, _, day = grow_date.split("_")

            wthr_grid = hdf[grow_date]["daily_grid"][...]
            day_reg_temp = extract_regional_weather(wthr_grid, region_filter)

            # If year is within list of years extract the relevant data
            temps[llind, grow_day_idx] = day_reg_temp

    hdf.close()
    return temps


# used to extract rainfall
def get_weather_anomaly_daily(
        weather, cult, regional_df, day_keys, tol):
    hdf = h5py.File(
        join(PARENT_DIR, "SimFarm2030_" + weather + ".hdf5"),
        "r")

    # Get the mean weather data for each month of the year
    uk_monthly_mean = hdf["all_years_mean"][...]

    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    # Loop over regions
    num_records = regional_df.shape[0]
    anom = np.full((num_records, 400), np.nan)
    wthr = np.full((num_records, 400), np.nan)
    for llind, row in tqdm(
            regional_df.iterrows(), total=num_records,
            desc=f"Extracting {weather}", colour='blue'):
        lat = row["Lat"]
        lng = row["Long"]
        year = row["Sow Year"]

        grow_days = day_keys[f"{lat}.{lng}"][f"{year}"]
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # Initialise arrays to hold results
        for grow_day_idx, grow_date in enumerate(grow_days):
            _, _, day = grow_date.split("_")

            wthr_grid = hdf[grow_date]["daily_grid"][...]

            reg_weather = extract_regional_weather(wthr_grid, region_filter)

            # If year is within list of years extract the relevant data
            wthr[llind, grow_day_idx] = reg_weather
            anom[llind, grow_day_idx] = reg_weather - uk_monthly_mean[int(day) - 1]

    hdf.close()
    return anom, wthr


# used to extract sunshine and the anomolies (difference between actual and mean for each point)
def get_weather_anomaly_monthly(
        weather, cult, regional_df, month_keys, tol):

    hdf = h5py.File(
        join(PARENT_DIR, "SimFarm2030_" + weather + ".hdf5"),
        "r")

    # Get the mean weather data for each month of the year
    uk_monthly_mean = hdf["all_years_mean"][...]

    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    # Loop over regions
    num_records = regional_df.shape[0]
    anom = np.full((num_records, 15), np.nan)
    wthr = np.full((num_records, 15), np.nan)
    for llind, row in tqdm(
            regional_df.iterrows(), total=num_records,
            desc=f"Extracting {weather}", colour='yellow'):
        lat = row["Lat"]
        lng = row["Long"]
        year = row["Sow Year"]

        grow_months = month_keys[f"{lat}.{lng}"][f"{year}"]
        region_filter = create_region_filter(lats, longs, lat, lng, tol)

        # Initialise arrays to hold results
        for grow_month_index, grow_date in enumerate(grow_months):
            _, month = grow_date.split("_")

            wthr_grid = hdf[grow_date]["monthly_grid"][...]

            reg_weather = extract_regional_weather(wthr_grid, region_filter)

            # If year is within list of years extract the relevant data
            wthr[llind, grow_month_index] = reg_weather
            anom[llind, grow_month_index] = reg_weather - uk_monthly_mean[int(month) - 1]

    hdf.close()
    return anom, wthr
