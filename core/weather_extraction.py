from os.path import abspath, dirname, join
import h5py
import numpy as np
import datetime
from collections import defaultdict
from ordered_set import OrderedSet

PARENT_DIR = dirname(dirname(abspath(__file__)))


# General Idea for weather extraction:
# Trying to build a 2d array of shape(lat-lon, num_day|num_months)
# i.e. each row corresponds to a "region (lat,lon)" and each
# column represents a mean weather value for day or month.
# It appears that max days = 400 and max months = 15

def read_from_existing_file(hdf):
    temp_max = hdf["Temperature_Maximum"][...]
    temp_min = hdf["Temperature_Minimum"][...]
    print("Temperature Extracted")
    precip_anom = hdf["Rainfall_Anomaly"][...]
    precip = hdf["Rainfall"][...]
    print("Rainfall Extracted")
    sun_anom = hdf["Sunshine_Anomaly"][...]
    sun = hdf["Sunshine"][...]
    print("Sunshine Extracted")
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
        lats, longs, years, ripe_days, _, sow_days, sow_months = regional_data
        sow_year = years - 1
        day_keys, month_keys = generate_hdf_keys(
            lats, longs, sow_year, sow_days, sow_months, ripe_days)
        data = extract_weather_data(
            filename, cult, lats, longs, sow_year,
            day_keys, month_keys, tol)
        write_dataset(filename, data)

    return data


def extract_weather_data(
        filename, cult, reg_lats, reg_longs, sow_year,
        reg_keys, reg_mth_keys, tol):
    print("Extracting meterological files")
    temp_max = get_temp(
        "tempmax", cult, reg_lats, reg_longs, sow_year, reg_keys, tol)
    temp_min = get_temp(
        "tempmin", cult, reg_lats, reg_longs, sow_year, reg_keys, tol)

    # Apply conditions from
    # https://ndawn.ndsu.nodak.edu/help-wheat-growing-degree-days.html
    temp_max[temp_max < 0] = 0
    temp_min[temp_min < 0] = 0
    print("Temperature Extracted")

    precip_anom, precip = get_weather_anomaly(
        "rainfall", cult, reg_lats, reg_longs, sow_year, reg_keys, tol)
    print("Rainfall Extracted")

    sun_anom, sun = get_weather_anomaly_monthly(
        "sunshine", cult, reg_lats, reg_longs, sow_year, reg_mth_keys, tol)
    print("Sunshine Extracted")

    # weather_anom_dict = {
    #     "rainfall": precip_anom,
    #     "sunshine": sun_anom
    # }  # unused ???
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


def generate_hdf_keys(
        reg_lats, reg_longs, sow_year, sow_days, sow_months, ripe_days):

    # Initialise the dictionary to hold keys
    day_keys_collection = defaultdict(dict)
    month_keys_collection = defaultdict(dict)

    # Loop over regions
    for (lat, long, sow_yr, sow_day, sow_month, ripe_time) in zip(
            reg_lats, reg_longs,
            sow_year, sow_days, sow_months,
            ripe_days):

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


def extract_region(lat, long, region_lat, region_long, weather, tol):

    # Get the boolean array for points within tolerence
    bool_cond = np.logical_and(
        np.abs(lat - region_lat) < tol, np.abs(long - region_long) < tol)

    # Get the extracted region
    ex_reg = weather[bool_cond]

    # Remove any nan and set them to 0 these correspond to ocean
    ex_reg = ex_reg[ex_reg < 1e8]

    if ex_reg.size == 0:
        print(f"Region not in coords: {region_lat} {region_long}")
        return np.nan
    else:
        return np.mean(ex_reg)


def get_temp(temp, cult, reg_lats, reg_longs, sow_year, reg_keys, tol):
    print(f'Getting the locations for new cultivar: {cult}')
    hdf = h5py.File(
        join(PARENT_DIR, "SimFarm2030_" + temp + ".hdf5"),
        "r")

    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    done_wthr = {}

    # Loop over regions
    print(f'Getting the temperature for those locations: {cult}')
    wthr = np.zeros((len(reg_lats), 400))
    for llind, (lat, long, year) in enumerate(
            zip(reg_lats, reg_longs, sow_year)):

        hdf_keys = reg_keys[str(lat) + "." + str(long)][str(year)]
        year_loc = f"{lat}_{long}_{year}"

        if year_loc in done_wthr:
            if tuple(hdf_keys) in done_wthr[year_loc]:
                print("Already extracted {year_loc}")
                wthr[llind, :] = \
                    done_wthr[year_loc][tuple(hdf_keys)]
                continue

        # Initialise arrays to hold results
        print(f'Initialising array: {llind}')
        key_ind = 0
        for key in hdf_keys:
            year, month, day = key.split("_")

            wthr_grid = hdf[key]["daily_grid"][...]

            ex_reg = extract_region(
                lats, longs, lat, long, wthr_grid, tol)

            # If year is within list of years extract the relevant data
            wthr[llind, key_ind] = ex_reg
            key_ind += 1

        done_wthr.setdefault(year_loc, {})[tuple(hdf_keys)] = wthr[llind, :]

    hdf.close()

    return wthr


def get_weather_anomaly(
        weather, cult, reg_lats, reg_longs, sow_year, reg_keys, tol):
    hdf = h5py.File(
        join(PARENT_DIR, "SimFarm2030_" + weather + ".hdf5"),
        "r")

    # Get the mean weather data for each month of the year
    uk_monthly_mean = hdf["all_years_mean"][...]

    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    done_wthr = {}

    # Loop over regions
    anom = np.full((len(reg_lats), 400), np.nan)
    wthr = np.full((len(reg_lats), 400), np.nan)
    for llind, (lat, long, year) in enumerate(
            zip(reg_lats, reg_longs, sow_year)):

        hdf_keys = reg_keys[str(lat) + "." + str(long)][str(year)]
        year_loc = f"{lat}_{long}_{year}"

        if year_loc in done_wthr:
            if tuple(hdf_keys) in done_wthr[year_loc]:
                print(f"Already extracted {year_loc}")
                wthr[llind, :] = done_wthr[year_loc][tuple(hdf_keys)]
                continue

        # Initialise arrays to hold results
        key_ind = 0
        for key in hdf_keys:
            year, month, day = key.split("_")

            wthr_grid = hdf[key]["daily_grid"][...]

            ex_reg = extract_region(
                lats, longs, lat, long, wthr_grid, tol)

            # If year is within list of years extract the relevant data
            wthr[llind, key_ind] = ex_reg
            anom[llind, key_ind] = ex_reg - uk_monthly_mean[int(day) - 1]
            key_ind += 1

        done_wthr.setdefault(year_loc, {})[
            tuple(hdf_keys)] = wthr[llind, :]

    hdf.close()
    return anom, wthr


def get_weather_anomaly_monthly(
        weather, cult, reg_lats, reg_longs, sow_year, reg_mth_keys, tol):

    hdf = h5py.File(
        join(PARENT_DIR, "SimFarm2030_" + weather + ".hdf5"),
        "r")

    # Get the mean weather data for each month of the year
    uk_monthly_mean = hdf["all_years_mean"][...]

    lats = hdf["Latitude_grid"][...]
    longs = hdf["Longitude_grid"][...]

    # Loop over regions
    anom = np.full((len(reg_lats), 15), np.nan)
    wthr = np.full((len(reg_lats), 15), np.nan)
    for llind, (lat, long, year) in enumerate(
            zip(reg_lats, reg_longs, sow_year)):

        hdf_keys = reg_mth_keys[str(lat) + "." + str(long)][str(year)]

        # Initialise arrays to hold results
        key_ind = 0
        for key in hdf_keys:
            year, month = key.split("_")

            wthr_grid = hdf[key]["monthly_grid"][...]

            ex_reg = extract_region(
                lats, longs, lat, long, wthr_grid, tol)

            # If year is within list of years extract the relevant data
            wthr[llind, key_ind] = ex_reg
            anom[llind, key_ind] = ex_reg - uk_monthly_mean[int(month) - 1]
            key_ind += 1

    hdf.close()
    return anom, wthr
