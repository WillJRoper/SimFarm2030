import h5py
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join

import pytest

from core.cultivar_pandas_utils import extract_data, parse_date

PARENT_DIR = dirname(dirname(abspath(__file__)))


# To compare with the old weather data we need to
# use UNSORTED cultivar data. Which we then sort to get
# the index order changes for comparison with the new
# weather data
def extract_data_orig(path):
    # TODO: Validate the file:
    #     * Check for duplicate entries

    #  Open the csv file
    data = pd.read_csv(
        path,
        usecols=("Lat", "Long", "Year", "Sow Month", "Ripe Time", "Yield"))

    #  Remove NaN entries
    data.dropna(subset=["Yield"], inplace=True)

    # Replace missing ripe times
    data.fillna((data["Ripe Time"].mean()), inplace=True)

    # Set up a new data frame to extract day and month from sow date with split value columns
    date = data['Sow Month'].apply(parse_date)
    sow_day = date.dt.day
    sow_mth_ini = date.dt.month

    # Format months
    sow_months = sow_mth_ini.apply('{0:0>2}'.format)
    data.drop(columns=["Sow Month"], inplace=True)
    data["Sow Month"] = sow_months

    # add sow day column
    data["Sow Day"] = np.int16(sow_day)

    # cast ripetime to int32 (unsure why?)
    data['Ripe Time'] = data['Ripe Time'].astype(np.int32)

    # Define Sow Year
    # FIXME: is this correct? shouldn't we count backwards from Year
    # by the number of sow days to calculate the sow year??
    data["Sow Year"] = data["Year"] - 1

    # # Get extreme outliers
    # lim = np.mean(yields) * 0.75
    # okinds = yields > lim
    #
    # print("Removed", yields[~okinds].size, "outliers with yields < %.3f" % lim, "Tons / Hectare")

    # # Eliminate extreme outliers
    # lats = lats[okinds]
    # longs = longs[okinds]
    # years = years[okinds]
    # ripe_time = ripe_time[okinds]
    # yields = yields[okinds]
    # sow_day = sow_day[okinds]
    # sow_month = sow_month[okinds]

    print("Training on", data.shape[0], "Regions (data points)")
    # The order of records in the weather extraction data (temp, sun, rain)
    # have to be consistent with those in the cultivar data (yield)
    # both use this function, so we sort here.
    return data


@pytest.mark.skip(
    reason="Manual verification needed")
def test_old_vs_new_weather_extraction():
    # This was all run in the interpreter manually to spot check
    # the new extracted weather. There are enough differences between
    # the old extract and new to make it difficult to verify with
    # automation. This test documents the steps required to do some
    # spot checking manually. We are happy that the diffs aren't so
    # significant to think that we have messed up the extract logic.
    # We expect many differences because there are known errors in
    # the old logic
    hdf_claire = h5py.File(
        join(PARENT_DIR, 'Climate_Data', 'Region_Climate_Claire.hdf5'),
        "r")
    hdf_all = h5py.File(
        join(PARENT_DIR, 'Climate_Data', 'all_cultivars_weather.hdf'),
        "r")

    claire_temp_orig = hdf_claire["Temperature_Maximum"][()]
    claire_temp_new = hdf_all["Claire"]["Temperature_Maximum"][()]

    _, orig_cols = claire_temp_orig.shape
    _, new_cols = claire_temp_new.shape

    # truncate the columns of the original extract data
    # they were hard-coded at 400
    claire_temp_orig = np.delete(
        claire_temp_orig, list(range(new_cols, orig_cols)),
        axis=1)

    claire_df_orig = extract_data_orig(
        join(PARENT_DIR, "example_data", "Claire_Data.csv")
    ).reset_index(drop=True)
    sorted_orig = claire_df_orig.sort_values(["Lat", "Long", "Year"])

    all_df_new = extract_data(
        join(PARENT_DIR, "All_Cultivars_Spreadsheets", "all_cultivars.csv"))
    claire_df_new = all_df_new[
        all_df_new.Cultivar.eq("Claire")].reset_index(drop=True)

    # When the Sow Day is <= 10 in the old weather extraction, we
    # have known incorrect values, because the month is accidently
    # mixed up with the day. So weather is fetched from 10th Jan
    # Instead of 1st Oct for example. All Sow Dates should be in
    # Autumn.
    orig_idx = sorted_orig[(sorted_orig['Sow Day'] > 10)].index.values
    new_idx = claire_df_new[(claire_df_new['Sow Day'] > 10)].index.values
    # shape was not equal, new had an extra row here
    # old (37, 355), new (38, 355)
    claire_df_new.iloc[26]

    claire_some = claire_temp_new[new_idx]
    claire_some_old = claire_temp_orig[orig_idx]
    # delete the extra row mentioned above
    a = np.delete(claire_some, 26, 0)
    diff = a - claire_some_old
    print(diff[0])
    # iterate through diff inspecting the changes
    # on manual inspection we sae the following:
    # * 2 extra values (probably Ripe time changes from 309 to 311)
    # * lots of records with no difference at all
    # * some records with a small handful of difference scattered
    # randomly
