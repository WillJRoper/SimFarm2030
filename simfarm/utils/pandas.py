from datetime import datetime
import numpy as np
from os.path import abspath, dirname, join
import pandas as pd


# list of expected sow date formats AA
MONTH_DAY_NUMS = '%d/%m'
MONTH_DAY_WORDS = '%d-%b'


PARENT_DIR = dirname(dirname(dirname(abspath(__file__))))


def parse_date(date_string):
    for format_ in (MONTH_DAY_NUMS, MONTH_DAY_WORDS):
        try:
            return datetime.strptime(date_string, format_)
        except ValueError:
            continue
    else:
        raise ValueError(f'Unknown date format {date_string}')


def extract_cultivar(cultivar):
    all_data = extract_data(
        join(PARENT_DIR, "All_Cultivars_Spreadsheets", "all_cultivars.csv"))
    return all_data[all_data.Cultivar == cultivar]


def extract_data(path):
    # TODO: Validate the file:
    #     * Check for duplicate entries

    #  Open the csv file
    data = pd.read_csv(
        path,
        usecols=("Cultivar", "Lat", "Long", "Year", "Sow Month", "Ripe Time", "Yield"))

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

    print("Training on", data.shape[0], "Regions (data points)")
    # The order of records in the weather extraction data (temp, sun, rain)
    # have to be consistent with those in the cultivar data (yield)
    # both use this function, so we sort here.
    data = data.sort_values(["Lat", "Long", "Year"])
    return data