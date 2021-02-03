from os.path import abspath, dirname, join
import pandas as pd

PARENT_DIR = dirname(dirname(abspath(__file__)))
YIELD_PATH = join(PARENT_DIR, "All_Cultivars_Spreadsheets", "yield_cleaned.csv")
RIPETIME_PATH = join(PARENT_DIR, "All_Cultivars_Spreadsheets", "ripe_time_cleaned.csv")


with open(RIPETIME_PATH) as rf, open(YIELD_PATH) as yf:
    rt_df = pd.read_csv(rf)
    y_df = pd.read_csv(yf)

    rt_comp_df = rt_df[
        ["Region", "Lat", "Long", "Year", "Sow Month"]
    ].reset_index(drop=True)
    y_comp_df = y_df[
        ["Region", "Lat", "Long", "Year", "Sow Month"]
    ].reset_index(drop=True)

    eq = rt_comp_df == y_comp_df
    assert(rt_comp_df.equals(y_comp_df))
