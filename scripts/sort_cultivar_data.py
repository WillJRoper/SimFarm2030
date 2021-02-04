from os.path import abspath, dirname, join
import pandas as pd

PARENT_DIR = dirname(dirname(abspath(__file__)))
YIELD_PATH = join(PARENT_DIR, "All_Cultivars_Spreadsheets", "Yield.csv")
RIPETIME_PATH = join(PARENT_DIR, "All_Cultivars_Spreadsheets", "Ripe Time.csv")
CLEANED_YIELD = join(PARENT_DIR, "All_Cultivars_Spreadsheets", "yield_cleaned.csv")
CLEANED_RIPE = join(PARENT_DIR, "All_Cultivars_Spreadsheets", "ripe_time_cleaned.csv")


def sort_df(df):
    df["RegionLower"] = df["Region"].str.lower()
    df = df.sort_values(['RegionLower', 'Year'])
    del(df["RegionLower"])
    return df


if __name__ == '__main__':
    with open(RIPETIME_PATH) as rf, open(YIELD_PATH) as yf:
        ripe_time_df = pd.read_csv(rf)
        yield_df = pd.read_csv(yf)

    rt_df = sort_df(ripe_time_df)
    rt_comp_df = rt_df[["Region", "Year"]].reset_index(drop=True)
    y_df = sort_df(yield_df)
    y_comp_df = y_df[["Region", "Year"]].reset_index(drop=True)

    with open(CLEANED_YIELD, "w") as y, open(CLEANED_RIPE, "w")  as r:
        y_df.to_csv(y, index=False)
        rt_df.to_csv(r, index=False)
