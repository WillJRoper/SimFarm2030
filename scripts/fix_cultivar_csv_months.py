from os.path import abspath, dirname, join
import pandas as pd

PARENT_DIR = dirname(dirname(abspath(__file__)))
YIELD_PATH = join(
    PARENT_DIR, "All_Cultivars_Spreadsheets", "yield_cleaned.csv")
RIPETIME_PATH = join(
    PARENT_DIR, "All_Cultivars_Spreadsheets", "ripe_time_cleaned.csv")

if __name__ == '__main__':
    with open(RIPETIME_PATH) as rf, open(YIELD_PATH) as yf:
        ripe_time_df = pd.read_csv(rf)
        yield_df = pd.read_csv(yf)

    yield_df["Sow Month"] = ripe_time_df["Sow Month"]

    with open(YIELD_PATH, "w") as yf:
        yield_df.to_csv(yf, index=False)
