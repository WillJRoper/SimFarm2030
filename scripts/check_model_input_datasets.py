from os.path import abspath, dirname, join
import pandas as pd


CULTIVAR = "Claire"
PARENT_DIR = dirname(dirname(abspath(__file__)))
ALL_CULTIVARS_FILE = join(
    PARENT_DIR, "All_Cultivars_Spreadsheets", "all_cultivars.csv")
CULTIVAR_FILE = join(
    PARENT_DIR, "example_data", f"{CULTIVAR}_Data.csv")


if __name__ == '__main__':
    with open(ALL_CULTIVARS_FILE) as cs, open(CULTIVAR_FILE) as c:
        all_cultivars_df = pd.read_csv(cs)
        cultivar_df = pd.read_csv(c)

    # TODO: turn this next block into an assert
    # The code can help check for inconsistencies between the old
    # method of generating input (excel template)
    # vs the new (extract_cultivar.py)
    # -----------------------------------------------------------------------
    # e_cultivar_df = all_cultivars_df[all_cultivars_df.Cultivar == CULTIVAR]
    # e_with_yield = e_cultivar_df[e_cultivar_df.Yield.notna()]
    # with_yield = cultivar_df[cultivar_df.Yield.notna()]
    # e_sorted = e_with_yield.sort_values(['Region', "Year"])
    # sorte = with_yield.sort_values(['Region', 'Year'])
    # es = e_sorted[["Ripe Time", "Yield"]].reset_index(drop=True).fillna(0)
    # ss = sorte[["Ripe Time", "Yield"]].reset_index(drop=True).fillna(0)
    # eq = es.eq(ss)
