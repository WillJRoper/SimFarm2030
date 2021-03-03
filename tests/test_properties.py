from os.path import abspath, dirname, join
import pandas as pd

PARENT_DIR = dirname(dirname(abspath(__file__)))


def groupby_build_cultivar_df(all_cults_df, cultivar="Claire"):
    cultivar_rows = []
    latlng_groups = all_cults_df.groupby(["Lat", "Long"])
    for (lat, lng), group in latlng_groups:
        for _, row in group.iterrows():
            if row.Cultivar == cultivar:
                cultivar_rows.append(row.to_dict())

    cultivar_df = pd.DataFrame.from_records(cultivar_rows)
    return cultivar_df


def test_all_cultivars_iteration():
    # Check that you can build equivalent cultivar dataframes, by
    # a) sorting on Lat, Lng, Year and filtering by cultivar
    # b) sorting on Lat, Lng, Year, grouping by LatLng and iterating through
    # the groups.
    # Cultivar data will be generate via a)
    # Weather data via b)
    # Both need to be ordered equivalently to run the model
    with open(
        join(
            PARENT_DIR,
            "All_Cultivars_Spreadsheets",
            "all_cultivars.csv")) as f:
        all_cults_df = pd.read_csv(f)

    ac_sorted_df = all_cults_df.sort_values(["Lat", "Long", "Year"])
    claire_df_a = ac_sorted_df[ac_sorted_df.Cultivar == "Claire"]
    claire_df_b = groupby_build_cultivar_df(ac_sorted_df)
    assert claire_df_b.equals(claire_df_a.reset_index(drop=True))
