from core.cultivar_pandas_utils import extract_data
from os.path import abspath, dirname, join
import pandas as pd
import numpy as np

PARENT_DIR = dirname(dirname(abspath(__file__)))


def test_extract():
    df = extract_data(
        join(PARENT_DIR, "example_data", "Test_Single_Data.csv"))
    expected_df = pd.DataFrame.from_dict({
        "Lat": [52.0834], "Long": [-1.4545],
        "Year": [2013], "Ripe Time": [1],
        "Yield": [8.76], "Sow Month": ['09'], "Sow Day": [10],
        "Sow Year": [2012]
    })
    expected_df['Sow Day'] = expected_df['Sow Day'].astype(np.int16)
    expected_df['Ripe Time'] = expected_df['Ripe Time'].astype(np.int32)
    assert df.equals(expected_df)
