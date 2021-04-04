import pandas as pd
import numpy as np
from os.path import abspath, dirname, join
from simfarm.cultivar_pandas_utils import extract_data

PARENT_DIR = dirname(dirname(abspath(__file__)))
CUR_DIR = dirname(abspath(__file__))


def test_extract_data():
    cultivar_data = extract_data(join(CUR_DIR, 'all_cultivars.csv'))
    cultivar_data = cultivar_data.reset_index(drop=True)
    e_data = pd.DataFrame.from_records([
        {
            'Cultivar': 'Alchemy',
            'Lat': 52.7525,
            'Long': -0.9713,
            'Year': 2006,
            'Ripe Time': 297,  # rounded down from 297.67 (cast)
            'Yield': 9.75,
            'Sow Month': '10',
            'Sow Day': 1,
            'Sow Year': 2005,
        },
        {
            'Cultivar': 'Ambrosia',
            'Lat': 52.7525,
            'Long': -0.9713,
            'Year': 2006,
            'Ripe Time': 293,
            'Yield': 11.21,
            'Sow Month': '10',
            'Sow Day': 1,
            'Sow Year': 2005,
        }
    ])
    # FIXME: Possibly worth removing this casting in the
    # extract_data function.
    # Also Sow Day, Sow Month, Sow Year could be a single datetime
    # column.
    e_data['Ripe Time'] = e_data['Ripe Time'].astype(np.int32)
    e_data['Sow Day'] = e_data['Sow Day'].astype(np.int16)
    assert cultivar_data.equals(e_data)

# TODO:
# add test for extract_cultivar
# it filters the all_cultivars.csv returning only the rows
# for the specified cultivar
