from collections import defaultdict
from functools import partial
from dataclasses import dataclass, field
from os.path import abspath, dirname, join
import pandas as pd
import numpy as np


PARENT_DIR = dirname(dirname(abspath(__file__)))
YIELD_PATH = join(
    PARENT_DIR, "All_Cultivars_Spreadsheets", "yield_cleaned.csv")
RIPETIME_PATH = join(
    PARENT_DIR, "All_Cultivars_Spreadsheets", "ripe_time_cleaned.csv")

ALL_CULTIVARS_FILE = join(
    PARENT_DIR, "All_Cultivars_Spreadsheets", "all_cultivars.csv")


@dataclass
class CultivarRecord:
    region: str
    county: str
    lat: float
    lng: float
    year: int
    month: str
    cultivar: str = field(init=False, default=None)
    sow_yield: float = field(init=False, default=np.NaN)
    ripe_time: float = field(init=False, default=np.NaN)

    def add_yield(self, yield_val):
        self.sow_yield = yield_val

    def add_ripe_time(self, ripe_time):
        self.ripe_time = ripe_time

    def add_cultivar(self, cultivar):
        if self.cultivar:
            # if yield already set the cultivar
            # check ripetime is passing in the same name
            assert(self.cultivar == cultivar)
        else:
            self.cultivar = cultivar


def get_group_rows(groups, group_name):
    try:
        group = groups.get_group(group_name)
    except KeyError:
        return iter([])
    else:
        return group.iterrows()


def process_group(records, group_key, yield_groups, ripe_groups):
    yields = get_group_rows(yield_groups, group_key)
    ripe_times = get_group_rows(ripe_groups, group_key)

    records = {}
    for _, data in yields:
        key = (data.Year, data['Sow Month'])
        region_data, cultivar_data = data[:6], data[6:]
        sow_date_records = records.get(
            key, defaultdict(partial(CultivarRecord, *region_data)))
        records[key] = add_cultivar_yields(cultivar_data, sow_date_records)

    for _, data in ripe_times:
        key = (data.Year, data['Sow Month'])
        region_data, cultivar_data = data[:6], data[6:]
        sow_date_records = records.get(
            key, defaultdict(partial(CultivarRecord, *region_data)))
        records[key] = add_cultivar_ripe_times(cultivar_data, sow_date_records)

    return [
        cultivar_record
        for sow_date_record in records.values()
        for cultivar_record in sow_date_record.values()]


def add_cultivar_yields(yields, records):
    for cultivar, val in yields.dropna().iteritems():
        cultivar = cultivar.title()
        records[cultivar].add_yield(val)
        records[cultivar].add_cultivar(cultivar)
    return records


def add_cultivar_ripe_times(ripe_times, records):
    for cultivar, val in ripe_times.dropna().iteritems():
        cultivar = cultivar.title()
        records[cultivar].add_ripe_time(val)
        records[cultivar].add_cultivar(cultivar)
    return records


def mush(yield_df, ripe_time_df):
    cultivar_records = {}
    yield_groups = yield_df.sort_values(
        ["Lat", "Long"]).groupby(['Lat', 'Long'])
    ripe_groups = ripe_time_df.sort_values(
        ["Lat", "Long"]).groupby(['Lat', 'Long'])
    all_keys = set(ripe_groups.groups.keys()) | set(yield_groups.groups.keys())
    records = []
    for key in all_keys:
        records.extend(
            process_group(cultivar_records, key, yield_groups, ripe_groups))
    return records


if __name__ == '__main__':
    with open(YIELD_PATH) as yf, open(RIPETIME_PATH) as rtf:
        yield_data = pd.read_csv(yf)
        ripe_time_data = pd.read_csv(rtf)

    records = mush(yield_data, ripe_time_data)
    cultivars_df = pd.DataFrame(
        [
            (
                r.cultivar,
                r.region, r.county, r.lat, r.lng,
                r.year, r.month,
                r.ripe_time, r.sow_yield
            )
            for r in records
        ],
        columns=[
            "Cultivar", "Region", "Region County", "Lat", "Long",
            "Year", "Sow Month", "Ripe Time", "Yield"]
    )
    with open(ALL_CULTIVARS_FILE, "w") as f:
        cultivars_df['Cultivar'] = cultivars_df['Cultivar'].str.title()
        cultivars_df.sort_values("Cultivar").to_csv(f, index=False)
