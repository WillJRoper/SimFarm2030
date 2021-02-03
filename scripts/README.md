# Scripts

This directory contains a series of small python scripts which
are used to clean the raw cultivar data and generate model
input datasets.

The following process was used to generate the `all_cultivars.csv` dataset:
* Run `sort_cultivar_data.py` to get the Ripe Time and Yield files ordered consistently,
  OUTPUT, ripe_time_cleaned.csv, yield_cleaned.csv
* MANUALY rename and reorder a few remaining regions that were out of order
* MANUALY correct a few lat lngs records
* Run `check_cultivar_csv_inconsistency.py` to validate the ripe_time_cleaned.csv and yield_cleaned.csv
* Run `fix_cultivar_csv_months.py` to substitute the yield_cleaned.csv Sow Month column with that of
  the ripe_time_cleaned.csv
* Run `extract_cultivars.py` to generate `all_cultivars.csv` from the two cleaned csv files
