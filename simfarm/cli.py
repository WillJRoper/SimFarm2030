import click
from model_daily_3d import cultivarModel
from simfarm.extract.weather import fetch_weather, extract_all_weather
from cultivar_pandas_utils import extract_cultivar
from create_figures import create_all_plots

import numpy as np
import time
import pickle
from os.path import abspath, dirname, join


PARENT_DIR = dirname(dirname(abspath(__file__)))

ALL_CULTIVARS_CSV = join(
    PARENT_DIR, "All_Cultivars_Spreadsheets",
    "all_cultivars.csv")

RAINFALL_HDF = join(PARENT_DIR, "SimFarm2030_rainfall.hdf5")
TEMP_MIN_HDF = join(PARENT_DIR, "SimFarm2030_tempmin.hdf5")
TEMP_MAX_HDF = join(PARENT_DIR, "SimFarm2030_tempmax.hdf5")
SUNSHINE_HDF = join(PARENT_DIR, "SimFarm2030_sunshine.hdf5")

WEATHER_OUTPUT_HDF = join(
    PARENT_DIR, "Climate_Data", "all_cultivars_weather.hdf")
EXTRACTED_WEATHER_HDF = WEATHER_OUTPUT_HDF


@click.group()
def cli():
    pass


@cli.group()
def extract():
    pass


@extract.command()
@click.argument(
    'cultivars_csv', default=ALL_CULTIVARS_CSV, type=click.Path(exists=True))
@click.argument(
    'sunshine_datafile', default=SUNSHINE_HDF, type=click.Path(exists=True))
@click.argument(
    'tempmin_datafile', default=TEMP_MIN_HDF, type=click.Path(exists=True))
@click.argument(
    'tempmax_datafile', default=TEMP_MAX_HDF, type=click.Path(exists=True))
@click.argument(
    'rainfall_datafile', default=RAINFALL_HDF, type=click.Path(exists=True))
@click.argument(
    'output_file', default=WEATHER_OUTPUT_HDF, type=click.Path())
def weather(
        cultivars_csv,
        sunshine_datafile, tempmin_datafile,
        tempmax_datafile, rainfall_datafile,
        output_file):

    extract_all_weather(
        sunshine_datafile, tempmin_datafile,
        tempmax_datafile, rainfall_datafile,
        cultivars_csv, output_file)


# TODO:
# make command for (extract soil) when we have data.
# create data directories for running the first time


@cli.command()
@click.argument('cultivar')
@click.argument('weather_datafile', default=EXTRACTED_WEATHER_HDF, type=click.Path(exists=True))
@click.option('--samples', default=75000, type=int, show_default=True) # minimum 
@click.option('--walkers', default=250, type=int, show_default=True) # minimum 18
def run(cultivar, weather_datafile, samples, walkers):
    cultivar_weather_data = fetch_weather(cultivar, weather_datafile)
    cultivar_data = extract_cultivar(cultivar)
    print(cultivar_data)

    tstart = time.time()
    simfarm = cultivarModel(
        cultivar, cultivar_data, cultivar_weather_data,
        metric='Yield', metric_units='t Ha$^{-1}$')
    simfarm.train_and_validate_model(nsample=samples, nwalkers=walkers)
    print(f'Train in {(time.time() - tstart):.2} seconds')


    # Write out object as pickle
    with open(
        join(
            PARENT_DIR, 'cultivar_models',
            simfarm.cult + '_' + simfarm.metric + '_model_daily_3d.pck'),
            'wb') as pfile1:
        pickle.dump(simfarm, pfile1)


    # https://emcee.readthedocs.io/en/stable/tutorials/autocorr/ - is it steps?
    tau = simfarm.model.get_autocorr_time()
    print(
        f"Number of steps until the initial start is 'forgotten' "
        f"{np.round(tau, decimals=2)}")


@cli.command()
@click.argument('pickle_file', type=click.File('rb'))
def plot(pickle_file):
    create_all_plots(pickle.load(pickle_file))


if __name__ == '__main__':
    cli()
