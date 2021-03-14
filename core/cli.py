import click
from model_daily_3d import cultivarModel
from extract_all_weather import fetch_weather
from cultivar_pandas_utils import extract_cultivar
from create_figures import create_all_plots

import numpy as np
import time
import pickle
from os.path import abspath, dirname, join


PARENT_DIR = dirname(dirname(abspath(__file__)))


@click.group()
def cli():
    pass


@cli.command()
# @click.argument()
def extract(simfarm):
    pass


@cli.command()
@click.argument('cultivar')
@click.option('--samples', default=75000, type=int, show_default=True) # minimum 
@click.option('--walkers', default=250, type=int, show_default=True) # minimum 18
def run(cultivar, samples, walkers):
    cultivar_weather_data = fetch_weather(cultivar)
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
