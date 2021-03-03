from model_daily_3d import cultivarModel
from extract_all_weather import fetch_weather
from cultivar_pandas_utils import extract_cultivar

import time
import pickle
import sys
import os
from os.path import abspath, dirname, join


PARENT_DIR = dirname(dirname(abspath(__file__)))


def get_non_hidden_filepaths():
    return [
        f for f in os.listdir(join(PARENT_DIR, 'example_data'))
        if not f.startswith('.')
    ]


if __name__ == "__main__":
    # Extract cultivar from command line input
    cultivar = sys.argv[1]

    cultivar_weather_data = fetch_weather(cultivar)
    cultivar_data = extract_cultivar(cultivar)
    print(cultivar_data)

    tstart = time.time()
    simfarm = cultivarModel(
        cultivar, cultivar_data, cultivar_weather_data,
        metric='Yield', metric_units='t Ha$^{-1}$')
    simfarm.train_and_validate_model(nsample=75000, nwalkers=250)
    print(f'Train in {(time.time() - tstart):.2} seconds')

    # Moved plotting functions -> create_figures.py revisit after tests created
    # simfarm.plot_walkers()
    # simfarm.plot_response()
    # simfarm.true_pred_comp()
    # simfarm.climate_dependence()

    # Write out object as pickle
    with open(
        join(
            PARENT_DIR, 'cultivar_models',
            simfarm.cult + '_' + simfarm.metric + '_model_daily_3d.pck'),
            'wb') as pfile1:
        pickle.dump(simfarm, pfile1)

    # simfarm.post_prior_comp()  <-- see above comment

    # https://emcee.readthedocs.io/en/stable/tutorials/autocorr/ - is it steps?
    tau = simfarm.model.get_autocorr_time()
    print(f"Number of steps until the initial start is 'forgotten' {tau:.3f}")
