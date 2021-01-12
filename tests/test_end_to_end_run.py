import pytest

from core.model_daily_3d import cultivarModel

import numpy as np
import time


def test_training():
    cult = 'Test'
    tstart = time.time()
    simfarm = cultivarModel(
        cult, region_tol=0.25, metric='Yield',
        metric_units='t Ha$^{-1}$')
    simfarm.train_and_validate_model(nsample=1200, nwalkers=125)
    assert np.mean(simfarm.resi) == 49
    assert np.median(simfarm.resi) == 63
    assert simfarm.resi == [12.1234, 23.434, 56.3343]

    # print(f'{(time.time()-tstart):.3} seconds to run training for {cult}_Data.csv')

# How to run test

    # * need to make imput folder configurable so that we dont
    # have to change dir to test to run tests.

    # cd tests
    # pytest .

    # in order to make this work:
    # import the cultivarModel from core.model_daily_3d
    # change the import utilities to core.utilities in model_daily_3d
    # add the __init__.py to core directory
