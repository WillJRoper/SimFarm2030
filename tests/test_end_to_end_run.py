from core.model_daily_3d import cultivarModel
import numpy as np


def test_training():
    cult = 'Test'
    simfarm = cultivarModel(
        cult, region_tol=0.25, metric='Yield',
        metric_units='t Ha$^{-1}$')
    simfarm.train_and_validate_model(nsample=1100, nwalkers=25)
    assert np.mean(simfarm.resi) != np.nan
    assert np.median(simfarm.resi) != np.nan
    assert 0.2 <= simfarm.af <= 0.5
    assert all(v != np.nan for v in simfarm.mean_params.values())
    # assert simfarm.resi == [12.1234, 23.434, 56.3343]
