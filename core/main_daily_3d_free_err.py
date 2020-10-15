from model_daily_3d_free_err import cultivarModel
import utilities
import time
import pickle
import sys
import os


# Extract cultivar from command line input
cult = sys.argv[1]
files = os.listdir("../example_data")
files.remove('.DS_Store')

for f in files:

    cult = f.split("_")[0]
    print(cult)

    tstart = time.time()
    simfarm = cultivarModel(cult, region_tol=0.25, metric='Yield',
                            metric_units='t Ha$^{-1}$')
    simfarm.train_and_validate_model(nsample=20000, nwalkers=250)
    print('Train', time.time() - tstart)

    simfarm.plot_walkers()
    simfarm.plot_response()

    # Write out object as pickle
    with open('../cultivar_models/' + simfarm.cult + '_' + simfarm.metric + '_model_daily_3d.pck', 'wb') as pfile1:
        pickle.dump(simfarm, pfile1)

    simfarm.post_prior_comp()

    tau = simfarm.model.get_autocorr_time()
    print("Steps until initial start 'forgotten'", tau)

    break
