from model_daily_3d import cultivarModel
# import utilities
import time
import pickle
import sys
import os


def get_non_hidden_filepaths():
    return [
        f for f in os.listdir('../example_data')
        if not f.startswith('.')
    ]


# Extract cultivar from command line input
cult = sys.argv[1]

if cult == "All":
    files = get_non_hidden_filepaths()
else:
    files = [cult + "_Data.csv", ]

print(files)  # to see if it knows what all is.
for f in files:
    cult = f.split("_")[0]
    print(f'{cult} is being processed')

    tstart = time.time()
    simfarm = cultivarModel(cult, region_tol=0.25, metric='Yield',
                            metric_units='t Ha$^{-1}$')
    simfarm.train_and_validate_model(nsample=75000, nwalkers=250)
    print(f'Train in {(time.time() - tstart):.2} seconds')

    # Moved plotting functions -> create_figures.py revisit after tests created
    # simfarm.plot_walkers()
    # simfarm.plot_response()
    # simfarm.true_pred_comp()
    # simfarm.climate_dependence()

    # Write out object as pickle
    with open('../cultivar_models/' + simfarm.cult + '_' + simfarm.metric
              + '_model_daily_3d.pck', 'wb') as pfile1:
        pickle.dump(simfarm, pfile1)

    # simfarm.post_prior_comp()  <-- see above comment

    tau = simfarm.model.get_autocorr_time()
    print(f"Steps until initial start 'forgotten' {tau:.3f}")
