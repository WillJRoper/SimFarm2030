# SimFarm2030
SimFarm2030 aims to use climate data collected by the Met Office to model wheat production metrics using a Bayesian inference approach. This approach provides extreme flexibility, allowing the model to be applied not only to wheat data but essentially any time series, climate sensitive data. This approach also allows for a huge dynamic range in what can be predicted, with predictions possible from highly localised grids to national production metrics.

## Dependencies
Most of the utilised packages can be found with an Anaconda python3 installation. The following are not included and can be pip installed (`pip install [package]`):
  
  netCDF4
  
  ftplib 
  
  fnmatch
  
  seaborn

  pandas

  numba

  matplotlib

  h5py
  
  emcee
  
Pystan (required to run the old pystan version of the code) is a little more involved, installation instruction can be found @:https://pystan.readthedocs.io/en/latest/getting_started.html

HDF5 files can be downloaded from (make sure to save them in the folder above simfarm):

https://drive.google.com/file/d/1nrf1RVgU4n-13NprK3oDwg5RA-cL_5Md/view?usp=sharing

https://drive.google.com/file/d/10GBNGBfMGCEAaAXK3sY68W4eCYcUS0mP/view?usp=sharing

https://drive.google.com/file/d/14swnAaInBR1eZcOl2V3lMdHzLVuF5LdU/view?usp=sharing

## Running the model

To run the latest version of the model, clone this repo and from that directory:

``` 
cd simfarm
python main_daily_3d.py All     # to train and validate on all datasets within example_data or...
python main_daily_3d.py Claire  # to train and validate on the Claire cultivar, this can be any cultivar contained in example_data
```
You will need to add a folders in the below structure (if they don't already exist in order to run the model):
Climate_Data
model_performance
  Chains
  Validation
  Predictionvstruth
  Corners
cultivar_models


## Running the tests

cd tests
pytest .

or (from anywhere)

PYTHONPATH=simfarm pytest tests
PYTHONPATH=simfarm pytest --ignore=tests/archive


## Plotting figures

* First - run the main_daily_3d.py
* Then locate the .pck file in simfarm/culivar_models to copy path to terminal
* Hash out any plots that you don't want to create in create_figures.py

PYTHONPATH=simfarm python create_figures.py /home/anisa/Code/SimFarm2030/cultivar_models/Skyfall_Yield_model_daily_3d.pck
