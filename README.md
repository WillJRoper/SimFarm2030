# SimFarm2030
SimFarm2030 aims to use climate data collected by the Met Office to model wheat production metrics using a Bayesian inference approach. This approach provides extreme flexibility, allowing the model to be applied not only to wheat data but essentially any time series, climate sensitive data. This approach also allows for a huge dynamic range in what can be predicted, with predictions possible from highly localised grids to national production metrics.

## Dependencies
Most of the utilised packages can be found with an Anaconda python3 installation. The following are not included and can be pip installed (pip install [package]):
  
  netCDF4
  
  ftplib 
  
  fnmatch
  
  seaborn
  
Pystan is a little more involved, installation instruction can be found @:https://pystan.readthedocs.io/en/latest/getting_started.html

## Climate Outputs

Plot Of Average Monthly Temp | Plot Of Daily Rainfall
------------ | -------------
<img src="https://raw.githubusercontent.com/AnBowell/SimFarm2030/master/Example_Images/month_temps.gif" width="500" height="600">| <img src="https://raw.githubusercontent.com/AnBowell/SimFarm2030/master/Example_Images/day_rain.gif" width="500" height="600">

### Here is a flow chart explaining preliminary worked carried out by Andrew Bowell (https://github.com/AnBowell/SimFarm2030) the process of adapting the FACYnation codes
<p align="center">
<img src="https://raw.githubusercontent.com/AnBowell/SimFarm2030/master/Example_Images/FlowChart/f2s.png">
</p>
