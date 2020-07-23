# spc-realtime
Download user-specified NCEP model data (full-vertical resolution native coordinates) and create forecast images emulating those available on the SPC Mesoanalysis page. 

## Basic Setup
### Creating the Anaconda environment. 
My `~/.condarc` file looks like this:
```
ssl_verify: true
channels:
- conda-forge
- defaults
```
Note that the order is important here for versioning control as newer versions of matplotlib and basemap cause issues with some of the map features.
```
conda create --name spc-realtime python=3.7

conda install matplotlib=2.2.4
conda install basemap=1.2.0 basemap-data-hires=1.2.0
conda install xarray netcdf4 cfgrib
conda install numba
conda install cdo
```

## Useage
