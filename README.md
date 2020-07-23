# spc-realtime
Download user-specified NCEP model data (full-vertical resolution native coordinates) and create forecast images emulating those available on the SPC Mesoanalysis page. Makes use of **[Numba](http://numba.pydata.org/)** to accelerate **[SHARPpy](https://github.com/sharppy/SHARPpy)** parcel functions.

# Basic Setup
## Creating the Anaconda environment.
My `~/.condarc` file looks like this:

```
ssl_verify: true
channels:
- conda-forge
- defaults
```

Note that the order is important here for versioning control as newer versions of matplotlib and basemap cause issues with some of the map features and do not install correctly if, say, `matplotlib` is installed last.

```
conda create --name spc-realtime python=3.7

conda install matplotlib=2.2.4
conda install basemap=1.2.0 basemap-data-hires=1.2.0
conda install xarray netcdf4 cfgrib
conda install numba
conda install cdo
```

# General Useage [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lcarlaw/spc-realtime/master?urlpath=lab)   

Testing things out using **[Binder (more info at the link)](https://mybinder.org/)** which you can access here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lcarlaw/spc-realtime/master?urlpath=lab). Every time the Github repository is updated, Binder will re-create a Docker image containing the environment's dependencies. This process may take awhile (much like the creation of an initial Anaconda environment), but once the image is built and no changes made to the repo, the loading process should take about a minute or so.
