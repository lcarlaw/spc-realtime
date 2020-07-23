from datetime import datetime, timedelta
import os, sys
import subprocess
import xarray as xr
from distutils.spawn import find_executable
if find_executable('cdo') is None:
    raise ValueError("cdo executable not found")
    sys.exit(1)

realtime_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/rap."
with open('./fields.txt') as f: vars = f.readlines()[0][:-1]
DATA_PATH = "%s/%s" % (os.environ['PWD'], '/data')
os.makedirs(DATA_PATH, exist_ok=True)

def download_data_realtime(run_time, **kwargs):
    """Download realtime RAP13 data from the NCEP NOMADS server. This function
    uses Wesley Ebisuzaki's excellent get_inv.pl and get_grib.pl scripts to
    allow downloading of a small subset of the full grib2 data files. Saves a
    ton of space and time! Further information can be found at:

    https://www.cpc.ncep.noaa.gov/products/wesley/fast_downloading_grib.html

    Parameters
    ----------
    run_time : string
        Time of the model run to be downloaded. YYYY-MM-DD/HH
    data_path : string (optional; default = ${PWD}/data)
        Local path in which to store the downloaded datafiles. If none specified
        will attempt to create a subdirectory in ${PWD}/data

    Returns
    -------
    Individual .grib2 files downloaded to the local system.

    """
    time = datetime.strptime(run_time, '%Y-%m-%d/%H')
    if time.hour in [3, 9, 15, 21]:
        fhrs = 39
    else:
        fhrs = 21
    date_str = str(time.year) + str(time.month).zfill(2) + str(time.day).zfill(2)
    hour_str = str(time.hour).zfill(2)

    data_path = kwargs.get('data_path', DATA_PATH)
    data_path = "%s/%s" % (data_path, run_time)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for fhr in range(1, fhrs+1):
        fhr = str(fhr).zfill(2)
        #fname = "rap.t%sz.awp130bgrbf%s.grib2" % (hour_str, fhr)
        fname = "rap.t%sz.awp252bgrbf%s.grib2" % (hour_str, fhr)
        url = "%s%s/%s" % (realtime_url, date_str[0:8], fname)

        full_name = data_path + '/' + fname

        # Shell argument string to invoke the get_grib.pl script.
        arg = './get_inv.pl ' + url + '.idx | egrep ' + "'" + vars +  \
               "'" + ' | ./get_grib.pl ' + url + ' ' + full_name
        if not os.path.exists(full_name):
            subprocess.call(arg, shell=True)
        else:
            print("Data already exists locally.")

    print("Merging original .grib2 files...")
    arg = "cdo mergetime %s/*.grib2 %s/RAP.grib2" % (data_path, data_path)
    subprocess.call(arg, shell=True)

    arg = "rm %s/rap.t*" % (data_path)
    subprocess.call(arg, shell=True)
    return data_path


time_str = "2020-07-23/07"
data_dir = download_data_realtime(time_str)
