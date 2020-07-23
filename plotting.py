import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap

from utils.mapinfo import domains
import sharptab.thermo as thermo
from sharptab.constants import G, ZEROCNK, MS2KTS

def make_basemap(bounds):
    """Create a basemap object so we don't have to re-create this over again.

    Parameters
    ----------
    bounds : list
        Defined in the ./utils/mapinfo.py file. Format for the list convention:
        [Left, Bottom, Right, Top]

    Returns
    -------
    m : Basemap plotting object
    """
    m = Basemap(projection='stere', llcrnrlon=bounds[0], llcrnrlat=bounds[1],
                 urcrnrlon=bounds[2], urcrnrlat=bounds[3], lat_ts=50,
                 lat_0=50, lon_0=-97., resolution='i')
    m.drawcoastlines(color='#a09ea0', linewidth=2)
    m.drawstates(color='#a09ea0', linewidth=2)
    m.drawcountries(color='#a09ea0', linewidth=2)
    #.drawcounties(color="#e6e6e6", linewidth=1)
    return m

def sharppy_calcs(**kwargs):
    """Perform the SHARPpy calculations and associated heavy-lifting. Creates
    the profile object and associated thermodynamic and kinematic variables.

    Parameters
    ----------
    tmpc : dictionary
        2-D array of Temperatures (in degrees C)
    dwpc : dictionary
        2-D array of Dewpoint Temperatures (in degrees C)
    hght : dictionary
        2-D array of Geopotential Heights (in meters)
    wdir : dictionary
        2-D array of Wind Directions
    wspd : dictionary
        2-D array of Wind Speeds (in knots)
    pres : dictionary
        2-D array of Pressure (in hPa)

    Returns
    -------
    ret : dictionary
        Dictionary containing arrays of various thermodynamic and kinematic
        parameters such as MUCAPE, MLCAPE, EBWD, etc.
    """

    tmpc = kwargs.get('tmpc')
    dwpc = kwargs.get('dwpc')
    hght = kwargs.get('hght')
    wdir = kwargs.get('wdir')
    wspd = kwargs.get('wspd')
    pres = kwargs.get('pres')

    mucape = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    mlcape = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    mlcin = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    mulpl = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    ebot = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    etop = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    eshr = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    esrh = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    estp = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    ebwd_u = np.zeros((tmpc.shape[1], tmpc.shape[2]))
    ebwd_v = np.zeros((tmpc.shape[1], tmpc.shape[2]))

    for j in range(tmpc.shape[1]):
        for i in range(tmpc.shape[2]):
            prof = profile.create_profile(pres=pres[:,j,i], tmpc=tmpc[:,j,i],
                                          hght=hght[:,j,i], dwpc=dwpc[:,j,i],
                                          wspd=wspd[:,j,i], wdir=wdir[:,j,i])

            # Effective inflow and shear calculations
            eff_inflow = params.effective_inflow_layer(prof)
            ebot[j,i] = interp.to_agl(prof, interp.hght(prof, eff_inflow[0]))
            etop[j,i] = interp.to_agl(prof, interp.hght(prof, eff_inflow[1]))

            # This isn't quite right...need to find midpoint between eff Inflow
            # bottom and EL
            ebwd_u[j,i], ebwd_v[j,i] = winds.wind_shear(prof, pbot=eff_inflow[0],
                                                        ptop=500)
            eshr[j,i] = utils.mag(ebwd_u[j,i], ebwd_v[j,i])

            # Bunkers storm motion function not implemented yet.
            #srwind = params.bunkers_storm_motion(prof)
            #esrh[j,i] = winds.helicity(prof, ebot[j,i], etop[j,i], stu=srwind[0],
            #                           stv = srwind[1])[0]
            esrh[j,i] = winds.helicity(prof, ebot[j,i], etop[j,i])[0]

            # Parcel buoyancy calculations
            mupcl = params.parcelx(prof, flag=3)
            mlpcl = params.parcelx(prof, flag=4)
            mucape[j,i] = mupcl.bplus
            mulpl[j,i] = mupcl.lplhght
            mlcape[j,i] = mlpcl.bplus
            mlcin[j,i] = mlpcl.bminus
            estp[j,i] = params.stp_cin(mlpcl.bplus, esrh[j,i], eshr[j,i],
                                     mlpcl.lclhght, mlpcl.bminus)

    ebwd_u = np.where(eshr < 24., np.nan, ebwd_u)
    ebwd_v = np.where(eshr < 24., np.nan, ebwd_v)
    eshr = np.where(eshr < 24., np.nan, eshr)
    esrh = np.where(esrh < -900., -99, esrh)
    estp = np.where(estp <-900, 0., estp)

    ret = {'mucape' : mucape,
           'mlcape' : mlcape,
           'mlcin' : mlcin,
           'mulpl' : mulpl,
           'ebot' : ebot,
           'etop': etop,
           'ebwd_u': ebwd_u,
           'ebwd_v': ebwd_v,
           'eshr': eshr,
           'esrh': esrh,
           'estp': estp
    }
    return ret

def process_files(filename, domain):
    """Read in the native-level data from the specified GRIB2 file, convert
    units, and output data dictionary.

    Parameters
    ----------
    filename : string
        Full path to GRIB2 file
    domain : string
        Domain plot area

    Returns
    -------
    ret : dictionary
        Dictionary containing the merged datasets and additional variables
        necessary for sharppy and plotting routines.
    """

    bounds = domains[domain]
    ds = xr.open_dataset(filename, engine='cfgrib',
                     backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid'}})
    lons = ds.longitude.values - 360.
    lats = ds.latitude.values
    idx_lon = np.where(np.logical_and(lons>=bounds[0]-1, lons<=bounds[2]+1),
                       1, np.nan)
    idx_lat = np.where(np.logical_and(lats>=bounds[1]-1, lats<=bounds[3]+1),
                       1, np.nan)
    idx = np.where(np.logical_and(idx_lon==1, idx_lat==1))
    data = ds.sel(x=slice(idx[1].min(), idx[1].max()),
                  y=slice(idx[0].min(), idx[0].max()))
    lons = data.longitude.values
    lats = data.latitude.values

    # Data read in
    print("Pressure read in...")
    pres = data.pres.values / 100.
    print("Geopotential Height read in...")
    hght = data.gh.values
    print("U-Wind read in...")
    uwnd = data.u.values
    print("V-Wind read in...")
    vwnd = data.v.values
    print("Temperature read in...")
    tmp = data.t.values
    print("Specific Humidity read in...")
    spfh = data.q.values
    dwp = thermo.dewpoint_from_specific_humidity(spfh, tmp, pres)

    # Define the new arrays and convert units
    tmpc = tmp - ZEROCNK
    dwpc = dwp
    wdir = thermo.wind_direction(uwnd, vwnd)
    wspd = thermo.wind_speed(uwnd, vwnd) * MS2KTS

    # For our jitted sharppy routines, need to be REALLY careful with units or
    # things will break.
    tmpc = np.array(tmpc, dtype='float64')
    dwpc = np.array(dwpc, dtype='float64')
    hght = np.array(hght, dtype='float64')
    wdir = np.array(wdir, dtype='float64')
    wspd = np.array(wspd, dtype='float64')
    pres = np.array(pres, dtype='int32')
    ret = {
        'tmpc': tmpc,
        'dwpc': dwpc,
        'hght': hght,
        'wdir': wdir,
        'wspd': wspd,
        'u': uwnd,
        'v': vwnd,
        'pres': pres,
        'lons': lons,
        'lats': lats,
        'time': data.valid_time.values
    }
    return ret

filename = './data/2020-07-23/07/rap.t07z.awp252bgrbf01.grib2'
data = process_files(filename, 'MW')
prof_data = {'pres':data['pres'], 'tmpc':data['tmpc'],
             'dwpc':data['dwpc'], 'hght':data['hght'],
             'wdir':data['wdir'], 'wspd':data['wspd']}
arrs = sharppy_calcs(**prof_data)
print(arrs)
