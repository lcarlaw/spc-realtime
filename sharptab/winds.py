''' Wind Manipulation Routines '''
import numpy as np
from numba import njit
from sharptab import interp, utils

@njit
def mean_wind(prof, pbot=850, ptop=250, dp=-1, stu=0, stv=0):
    '''
    Calculates a pressure-weighted mean wind through a layer. The default
    layer is 850 to 200 hPa.

    Parameters
    ----------
    prof: profile object
        Profile object
    pbot : number (optional; default 850 hPa)
        Pressure of the bottom level (hPa)
    ptop : number (optional; default 250 hPa)
        Pressure of the top level (hPa)
    dp : negative integer (optional; default -1)
        The pressure increment for the interpolated sounding
    stu : number (optional; default 0)
        U-component of storm-motion vector (kts)
    stv : number (optional; default 0)
        V-component of storm-motion vector (kts)

    Returns
    -------
    mnu : number
        U-component (kts)
    mnv : number
        V-component (kts)

    '''
    if dp > 0: dp = -dp
    #if not utils.QC(pbot) or not utils.QC(ptop):
    #    return ma.masked, ma.masked
    #if prof.wdir.count() == 0:
    #    return ma.masked, ma.masked

    ps = np.arange(pbot, ptop+dp, dp)
    u, v = interp.components(prof, ps)
    # u -= stu; v -= stv

    mnu = utils.weighted_average(u, ps) - stu
    mnv = utils.weighted_average(v, ps) - stv

    #return ma.average(u, weights=ps)-stu, ma.average(v, weights=ps)-stv
    return mnu, mnv


@njit
def wind_shear(prof, pbot=850, ptop=250):
    '''
    Calculates the shear between the wind at (pbot) and (ptop).

    Parameters
    ----------
    prof: profile object
        Profile object
    pbot : number (optional; default 850 hPa)
        Pressure of the bottom level (hPa)
    ptop : number (optional; default 250 hPa)
        Pressure of the top level (hPa)

    Returns
    -------
    shu : number
        U-component (kts)
    shv : number
        V-component (kts)

    '''
    #if np.count_nonzero(~np.isnan(prof.wdir)) == 0 or not utils.QC(ptop) or not utils.QC(pbot):
    #    return np.nan, np.nan
    ubot, vbot = interp.components(prof, pbot)
    utop, vtop = interp.components(prof, ptop)
    shu = utop - ubot
    shv = vtop - vbot
    return shu, shv

@njit
def helicity(prof, lower, upper, stu=0, stv=0, dp=-1, exact=True):
    '''
    Calculates the relative helicity (m2/s2) of a layer from lower to upper.
    If storm-motion vector is supplied, storm-relative helicity, both
    positve and negative, is returned.

    Parameters
    ----------
    prof : profile object
        Profile Object
    lower : number
        Bottom level of layer (m, AGL)
    upper : number
        Top level of layer (m, AGL)
    stu : number (optional; default = 0)
        U-component of storm-motion (kts)
    stv : number (optional; default = 0)
        V-component of storm-motion (kts)
    dp : negative integer (optional; default -1)
        The pressure increment for the interpolated sounding (mb)
    exact : bool (optional; default = True)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)

    Returns
    -------
    phel+nhel : number
        Combined Helicity (m2/s2)
    phel : number
        Positive Helicity (m2/s2)
    nhel : number
        Negative Helicity (m2/s2)

    '''
    #if np.count_nonzero(~np.isnan(prof.wdir)) == 0 or not utils.QC(lower) or not utils.QC(upper) or not utils.QC(stu) or not utils.QC(stv):
    #    return np.nan, np.nan, np.nan

    if lower != upper:
        lower = interp.to_msl(prof, lower)
        upper = interp.to_msl(prof, upper)
        plower = interp.pres(prof, lower)
        pupper = interp.pres(prof, upper)
        if np.isnan(plower) or np.isnan(pupper):
            return -999., -999., -999.

        if exact:
            ind1 = np.where(plower >= prof.pres)[0].min()
            ind2 = np.where(pupper <= prof.pres)[0].max()
            u1, v1 = interp.components(prof, plower)
            u2, v2 = interp.components(prof, pupper)
            u_temp = prof.u[ind1:ind2+1]
            u_temp = u_temp[~np.isnan(u_temp)]
            v_temp = prof.v[ind1:ind2+1]
            v_temp = v_temp[~np.isnan(v_temp)]
            u = ()
            u = np.append(u, u1)
            u = np.append(u, u_temp)
            u = np.append(u, u2)
            v = ()
            v = np.append(v, v1)
            v = np.append(v, v_temp)
            v = np.append(v, v2)
        else:
            ps = np.arange(plower, pupper+dp, dp)
            u, v = interp.components(prof, ps)
        sru = utils.KTS2MS(u - stu)
        srv = utils.KTS2MS(v - stv)
        layers = (sru[1:] * srv[:-1]) - (sru[:-1] * srv[1:])
        phel = layers[layers > 0].sum()
        nhel = layers[layers < 0].sum()

    else:
        phel = nhel = 0
    return phel+nhel, phel, nhel

"""
@njit
def non_parcel_bunkers_motion(prof):
    '''
    Compute the Bunkers Storm Motion for a Right Moving Supercell

    Parameters
    ----------
    prof : profile object
        Profile Object

    Returns
    -------
    rstu : number
        Right Storm Motion U-component (kts)
    rstv : number
        Right Storm Motion V-component (kts)
    lstu : number
        Left Storm Motion U-component (kts)
    lstv : number
        Left Storm Motion V-component (kts)

    '''
    #if prof.wdir.count() == 0:
    #    return ma.masked, ma.masked, ma.masked, ma.masked

    d = utils.MS2KTS(7.5)     # Deviation value emperically derived as 7.5 m/s
    msl6km = interp.to_msl(prof, 6000.)
    p6km = interp.pres(prof, msl6km)
    # SFC-6km Mean Wind
    mnu6, mnv6 = mean_wind_npw(prof, prof.pres[prof.sfc], p6km)
    # SFC-6km Shear Vector
    shru, shrv = wind_shear(prof, prof.pres[prof.sfc], p6km)

    # Bunkers Right Motion
    tmp = d / utils.mag(shru, shrv)
    rstu = mnu6 + (tmp * shrv)
    rstv = mnv6 - (tmp * shru)
    lstu = mnu6 - (tmp * shrv)
    lstv = mnv6 + (tmp * shru)

    return rstu, rstv, lstu, lstv
"""
