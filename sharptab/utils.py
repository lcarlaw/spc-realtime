''' Frequently used functions '''
import numpy as np
from numba import njit
from sharptab.constants import MISSING, TOL

@njit
def weighted_average(field, weight):
    '''
    Support np.average() functionality for numba.

    Numba does not supported np.average(). Without weights this is equivalent
    to np.mean(), but there are several pressure-weighted variables which rely
    on this functionality.

    Parameters
    ----------
    field : numpy array
        Array containing data to be averaged
    weight : numpy array
        An array of weights associated with the values in a. Each value in a
        contributes to the average according to its associated weight.

    Returns
    -------
    field : number
        Weighted average of the field input.

    '''
    c = 0.
    cw = 0.
    for ind in range(field.shape[0]):
        c += field[ind] * weight[ind]
        cw += weight[ind]
    field = c / cw
    return field

@njit
def MS2KTS(val):
    '''
    Convert meters per second to knots

    Parameters
    ----------
    val : float, numpy_array
        Speed (m/s)

    Returns
    -------
    Val converted to knots (float)

    '''
    return val * 1.94384449

@njit
def KTS2MS(val):
    '''
    Convert knots to meters per second

    Parameters
    ----------
    val : float, numpy_array
        Speed (kts)

    Returns
    -------
        Val converted to meters per second (float)

    '''
    return val * 0.514444

@njit
def mag(u, v, missing=MISSING):
    '''
    Compute the magnitude of a vector from its components

    Parameters
    ----------
    u : number, array_like
        U-component of the wind
    v : number, array_like
        V-component of the wind
    missing : number (optional)
        Optional missing parameter. If not given, assume default missing
        value from sharppy.sharptab.constants.MISSING

    Returns
    -------
    mag : number, array_like
        The magnitude of the vector (units are the same as input)

    '''
    return np.sqrt(u**2 + v**2)

@njit
def vec2comp(wdir, wspd, missing=MISSING):
    '''
    Convert direction and magnitude into U, V components

    Parameters
    ----------
    wdir : number, array_like
        Angle in meteorological degrees
    wspd : number, array_like
        Magnitudes of wind vector (input units == output units)
    missing : number (optional)
        Optional missing parameter. If not given, assume default missing
        value from sharppy.sharptab.constants.MISSING

    Returns
    -------
    u : number, array_like (same as input)
        U-component of the wind (units are the same as those of input speed)
    v : number, array_like (same as input)
        V-component of the wind (units are the same as those of input speed)

    '''
    #if not QC(wdir) or not QC(wspd):
    #    return ma.masked, ma.masked

    #wdir = ma.asanyarray(wdir).astype(np.float64)
    #wspd = ma.asanyarray(wspd).astype(np.float64)
    #wdir.set_fill_value(missing)
    #wspd.set_fill_value(missing)
    #assert wdir.shape == wspd.shape, 'wdir and wspd have different shapes'
    #if wdir.shape:
        #wdir[wdir == missing] = ma.masked
        #wspd[wspd == missing] = ma.masked
        #wdir[wspd.mask] = ma.masked
        #wspd[wdir.mask] = ma.masked
    u, v = _vec2comp(wdir, wspd)
    u[np.fabs(u) < TOL] = 0.
    v[np.fabs(v) < TOL] = 0.
    #else:
    #    if wdir == missing:
    #        wdir = ma.masked
    #        wspd = ma.masked
    #    elif wspd == missing:
    #        wdir = ma.masked
    #        wspd = ma.masked
    #    u, v = _vec2comp(wdir, wspd)
    #    if ma.fabs(u) < TOL:
    #        u = 0.
    #    if ma.fabs(v) < TOL:
    #        v = 0.
    return u, v

@njit
def _vec2comp(wdir, wspd):
    '''
    Underlying function that converts a vector to its components

    Parameters
    ----------
    wdir : number, masked_array
        Angle in meteorological degrees
    wspd : number, masked_array
        Magnitudes of wind vector

    Returns
    -------
    u : number, masked_array (same as input)
        U-component of the wind
    v : number, masked_array (same as input)
        V-component of the wind

    '''
    u = wspd * np.sin(np.radians(wdir)) * -1
    v = wspd * np.cos(np.radians(wdir)) * -1
    return u, v

#@njit
#def QC(val):
#    '''
#        Tests if a value is type np.nan
#
#        '''
#    if np.isnan(val): return False
#    return True
