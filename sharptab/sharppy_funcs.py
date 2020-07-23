
from sharptab.constants import *
import sharptab.thermo as thermo
import sharptab.profile as profile
import sharptab.params as params
import sharptab.interp as interp
import sharptab.winds as winds
import sharptab.utils as utils
'''
import sharppy
import sharppy.sharptab.profile as profile
import sharppy.sharptab.interp as interp
import sharppy.sharptab.winds as winds
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo
'''

import numpy as np
import numpy.ma as ma

def calcs(**kwargs):
    tmpc = kwargs.get('tmpc')
    dwpc = kwargs.get('dwpc')
    hght = kwargs.get('hght')
    wdir = kwargs.get('wdir')
    wspd = kwargs.get('wspd')
    pres = kwargs.get('pres')

    # For our jitted sharppy routines, need to be REALLY careful with units or
    # things will break. Probably overkill here, but worth it to be safe!
    #tmpc = np.array(tmpc, dtype='float64')
    #dwpc = np.array(dwpc, dtype='float64')
    #hght = np.array(hght, dtype='float64')
    #wdir = np.array(wdir, dtype='float64')
    #wspd = np.array(wspd, dtype='float64')
    #pres = np.array(pres, dtype='int32')

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
            #prof = profile.create_profile(pres=pres, tmpc=tmpc[:,j,i],
            #                              hght=hght[:,j,i], dwpc=dwpc[:,j,i],
            #                              wspd=wspd[:,j,i], wdir=wdir[:,j,i])

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
