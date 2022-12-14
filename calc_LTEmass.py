import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from astropy.io import fits
from astropy.stats import mad_std
from astropy import stats
from astropy import units as u
from scipy.optimize import curve_fit
import h5py
from astrodendro import Dendrogram
import sys
from scipy.optimize import fsolve
import pandas as pd
import multiprocessing as mp
import warnings
warnings.simplefilter('ignore')

#choices
SAVE=True
equiv_radius=False
TITLES = True
use_filling_factor = True
f=1

#units
cms=u.cm/u.s
kms = u.km/u.s
ms = u.m/u.s
Hz = 1/u.s
Kkms = u.K/u.km/u.s
grav_units = (u.m**3)/(u.kg*u.s**2)




def column_density_13(t12, t13, freq13, ncl, mass_factor):
        print('t13 max', np.nanmax(t13))
        #constants
        tul_12 = 11.06*u.K #K for 12CO(2-1)
        tul_13 = 10.58*u.K #K for 13CO(2-1)
        gu = 5
        nu_naught = freq13
        tcmb = 2.73*u.K
        B = 2.644*u.K #K for 13CO
        Aul = 6.03*10**(-7)*Hz #http://www.ifa.hawaii.edu/users/jpw/classes/star_formation/reading/molecules.pdf
        c = 2.998*10**8*ms


        #assume 12CO(2-1) is optically thick and use firecracker eqn 2 to find tex
        #tex = tul_12/np.log((tul_12.value/t12)+1)
        tex = tul_12/np.log((11.255762*f+t12)/(t12+0.195762*f))
        print("tex max, tex mean", np.nanmax(tex), np.nanmean(tex))

        #assume 13CO(2-1) is optically thin and use eqn 2 to calculate its optical depth
        tau_13 = -np.log(1-((t13/f)/tul_13.value)*(((1/(np.exp(tul_13/tex)-1))-(1/(np.exp(tul_13/tcmb)-1)))**(-1)))
        print("tau 13 max, mean:", np.nanmax(tau_13), np.nanmean(tau_13))


        #calculate column density for 13CO
        Q = (tex.value/B.value)+(1/3)
        print("q:", np.nanmax(Q))

        vel_channel = 4999.99999998*u.m/u.s # channel size of velocity 5000 m/s
        N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * (Q/(1-np.exp(-tul_13/tex))) * tau_13*vel_channel*np.sqrt(1) *freq13/c
        N_13 = (N_13.value)/((u.m)**2)
        N_13 = N_13.to(1/(u.cm**2))
        dens = N_13 * mass_factor
        print('mean, max dens', np.nanmean(dens), np.nanmax(dens))
        t13area = np.where(t13!=np.nan, 1., np.nan)
        mass = t13area*dens
        print('unsummed mass (msol/pc) mean, max, sum', np.nanmean(mass), np.nanmax(mass), np.nansum(mass))
        mom0_mass = np.nansum(dens, axis=0)
        summed_mass = np.nansum(mass.value)
        print('mass/1E6', summed_mass/10**6)
        return summed_mass



def define(clumps12, COtemp12, COtemp13, freq13, mass_factor):
    global run_mp
    def run_mp(ncl):
        ncl_list.append(ncl)
        if stype == 'clump':
            t12 = np.where(ncl==clumps12, COtemp12, np.nan)
            t13 = np.where(ncl==clumps12, COtemp13, np.nan)
        else:
            mask = clumps12[ncl].get_mask()
            t12 = np.where(mask, COtemp12, np.nan)
            t13 = np.where(mask, COtemp13, np.nan)
        print('t13 max:', np.nanmax(t13), 't12 max:',np.nanmax(t12))
        mom013 = np.nansum(t13, axis=0)
        mom012 = np.nansum(t12,axis=0)
        if (26*np.nansum(mom013) <0.25*np.nansum(mom012)):
            final_mass = np.nan
        else:
            final_mass = column_density_13(t12, t13, freq13, ncl,mass_factor)
        props = np.array([ncl, final_mass]) 
        return props
    return run_mp

if __name__ == "__main__":
        TEST = True
        SAVE = False
        stype = 'clump'
        cubefile12 = 'ab6highpbcor.fits'
        cubefile13 = 'ab6lowpbcor.fits'
        maskfile12 = 'ab612co21.clumps.4.25nsig.1.75leaf.5.5pkmin.300npixmin_newrms.fits'
        nsig=2 #typically use 5sigma in maps but maybe use sigma used for qc
        dpc = 22*10**6 #distance in pc
        pc = 3.09*10**18 #cm/pc
        mH2 = 2*(1.6735575E-24) # in g
        msol = 1.989 * 10**33 # in g
        
        if TEST:
            procs = 1
        else:
            procs = 4

        #HEADER DATA
        if stype == 'clump':
            clumps12 = fits.getdata(maskfile12)
        else:
            clumps12 = Dendrogram.load_from(maskfile12)
        COdata12 = fits.getdata(cubefile12)
        COdata13 = fits.getdata(cubefile13)
        COdata_header12 = fits.getheader(cubefile12)
        COdata_header13 = fits.getheader(cubefile13)
        arcsec_per_pixel = COdata_header12['CDELT2']*3600
        print('arcsec per pix: ', arcsec_per_pixel)
        pixelarea = ((arcsec_per_pixel/206265.) * dpc * pc)**2 # cm^2
        print('pixelarea',pixelarea)
        mass_factor = 1.36*mH2*pixelarea*(2*10**4)*70/msol
        print('mass factor',mass_factor)

        bmaj12 = COdata_header12['bmaj']*3600 #converted to arcsec
        bmin12 = COdata_header12['bmin']*3600 #converted to arcsec
        bmaj13 = COdata_header13['bmaj']*3600 #converted to arcsec
        bmin13 = COdata_header13['bmin']*3600 #converted to arcsec

        freq12 = COdata_header12['CRVAL3']*Hz
        freq13 = COdata_header13['CRVAL3']*Hz
        print('freq12, 13:', freq12, freq13)
        rms12 = stats.mad_std(COdata12[~np.isnan(COdata12)])
        rms_K12 = 1.222 * 10**3 * (rms12 * 1000) / (freq12.to(u.GHz).value**2 * bmin12 * bmaj12)
        rms13 = stats.mad_std(COdata13[~np.isnan(COdata13)])
        rms_K13 = 1.222 * 10**3 * (rms13 * 1000) / (freq13.to(u.GHz).value**2 * bmin13 * bmaj13)

        COtemp12 = 1.222 * 10**3 * (COdata12 * 1000) / (freq12.to(u.GHz).value**2 * bmin12 * bmaj12) #convert to K
        COtemp13 = 1.222 * 10**3 * (COdata13 * 1000) / (freq13.to(u.GHz).value**2 * bmin13 * bmaj13)


        if stype == 'clump':
            clmax = np.nanmax(clumps12)  
        else:
            clmax = len(clumps12)
        run_mp = define(clumps12,COtemp12,COtemp13, freq13,mass_factor)
        if TEST:
            props_array = []
            for ncl in np.array([92]):#,200,276, 374, 452, 486, 498, 507, 578]):
                props = run_mp(ncl)
                props_array.append(props)
                print('done')
            
            
        else:
            print('Starting mp clump props for ', clmax, ' structures.')
            print
            pool = mp.Pool(processes=procs)
            if stype=='clump':
                props_array = np.array(pool.map(run_mp, np.arange(1,clmax+1)))
            else:
                props_array = np.array(pool.map(run_mp, np.arange(clmax+1)))
            pool.close()
            pool.join()

            print
            print('Done parallel computing')
            print(props_array.shape)


        if SAVE:
            np.savetxt('LTEmass_ant.txt', props_array, fmt='%1.4g', delimiter='\t')
            print 
            print('Done saving properties!')
