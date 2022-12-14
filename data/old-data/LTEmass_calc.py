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
import warnings
import pandas as pd

import warnings
warnings.simplefilter('ignore')

#choices
SAVE=True
equiv_radius=False
TITLES = True
check_12_optically_thin = False
use_filling_factor = True
f=1

#units
cms=u.cm/u.s
kms = u.km/u.s
ms = u.m/u.s
Hz = 1/u.s
Kkms = u.K/u.km/u.s
grav_units = (u.m**3)/(u.kg*u.s**2)

allsum_list = ['mass from summing all']
momsum_list = ['mass from summing mom0']

mom0_array = []
#lists
if TITLES:
        ncl_list = ["ncl"]
        number_density_sum_list = ["13CO number density sum"]
        number_density_mean_list = ["13CO number density mean"]
        number_density_max_list = ["13CO number density max"]
else:
        ncl_list = []
        number_density_sum_list = []
        number_density_mean_list = []
        number_density_max_list = []


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

        #if check_12_optically_thin:
                #if ncl in [127, 137, 155, 157, 162]:
                        #tex = t12*u.K
                #else:
                        #tex = tul_12/np.log((tul_12.value/t12)+1)
        #else:
                #tex = tul_12/np.log((tul_12.value/t12)+1)
        #assume 12CO(2-1) is optically thick and use firecracker eqn 2 to find tex
#       tex = tul_12/np.log((tul_12.value/t12)+1)
        tex = tul_12/np.log((11.255762*f+t12)/(t12+0.195762*f))
        print("tex max, tex mean", np.nanmax(tex), np.nanmean(tex))

        #assume 13CO(2-1) is optically thin and use eqn 2 to calculate its optical depth
        tau_13 = -np.log(1-((t13/f)/tul_13.value)*(((1/(np.exp(tul_13/tex)-1))-(1/(np.exp(tul_13/tcmb)-1)))**(-1)))
        print("tau 13 max, mean:", np.nanmax(tau_13), np.nanmean(tau_13))

        print('-np.log(1-((t13/f)/tul_13.value)', np.nanmean(-np.log(1-((t13/f)/tul_13.value))))
        print('(((1/(np.exp(tul_13/tex)-1))-(1/(np.exp(tul_13/tcmb)-1)))**(-1)))', np.nanmean((((1/(np.exp(tul_13/tex)-1))-(1/(np.exp(tul_13/tcmb)-1)))**(-1))))
        print('((1/(np.exp(tul_13/tex)-1))', np.nanmean(((1/(np.exp(tul_13/tex)-1)))))
        print('(1/(np.exp(tul_13/tcmb)-1)))**(-1)))', np.nanmean((1/(np.exp(tul_13/tcmb)-1))**(-1)))
        #calculate column density for 13CO
        Q = (tex.value/B.value)+(1/3)
        print("q:", np.nanmax(Q))
#       vel_disp = get_props(propfile12, propfile13, ncl)[0]
        vel_disp = 5*u.km/u.s
        vel_channel = 4999.99999998*u.m/u.s # channel size of velocity 5000 m/s
#no np sqrt 2pi
        N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * (Q/(1-np.exp(-tul_13/tex))) * tau_13*vel_channel*np.sqrt(1) *freq13/c
#       N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * (Q/(1-np.exp(-tul_13/tex))) * tau_13*vel_channel*np.sqrt(2*np.pi) *freq13/c  #vel       #best so far
#       print('v/c', freq13/c)
#       N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * (Q/(1-np.exp(-tul_13/tex))) * tau_13*np.sqrt(2*np.pi) * 3844959.96893  #freq

        mom0n = np.nansum(N_13, axis=0)
        mom0n=mom0n.value
        mom0n=mom0n/(u.m**2)
        mom0n=mom0n.to(1/u.cm**2)
        mom0_array.append(mom0n.value)
        mom0n=mom0n.value

        N_13 = (N_13.value)/((u.m)**2)
        N_13 = N_13.to(1/(u.cm**2))
        #dens = N_13*(9.24394022969462*10**(-14))
        #dens = dens *70*4*10000/2000000
        dens = N_13 * mass_factor
        print('mean, max dens', np.nanmean(dens), np.nanmax(dens))
        t13area = np.where(t13!=np.nan, 1., np.nan)
        mass = t13area*dens
        print('unsummed mass (msol/pc) mean, max, sum', np.nanmean(mass), np.nanmax(mass), np.nansum(mass))
        mom0_mass = np.nansum(dens, axis=0)
        print('np.nansum(mass), np.nansum(mom0_mass) / 1E7', np.nansum(mass),   np.nansum(mom0_mass)/(10**7))
        allsum = np.nansum(mass.value)
        momsum = np.nansum(mom0_mass.value)
        allsum_list.append(allsum)
        momsum_list.append(momsum)
        return N_13





if __name__ == "__main__":
#N_13 = column_density_13(2,2.5)
#calc_mass(N_13, 20)
#print("done")
        cubefile12 = 'ab612co21.fits'
        cubefile13 = 'ab6low.fits'
        clumpfile12 = 'ab612co21.clumps.5nsig.8pkmin.fits'
        nsig=2 #typically use 5sigma in maps but maybe use sigma used for qc
        dpc = 22*10**6 #distance in pc
        pc = 3.09*10**18 #cm/pc
        mH2 = 2*(1.6735575E-24) # in g
        msol = 1.989 * 10**33 # in g

        #HEADER DATA
        clumps12 = fits.getdata(clumpfile12)
        clumps13 = 1
        #ncl_total12 = np.nanmax(clumps12)
        #ncl_total13 = np.nanmax(clumps13)

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


#       for ncl in range(1,101):
#       for ncl in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 4$
        for ncl in [1,2]:
                print(ncl)
                ncl_list.append(ncl)
                t12 = np.where(ncl==clumps12, COtemp12, np.nan)
                t13 = np.where(ncl==clumps12, COtemp12, np.nan)
                print('t13 max:', np.nanmax(t13), 't12 max:',np.nanmax(t12))
                mom013 = np.nansum(t13, axis=0)
                mom012 = np.nansum(t12,axis=0)
                if (26*np.nansum(mom013) <0.25*np.nansum(mom012)):
                        allsum_list.append(np.nan)
                        momsum_list.append(np.nan)
                else:
                        column_density_13(t12, t13, freq13, ncl,mass_factor)

        print("lists")
        print(list(zip(ncl_list, number_density_sum_list, number_density_mean_list, number_density_max_list)))

        props_lists = zip(ncl_list, allsum_list, momsum_list)
        print(list(props_lists))
        if SAVE:
                with open('LTEmass_12CO21.csv', 'w') as f:
                        writer=csv.writer(f, delimiter=',')
                        writer.writerows(zip(ncl_list,allsum_list, momsum_list))
        print("done")
