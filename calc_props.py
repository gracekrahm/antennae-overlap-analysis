import numpy as np
import matplotlib as mpl
#mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches
from astropy.io import fits
from scipy.optimize import curve_fit
import h5py
#from scipy.ndimage.filters import gaussian_filter
from astrodendro import Dendrogram
import peakutils.peak
import datetime
import multiprocessing as mp
import sys
from astropy.stats import mad_std
from astropy import stats
#import astropy.wcs
from astropy.stats import mad_std
from astropy import stats
from astropy import units as u
import pandas as pd
import csv

import warnings
warnings.simplefilter('ignore')

pc = 3.09 * 10**18 # cm
mH2 = 2. * 1.67*10**(-24) # mass of H2 in g
Msol = 2.*10**33 # mass of sun in g
G = 6.67 * 10**(-8) # in cgs

def density_calc(mlumco,errmlumco,R,errR):
    rsp = (np.pi*R**2)**2
    dddr = -3*mlumco*np.pi/rsp
    dddm = (np.pi*R**2)/rsp
    return np.sqrt((dddr*errR.value)**2 + (dddm*errmlumco)**2)

def extPressure_k(Mass, SigmaV, Radius):
    Pi = 0.5
    Radius *= 3.09 * 10**18
    SigmaV *= 10**5
    Mass *= 2 * 10**33
    p_constant = (3 * 0.5)/(4*np.pi*(1.38*10**(-16)))
    return p_constant*(Mass * SigmaV**2) / (Radius**3)

def extPressure_k_err(Mass, SigmaV, Radius, errmlumco, errsigv, errR):
    Radius *= 3.09 * 10**18
    SigmaV = SigmaV * 10**5
    Mass = Mass * 2 * 10**33
    errR = errR * 3.09 * 10**18
    errsigv = errsigv * 10**5
    p_constant = 3 * 0.5/(4*np.pi*(1.38*10**(-16)))
    errmlumco = errmlumco * 2 * 10**33
    dpdr = -3*p_constant*Mass*(SigmaV**2)/(Radius**4)
    dpdm = (p_constant*(SigmaV**2))/(Radius**3)
    dpdv = 2*p_constant*Mass*SigmaV/(Radius**3)
    error = np.sqrt((dpdr*errR)**2 + (dpdm*errmlumco)**2 + (dpdv*errsigv)**2)
    return error


def calc_alphavir(mass, meansigv, R):
  return 5. * (meansigv * 10**5)**2 * (R * pc) / (G * (mass * Msol))
def calc_alphavir_err(mass, meansigv, R, errmass, errsigv, errR):
    erralphavir = np.sqrt(((10. * (meansigv * 10**5) * (R * pc) / (G * (mass * Msol))) * errsigv * 10**5)**2 +
                          ((5. * (meansigv * 10**5)**2 / (G * (mass * Msol))) * errR * pc)**2 +
                          ((5. * (meansigv * 10**5)**2 * (R * pc) / (G * (mass * Msol)**2)) * errmass * Msol)**2)
    return erralphavir


def TB(cube, bmin, bmaj, freq):
    return 1.222 * 10**3 * (cube * 1000) / (freq**2 * bmin * bmaj)

def gaussian(dat, A0, sigx, x0):
    return A0 * np.exp(-(dat - x0)**2/(2*sigx**2))


def fitEllipse(cont):
    # From online stackoverflow thread about fitting ellipses
    x=cont[:,0]
    y=cont[:,1]

    x=x[:,None]
    y=y[:,None]

    D=np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
    S=np.dot(D.T,D)
    C=np.zeros([6,6])
    C[0,2]=C[2,0]=2
    C[1,1]=-1
    E,V=np.linalg.eig(np.dot(np.linalg.inv(S),C))
    n=np.argmax(E)
    a=V[:,n]

    #-------------------Fit ellipse-------------------
    b,c,d,f,g,a=a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
    num=b*b-a*c
    cx=(c*d-b*f)/num
    cy=(a*f-b*d)/num

    angle=0.5*np.arctan(2*b/(a-c))*180/np.pi
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    a=np.sqrt(abs(up/down1))
    b=np.sqrt(abs(up/down2))

    params=[cx,cy,a,b,angle]

    return params

def define_get_clump_props(Galaxy, stype, clumps, TCO, nsig, rms, D_Gal, arcsec_per_pixel, XCO, alphaco, PLOT=False, verbose=False):
    # d can be either a dendrogram file or a clumpfind array, stype (structure type) will tell the code which it is
    global get_clump_props
    def get_clump_props(ncl):
        print('Computing Clump ', ncl, datetime.datetime.now())
        if stype=='dendro':
            if clumps[ncl].parent == None:
                cltype = 0
            elif clumps[ncl].is_leaf:
                cltype = 2
            elif clumps[ncl].is_branch:
                cltype = 1
            else:
                cltype = 0
            mask = clumps[ncl].get_mask()
            if line=='12CO32' and mask.shape==(100,2646,2646):
                    mask = mask[:-4,:,:]
            cldat = np.where(mask, TCO, np.nan)
            regionmask = np.where(mask, ncl, np.nan)
            n_sgmc3 = np.count_nonzero(regionmask[0:2646,0:843]==ncl)
            n_sgmc35 = np.count_nonzero(regionmask[0:2646,843:1017]==ncl)
            n_sgmc2 = np.count_nonzero(regionmask[0:2646,1017:1557]==ncl)
            n_sgmc1 = np.count_nonzero(regionmask[0:2646,1557:2272]==ncl)

            if np.all(np.isnan(cldat)):
                blank = np.full(22, np.nan)
                blank[0] = ncl
                print('clump ',ncl, ' is all nans')
                print
                return blank

        else:
            n_sgmc3 = np.count_nonzero(clumps[0:2646,0:843]==ncl)
            n_sgmc35 = np.count_nonzero(clumps[0:2646,843:1017]==ncl)
            n_sgmc2 = np.count_nonzero(clumps[0:2646,1017:1557]==ncl)
            n_sgmc1 = np.count_nonzero(clumps[0:2646,1557:2272]==ncl)
            cldat = np.where(clumps==ncl, TCO, np.nan)
            cltype = 2
            if np.all(np.isnan(cldat)):
                blank = np.full(22, np.nan)
                blank[0] = ncl
                print('clump is all nans')
                print
                return blank
        if n_sgmc3>n_sgmc2 and n_sgmc3>n_sgmc35 and n_sgmc3>n_sgmc1:
            print('reg SGMC 345')
            SGMC = 345
            line_ratio = line_ratios_list[0]
        if n_sgmc35>n_sgmc3 and n_sgmc35>n_sgmc2 and n_sgmc35>n_sgmc1:
            print('reg SGMC 345')
            line_ratio = line_ratios_list[1]
            SGMC = 345
        if n_sgmc3<n_sgmc2 and n_sgmc2>n_sgmc35 and n_sgmc2>n_sgmc1:
            print('reg SGMC 2 probably')
            line_ratio = line_ratios_list[1]
            SGMC = 2
        if n_sgmc1>n_sgmc2 and n_sgmc1>n_sgmc35 and n_sgmc3<n_sgmc1:
            print('reg SGMC 1')
            line_ratio = line_ratios_list[2]
            SGMC = 1
        try:
            SGMC_def = SGMC
        except:
            SGMC_def = 0
            line_ratio = line_ratio = line_ratios_list[1]

            
        # Mask the map to include just the clump of interest (these are 3D still)

        

        clmom0 = np.nansum(cldat, axis=0)
        clmom8 = np.nanmax(cldat, axis=0)
        # Calculate moment maps for the clump

        COmax = np.nanmax(clmom8)
        print('comax', COmax)
        argmax = np.unravel_index(np.nanargmax(cldat), (cldat.shape))
        
        #convert pixel values to ra and dec
        w = astropy.wcs.WCS(hdr, naxis=2, fix=True, translate_units='h')
        coords = astropy.wcs.utils.pixel_to_skycoord(argmax[1],argmax[0],w)
        ra = coords.ra.value/15
        dec = coords.dec.value

        mask_to_ones = np.where(clmom8 > nsig*rms, 1., np.nan)
        cube_to_ones = np.where(cldat > nsig*rms, 1., np.nan)
        # 2D map of 1 where the clump is above 5sigma and nan where it's below 5 sigma
        # 3D map of 1 where the clump is above 5sigma and nan where it's below 5 sigma

        Npix = np.nansum(mask_to_ones)
        Nvox = np.nansum(cube_to_ones)
        print('Npix, Vvox: ', Npix, Nvox)
        # How many pixels are above 5==sigma

        
        lumco = np.nansum(clmom0)*u.K * deltav * asarea * cdelt2**2
        print('lumco', lumco)
        mlumco = alphaco * lumco
        print('mlumco', mlumco)
        errlumco = np.sqrt((np.sqrt(Nvox) * rms.value)**2 + (deltav.value * asarea.value *cdelt2.value**2)**2)
        errmlumco = np.sqrt((0.1*mlumco.value)**2 + (alphaco.value * errlumco)**2)
        #calculate luminosity and mass


        #clprof = np.nansum(cldat * clmom8, axis=(1,2)) / np.nansum(clmom8)
        clprof = np.nanmean(cldat, axis=(1,2))
        # Intensity-weighted mean line profile
        clprof = np.where(np.isnan(clprof), 0.0, clprof)
        Tmax = np.nanmax(clprof)
        vmax = vels[np.nanargmax(clprof)]
        print('vmax = ', vmax)

        # Make an average velocity profile for the clump (so going from 3D and collapsing over the two spatial dimensions)
        try:
            sol, cov = curve_fit(gaussian, vels, clprof, p0=[Tmax,2.5,vmax])
            errsigv = np.sqrt(np.diag(cov)[1]) # Get standard deviation from variance array
        except:
            sol = np.array([np.nan,np.nan,np.nan])
            cov = np.array([np.nan,np.nan,np.nan])
            errsigv = np.sqrt(np.diag(cov)[1,1])
        print('gaussian solution: ', sol)
        meansigv = sol[1]
        print('Sigma v and error: ', meansigv, errsigv)
        # Fit a Gaussian and get the linewidth and its error

        if PLOT:
            plt.plot(vels, clprof)
            plt.plot(vels, gaussian(vels, *sol), label='Gaussian fit')
            #plt.xlim(vels[0],vels[30])
            plt.legend()
            plt.show()

        zeromom0 = np.where(clmom0>0, clmom0, 0.0) # So it's zero instead of nan outside of clump
        mask_to_ones_full = np.where(clmom0 > 0, 1., 0.) # ones and zeros
        wholemapmom0 = np.nansum(TCO, axis=0)

        if PLOT:
            ax = plt.subplot(111)
            plt.imshow(wholemapmom0, origin='lower', cmap='Greys', vmax = np.min([np.max([5.*np.nanmax(clmom0), 8*rms]), np.nanmax(wholemapmom0)]), vmin=0)
            plt.colorbar()
            plt.contour(mask_to_ones_full,[0.5], linewidths=0.5) # Half power contour

        contlines = plt.contour(zeromom0.value,[0.5*np.nanmax(zeromom0.value)]) # Half power contour
        for i in np.arange(len(contlines.allsegs[0])):
            if i==0:
                dat0 = np.array(contlines.allsegs[0][i])
            else:
                dat0 = np.concatenate((dat0, np.array(contlines.allsegs[0][i])))


            # It splits up the array of contours weird unless I do this

        # Fit an ellipse based on stack overflow code
        try:
            params1=fitEllipse(dat0)
            xc,yc,a,b,theta = params1
        except:
            xc,yc,a,b,theta = np.nan,np.nan,np.nan,np.nan,np.nan

        t = np.arange(0,2.01, 0.01) * np.pi
        xt = xc + a*np.cos(theta)*np.cos(t) - b*np.sin(theta)*np.sin(t)
        yt = yc + a*np.sin(theta)*np.cos(t) + b*np.cos(theta)*np.sin(t)
        x = dat0[:,0]
        y = dat0[:,1]
        xarr = np.reshape(np.repeat(x, len(t)), (len(x), len(t)))
        yarr = np.reshape(np.repeat(y, len(t)), (len(y), len(t)))
        d = np.sqrt((xarr - xt)**2 + (yarr - yt)**2)
        res = np.nanmin(d, axis=1)

        if PLOT:
            ell = matplotlib.patches.Ellipse((xc,yc), 2*a, 2*b, theta, fill=False, edgecolor='cyan', linewidth=0.5)
            ax.add_artist(ell)
            plt.xlim(xc - 2*np.sqrt(Npix), xc + 2*np.sqrt(Npix))
            plt.ylim(yc - 2*np.sqrt(Npix), yc + 2*np.sqrt(Npix))
            plt.show()

        a = ((arcsec_per_pixel/206265.) * D_Gal)* a
        b = ((arcsec_per_pixel/206265.) * D_Gal)* b
        # Convert to pc

        print('fitted ellipse params (pc, pc, deg)')
        print(a, b, theta)

        Rell = np.sqrt(a * b).value
        Rellsig = Rell * (2./2.35) # pc, convert HWHM to sigma

        # Estimating an error in R based on how non-circular it is
        errR = np.nanmean(res) * ((arcsec_per_pixel/206265.) * D_Gal) * (2./2.35)

        if deconvolve:
            R = np.sqrt((Rellsig)**2 - (brad.value/2)**2)
            errR = Rellsig * errR / np.sqrt(np.abs((Rellsig)**2 - (brad.value/2)**2))
        else:
            R = Rellsig
            errR = errR

        if xradius:
            R = 1.91 * R
            errR = 1.91 * errR

        print('Final R and error: ', R, errR)



        area = np.nansum(mask_to_ones_full) * (arcsec_per_pixel/206265. * D_Gal)**2 # convert to pc2
        perim = abs(np.diff(mask_to_ones_full, axis=0)).sum() + abs(np.diff(mask_to_ones_full, axis=1)).sum()
        perim = perim * arcsec_per_pixel/206265. * D_Gal
        print('area and perimeter: ', area, perim)

        print
        print
        pressure = extPressure_k(mlumco.value, meansigv, R)
        alphavir = calc_alphavir(mlumco.value, meansigv, R)
        erralphavir = calc_alphavir_err(mlumco.value, meansigv, R, errmlumco, errsigv, errR.value)
        pressure_err = extPressure_k_err(mlumco.value, meansigv, R, errmlumco, errsigv, errR.value)
        density = mlumco.value/(np.pi*R**2)
        densityerr = densityerr = density_calc(mlumco.value,errmlumco,R,errR)
        props = np.array([ncl, cltype, argmax[2], argmax[1], argmax[0], ra,dec,SGMC, Npix, Nvox, lumco.value, errlumco, COmax.value, mlumco.value, errmlumco, meansigv, errsigv, a.value, b.value, R, errR.value, area.value, perim.value, density,densityerr,pressure, pressure_err,alphavir,erralphavir])

        return props

    return get_clump_props



if __name__=='__main__':

    SAVE = True
    PLOT = False
    TEST = True
    verbose = True
    nsig = 5 # Typically use a cut of 5 sigma in maps
    D_Gal = 22*10**6 * u.pc
    Galaxy = 1313
    deconvolve = True
    xradius = True
    stype = 'clump'
    maskfile = 'ab612co21.clumps.6sig350npix.fits'
    cubefile = 'ab612co21.fits'
    cubefilepbcor = 'ab6highpbcor.fits'
    pbfile = 'ab612co21pb.fits'
    solMass = u.def_unit('solMass')
    XCO = (0.5 * 10**20)*(u.cm**2)*u.s/(u.K*u.km) # cm^2 / (K km/s) - standard Galactic value from Bolattio+2013
    alphaco = 0.8 * u.solMass * u.s / (u.K * u.km * u.pc**2)
    as2 = 1 * u.arcsec**2
    asarea = (as2*D_Gal**2).to(u.pc**2,equivalencies=u.dimensionless_angles())
    if TEST:
        procs = 1
    else:
        procs = 4

    print('Running clump analysis for Antennae')
    print('Loading files', datetime.datetime.now())
    if stype == 'clump':
        clumps = fits.getdata(maskfile) # Standard clumpfind output fits
    else:
        clumps = Dendrogram.load_from(maskfile)
    COdat = fits.getdata(cubefilepbcor)
    hdr=fits.getheader(cubefilepbcor)
    vels = np.arange(1300, 1800, 5)
    rms = stats.mad_std(COdat[~np.isnan(COdat)])
    if stype == 'clump':
        clmax = np.nanmax(clumps)
        if line=='12CO32' and clumps.shape==(100,2646,2646):
            clumps = clumps[:-4,:,:]
        #print('maskfile shape',clumps.shape)
    else:
        clmax = len(clumps)
        
    bmaj=hdr['bmaj'] * 3600*u.arcsec # arcsec
    bmin=hdr['bmin'] * 3600*u.arcsec # arcsec
    freq=hdr['CRVAL3'] / 10**9 # GHz
    freq=freq*u.GHz
    TCO = TB(COdat, bmaj, bmin, freq).value*u.K*u.km/u.s # K km/s
    rms = TB(rms, bmaj, bmin, freq).value*u.K*u.km/u.s 
    brad = np.sqrt(bmaj*bmin)/206265. * D_Gal
    cdelt2 = hdr['cdelt2']*3600*u.arcsec
    if hdr['ctype3'][0:4]=="FREQ":
        nu0=hdr['restfrq']
        dnu=hdr['cdelt3']
        deltav=2.99792458e5 * np.absolute(dnu)/nu0 * u.km / u.s    
    else:
        deltav = abs(hdr['cdelt3'])/1000. * u.km / u.s
        
    arcsec_per_pixel = hdr['CDELT2']*3600
    print('arcsec per pix: ', arcsec_per_pixel)
    pixelarea = ((arcsec_per_pixel/206265.) * D_Gal * pc)**2 # cm^2
    print(pixelarea)

    get_clump_props = define_get_clump_props(Galaxy, stype, clumps, TCO, nsig, rms, D_Gal, arcsec_per_pixel, XCO, alphaco, PLOT=PLOT, verbose=verbose)

    print('Function defined', datetime.datetime.now())
    print

    if TEST:
        props_array = []
        for ncl in range(1,3):
            props = get_clump_props(ncl)
            props_array.append(props)
            print('Done', datetime.datetime.now())


    else:
        print('Starting mp clump props for ', clmax, ' structures.', datetime.datetime.now())
        print
        pool = mp.Pool(processes=procs)
        if stype=='clump':
            props_array = np.array(pool.map(get_clump_props, np.arange(1,clmax+1)))
        else:
            props_array = np.array(pool.map(get_clump_props, np.arange(clmax)))
        pool.close()
        pool.join()

        print
        print('Done parallel computing', datetime.datetime.now())
        print(props_array.shape)

    if SAVE:
        np.savetxt('12CO32props_6sig_350npix.txt', props_array, fmt='%1.4g', delimiter=',')
        print
        print('Done saving properties!', datetime.datetime.now())
    # Quick and dirty way to save an array, there are better ways to do this probably
