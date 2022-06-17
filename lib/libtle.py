#!/usr/bin/python3.7

import numpy as np
from astropy.time import Time
from datetime import datetime
from sgp4 import exporter
from sgp4.api import Satrec, WGS84
import lib.libastro as la


LOOP_ERR_TOL = 1e-13
MAX_LOOP_NUM = 20

def acose(x):
    """ acos limited to inputs in range [0,1]
    """

    if (x >= 1):
        rVal = 0
    elif (x <= -1):
        rVal = np.pi
    else:
        rVal = np.arccos(x)

    return rVal

def generate_tle(jd_epoch=0,meanElem=np.zeros(6),satnumber=99999,bstar=0,tle_file=None):
    e = meanElem[0]        
    n_oday = meanElem[1] # mean motion (orbits per day)
    ma = meanElem[2]   
    incl = meanElem[3]       
    argpo = meanElem[4]       
    raan = meanElem[5]       
    
    no_kozai = n_oday*(2*np.pi) / (24*60)

    satrec = Satrec()
    satrec.sgp4init(
        WGS84,
        'i',
        int(satnumber), # Satellite Number
        (jd_epoch-2433281.5), # Epoch
        bstar, # drag coefficient (1/earth radii)
        0,  # ndot not used
        0,  # nddot not used
        e,  # eccentricity
        np.deg2rad(argpo), # argument of perigee rad
        np.deg2rad(incl), # inclination rad
        np.deg2rad(ma),  # mean anomaly rad
        no_kozai, # mean motion (radians/minute)
        np.deg2rad(raan)
        )
    
    line1,line2 = exporter.export_tle(satrec)
    
    return line1.replace('\x00',' '), line2



def rvel(reci, veci,verbose_enable=False):
    """ convert osculating position and velocity vectors
    to components of two line element set (TLE)

    input:
    reci = eci position vector (kilometers)
    veci = eci velocity vector (kiometers/second)

    output:

    elem(1) = xe             = orbital eccentricity (non-dimensional)
    elem(2) = xMeanMotion    = mean motion (orbits per day)
    elem(3) = xMeanAnom      = mean anomaly (deg)
    elem(4) = xincl          = orbital inclination (deg)
    elem(5) = xArgPer        = argument of perigee (deg)
    elem(6) = xRAAN          = right ascension of ascending node (deg)

    reference: Scott Campbell's Satellite Orbit Determination
            web site http://sat.belastro.net/satelliteorbitdetermination.com/
    error correction by David Gerhardt, ported to python (DJU)
    """

    pi2 = 2.0 * np.pi
    xke = 0.0743669161331734132
    xj3 = -2.53881e-6
    req = 6378.135     #[km] earth equatorial radius
    ck2 = 5.413079e-4

    # convert position vector to earth radii
    # and velocity vector to [earth radii per minute]

    rr2 = np.zeros(3)
    vv2 = np.zeros(3)
    vk = np.zeros(3)

    for i in range(3):
        rr2[i] = reci[i] / req # [earth radii] ECI position vector
        vv2[i] = 60 * veci[i] / req # [earth radii / sec] ECI velocity vector
        vk[i] = vv2[i] / xke # velocity / sqrt(mu_E)

    # Find the Keplerian Elements
    h = np.cross(rr2, vk);  # angular momentum
    pl = np.dot(h, h) # orbit parameter

    vz = np.array([0, 0, 1])

    n = np.cross(vz, h) # orbit normal
    
    if (n[1] == 0 and n[2] == 0):
        n[0] = 1

    n = n/np.linalg.norm(n) # normalize

    rk = np.linalg.norm(rr2)
    rdotk = np.dot(rr2, vv2) / rk
    rfdotk = np.linalg.norm(h) * xke / rk
    temp = np.dot(rr2, n) / rk

    uk = acose(temp)
    if (rr2[2] < 0):
        uk = pi2 - uk

    vz = np.cross(vk, h)
    vy = -1/rk*rr2

    vec = vz + vy
    ek = np.linalg.norm(vec)
    if (ek >= 1):
        print('pos/vel vectors lead to open orbit\n')
        return

    xnodek = np.arctan2(n[1], n[0]) # RAAN, REPLACED with atan2
    if xnodek < 0:
        xnodek = xnodek + pi2
    
    temp  = np.sqrt(h[0]**2 + h[1]**2)
    xinck = np.arctan2(temp, h[2]) # inclination
    temp  = np.dot(vec, n) / ek
    wk    = acose(temp)

    if (vec[2] < 0):
        wk = np.mod(pi2 - wk, pi2); # argument of periapsis
    
    aodp = pl / (1 - ek**2) # semi-major axis
    xn = xke * aodp**(-1.5) # Keplerian Mean Motion

    """
    in the first loop the osculating elements rk, uk, xnodek, xinck, rdotk,
    and rfdotk are used as anchors to find the corresponding final sgp4
    mean elements r, u, xnodeo, xincl, rdot, and rfdot. several other final
    mean values based on these are also found: betal, cosio, sinio, theta2,
    cos2u, sin2u, x3thm1, x7thm1, x1mth2. in addition, the osculating values
    initially held by aodp, pl, and xn are replaced by intermediate
    (not osculating and not mean) values used by sgp4. the loop converges
    on the value of pl in about four iterations.
    """
    
    # seed value for first loop

    xincl = xinck
    u = uk

    for i in range(MAX_LOOP_NUM):
        
        a2 = pl
        betal = np.sqrt(pl / aodp)
        temp1 = ck2 / pl
        temp2 = temp1 / pl
        cosio = np.cos(xincl)
        sinio = np.sin(xincl)
        sin2u = np.sin(2*u)
        cos2u = np.cos(2*u)
        
        theta2 = cosio * cosio
        x3thm1 = 3 * theta2 - 1
        x1mth2 = 1 - theta2
        x7thm1 = 7 * theta2 - 1
        
        r = (rk - 0.5 * temp1 * x1mth2 * cos2u) / (1  - 1.5 * temp2 * betal * x3thm1)
        u = uk + 0.25 * temp2 * x7thm1 * sin2u
        
        xRAAN = xnodek - 1.5 * temp2 * cosio * sin2u
        xincl = xinck  - 1.5 * temp2 * cosio * sinio * cos2u
    
        rdot  = rdotk  + xn * temp1 * x1mth2 * sin2u
        rfdot = rfdotk - xn * temp1 *(x1mth2 * cos2u + 1.5 * x3thm1)
        temp = r * rfdot / xke
        pl = temp * temp
        
        # vis-visa equation
        temp = 2/r - (rdot**2 + rfdot**2)/xke**2
        aodp = 1 / temp
        
        xn = xke * aodp**(-1.5)
        
        if (np.abs(a2 - pl) < LOOP_ERR_TOL):
            if verbose_enable:
                print('Stop iteration due loop errer tolerance exceeded')
            break
        
        if (i >= MAX_LOOP_NUM-1):
            if verbose_enable:
                print('Maximum iteration exceeded')
            break

    """"
    the next values are calculated from constants and a combination of mean
    and intermediate quantities from the first loop. these values all remain
    fixed and are used in the second loop.
    """

    # preliminary values for the second loop

    ecose = 1 - r / aodp
    esine = r * rdot / (xke * np.sqrt(aodp))
    elsq = 1 - pl / aodp
    a3ovk2 = -xj3 / ck2
    xlcof = 0.125 * a3ovk2 * sinio * (3 + 5*cosio) / (1 + cosio)
    aycof = 0.25  * a3ovk2 * sinio
    temp1 = esine / (1 + np.sqrt(1 - elsq))
    cosu = np.cos(u)
    sinu = np.sin(u)

    """
    the second loop normally converges in about six iterations to the final
    mean value for the eccentricity, eo. the mean perigee, omegao, is also
    determined. cosepw and sinepw are found to twelve decimal places and
    are used to calculate an intermediate value for the eccentric anomaly,
    temp2. temp2 is then used in kepler's equation to find an intermediate
    value for the true longitude, capu.
    """
    
    # seed values for loop

    xe = np.sqrt(elsq)
    xArgPer = wk
    axn = xe * np.cos(xArgPer)

    for i in range(MAX_LOOP_NUM):
        a2 = xe
        beta = 1 - xe**2
        temp = 1 / (aodp * beta)
        aynl = temp * aycof
        ayn = xe * np.sin(xArgPer) + aynl
        
        cosepw = r * cosu / aodp + axn - ayn * temp1
        sinepw = r * sinu / aodp + ayn + axn * temp1
        
        axn = cosepw * ecose + sinepw * esine
        ayn = sinepw * ecose - cosepw * esine
        
        xArgPer = np.mod(np.arctan2(ayn - aynl, axn), pi2)
        
        if (xe > 0.5):
            xe = 0.9 * xe + 0.1 * (axn / np.cos(xArgPer))
        else:
            xe = axn / np.cos(xArgPer)
        
        if (xe > 0.999):
            xe = 0.999

        if (np.abs(a2 - xe) < LOOP_ERR_TOL):
            # print('Loop 2: %1.0f\n',i);
            break
        
        if (i >= MAX_LOOP_NUM-1):
            if verbose_enable:
                print('Maximum iteration exceeded')
            break


    temp2 = np.arctan2(sinepw, cosepw)
    capu = temp2 - esine
    xll = temp * xlcof * axn

    # xll adjusts the intermediate true longitude
    # capu, to the mean true longitude, xl
    xl = capu - xll

    xMeanAnom = np.mod(xl - xArgPer, pi2)

    # the third loop usually converges after three iterations to the
    # mean semi-major axis, a1, which is then used to find the mean motion, xno

    a0 = aodp
    a1 = a0
    beta2 = np.sqrt(beta)
    temp = 1.5 * ck2 * x3thm1 / (beta * beta2)
    for i in range(MAX_LOOP_NUM):
        a2 = a1
        d0 = temp/a0**2
        a0 = aodp * (1 - d0)
        d1 = temp/a1**2
        a1 = a0/(1 - d1/3 - d1**2 - 134*d1**3/81)
        if (np.abs(a2 - a1) < LOOP_ERR_TOL):
            #print('Loop 3: %1.0f\n',i);
            break

        if (i >= MAX_LOOP_NUM-1):
            if verbose_enable:
                print('Maximum iteration exceeded')
            break

    xMeanMotion = xke * a1**(-1.5) # [rad/min]

    # save to output vector
    meanElem = np.zeros(6)
    meanElem[0] = xe;                     # orbital eccentricity (non-dimensional)
    meanElem[1] = xMeanMotion*1440/pi2    # mean motion (orbits per day)
    meanElem[2] = xMeanAnom*180/np.pi     # mean anomaly (deg)
    meanElem[3] = xincl*180/np.pi         # orbital inclination (deg)
    meanElem[4] = xArgPer*180/np.pi       # argument of perigee (deg)
    meanElem[5] = xRAAN*180/np.pi         # right ascension of ascending node (deg)

    if np.isnan(xe):
        print('wait')
    return meanElem

def meanElemCorr(mE1,mE2,mE3):
    """performs mE1 + mE1 - mE3
    it gets complicated because angles are involved
    
    INPUTS
    meanElem[0] = xe;         # orbital eccentricity (non-dimensional)
    meanElem[1] = xMeanMotion # mean motion (orbits per day)
    meanElem[2] = xMeanAnom   # mean anomaly (deg)
    meanElem[3] = xincl       # orbital inclination (deg)
    meanElem[4] = xArgPer     # argument of perigee (deg)
    meanElem[5] = xRAAN       # right ascension of ascending node (deg)
    """
    
    out = np.zeros(6)

    # eccentricity
    out[0] = mE1[0] + mE2[0] - mE3[0]

    # mean motion
    out[1] = mE1[1] + mE2[1] - mE3[1]

    # mean anomaly: 0 - 360
    out[2] = np.mod(mE1[2] + mE2[2] - mE3[2],360) # [deg]

    # inclination: 0 - 180 (can't roll over)
    out[3] = np.abs(mE1[3] + mE2[3] - mE3[3])

    # argPer: 0 - 360
    out[4] = np.mod(mE1[4] + mE2[4] - mE3[4],360)

    # RAAN: 0 - 360
    out[5] = np.mod(mE1[5] + mE2[5] - mE3[5],360)
    
    return out

def MagDiff(vec1,vec2):
    # function calculates the magnitude of the difference between two vectors
    vecDiff = vec1 - vec2
    outMagDiff = np.sqrt(vecDiff[0]**2+vecDiff[1]**2+vecDiff[2]**2)

    return outMagDiff


def RV2TLE(rECI,vECI,t,corrNum=2,satnumber=99999,bstar=0,verbose_enable=False):
    """ function solves for the TLE that matches the provided pos/vel vector. One
    TLE is produced for each rECI/vECI/epochDN state vector.
    
    INPUTS
    rECI - [3xN] (m) position vector in ECI frame
    vECI - [3xN] (m/s) velocity vector in ECI frame
    t    - [1xN] unix time epoch vector
    corrNum - [1x1] number of SGP4-based corrections. Defaults to 2
    """

    t_datetime = [datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in t]
    t_vec      = Time(list(t_datetime), scale='utc')

    xbStarDrag = 0*0.18519e-3
    
    # Setup
    errEstR = np.zeros([corrNum+1,len(t_vec)])
    errEstV = np.zeros([corrNum+1,len(t_vec)])
    meanElem = np.zeros([6,corrNum+1,len(t_vec)])

    line1_list = [[None]*(corrNum+1)]*len(t_vec)
    line2_list = [[None]*(corrNum+1)]*len(t_vec)

    for i in range(len(t_vec)):
        
        if verbose_enable:
            print(i)
        # Initial Mean Element Fit
        # solve for k-elements

        meanElem[:,0,i] = rvel(rECI[:,i]/1000,vECI[:,i]/1000,verbose_enable=verbose_enable)  # initial fit mean elements
        
        # create TLE using initial fit mean elements
        line1_list[i][0], line2_list[i][0] = generate_tle(t_vec[i].jd,meanElem[:,0,i],satnumber=satnumber,bstar=bstar)
        
        # determine pos/vel using initial fit mean elements
        data = la.sgp4_tle('', t[i], ITRF=False,line1=line1_list[i][0],line2=line2_list[i][0])
        
        rECI_fit = data['sat_pos'] * 1000
        vECI_fit = data['sat_vel'] * 1000

        # calculate & store error
        errEstR[0,i] = MagDiff(rECI[:,i],rECI_fit) # [m]
        errEstV[0,i] = MagDiff(vECI[:,i],vECI_fit) # [m/s]
        
        for j in range(corrNum):

            # Corrections using SPG4
            # the first estimate is often good enough. but much of the time,
            # correction by comparison to SGP4 output will help

            # plug the latest fit mean elements into rvel
            meanElem_corr = rvel(rECI_fit/1000,vECI_fit/1000,verbose_enable=verbose_enable)
            
            # form the new mean elements by correcting previous estimate
            #    meanElem_new = meanElem_old + meanElem_init - meanElem_corr;
            meanElem[:,j+1,i] = meanElemCorr(meanElem[:,j,i],meanElem[:,0,i],meanElem_corr)
            
            # replace fitTLE with estimate from the new corrected mean elements 
            line1_list[i][j+1], line2_list[i][j+1] = generate_tle(t_vec[i].jd,meanElem[:,j+1,i],satnumber=satnumber,bstar=bstar)

            # determine pos/vel using initial fit mean elements
            data = la.sgp4_tle('', t[i], ITRF=False,line1=line1_list[i][j+1],line2=line2_list[i][j+1])
            
            rECI_fit = data['sat_pos'] * 1000
            vECI_fit = data['sat_vel'] * 1000

            # calculate & store error
            errEstR[j+1,i] = MagDiff(rECI[:,i],rECI_fit) # [m]
            errEstV[j+1,i] = MagDiff(vECI[:,i],vECI_fit) # [m/s]

    return line1_list, line2_list, meanElem, errEstR, errEstV