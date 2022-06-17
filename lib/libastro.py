#!/usr/bin/python3.7

from lib.pyIGRF import igrf_utils as iut
import os
import numpy as np
from datetime import datetime
from sgp4.api import Satrec, WGS84
from astropy.time import Time
from astropy import coordinates as coord, units as u
from astropy.coordinates import solar_system_ephemeris, get_sun, get_moon
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy import interpolate


# Load in the file of coefficients
IGRF_FILE = r'./lib/pyIGRF/IGRF13.shc'
igrf = iut.load_shcfile(IGRF_FILE, None)

AU_KM = 149597871  # km


def qMult(q1, q2):
    """ Quaternion multiplication """

    q = np.zeros(4).transpose()
    q[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1]
    q[1] = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0]
    q[2] = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3]
    q[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]
    return q


def euler2a(euler):
    """ Euler angles (ZYX) to direction cosine matrix """

    phi = euler[0]
    theta = euler[1]
    psi = euler[2]
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # body to inertial
    a = np.array([[cpsi*cth, -spsi*cphi+cpsi*sth*sphi,  spsi*sphi+cpsi*cphi*sth],
                 [spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi],
                 [-sth,      cth*sphi,                  cth*cphi]])

    a = a.transpose()

    return a


def a2euler(R):
    """ Direction cosine matrix to Euler angles (ZYX) """
    R = R.transpose()
    if (abs(R[2, 0]) != 1):
        theta = -np.arcsin(R[2, 0])
        phi = np.arctan2(R[2, 1]/np.cos(theta), R[2, 2]/np.cos(theta))
        psi = np.arctan2(R[1, 0]/np.cos(theta), R[0, 0]/np.cos(theta))
    else:
        psi = 0
        lambda_a = np.arctan2(R[0, 1], R[0, 2])
        if (R[2, 0] == -1):
            theta = np.pi/2
            phi = psi + lambda_a
        else:
            theta = -np.pi/2
            phi = -psi + lambda_a

    euler = np.array([phi, theta, psi])
    return euler


def q2a(q):
    """ Quaternion to direction cosine matrix """
    A = np.array([[q[0]**2-q[1]**2-q[2]**2+q[3]**2,
                   2*(q[0]*q[1]+q[2]*q[3]),
                   2*(q[0]*q[2]-q[1]*q[3])],
                  [2*(q[0]*q[1]-q[2]*q[3]),
                   -q[0]**2+q[1]**2 - q[2]**2+q[3]**2,
                   2*(q[1]*q[2]+q[0]*q[3])],
                  [2*(q[0]*q[2]+q[1]*q[3]),
                   2*(q[1]*q[2]-q[0]*q[3]),
                   -q[0]**2-q[1]**2+q[2]**2+q[3]**2]])

    return A


def a2q(a):
    """ Transforms DCM into quaternion """

    q = np.zeros(4).transpose()

    tr = np.trace(a)
    if (tr > 0):
        s = 0.5 / np.sqrt((tr + 1.0))
        q[3] = 0.25 / s
        q[0] = (a[2, 1] - a[1, 2]) * s
        q[1] = (a[0, 2] - a[2, 0]) * s
        q[2] = (a[1, 0] - a[0, 1]) * s
    else:
        if (a[0, 0] > a[1, 1]) and (a[0, 0] > a[2, 2]):
            s = 2 * np.sqrt(1 + a[0, 0] - a[1, 1] - a[2, 2])

            q[3] = (a[2, 1] - a[1, 2]) / s
            q[0] = s/4.0
            q[1] = (a[0, 1] + a[1, 0]) / s
            q[2] = (a[0, 2] + a[2, 0]) / s
        elif (a[1, 1] > a[2, 2]):
            s = 2 * np.sqrt(1 + a[1, 1] - a[0, 0] - a[2, 2])

            q[3] = (a[0, 2] - a[2, 0]) / s
            q[0] = (a[0, 1] + a[1, 0]) / s
            q[1] = s/4.0
            q[2] = (a[2, 1] + a[1, 2]) / s
        else:
            s = 2 * np.sqrt(1 + a[2, 2] - a[0, 0] - a[1, 1])
            q[3] = (a[1, 0] - a[0, 1]) / s
            q[0] = (a[0, 2] + a[2, 0]) / s
            q[1] = (a[1, 2] + a[2, 1]) / s
            q[2] = s/4.0

    return q


def q2euler(q):
    """ Quaternion to Euler angles (ZYX) """

    c11 = 2*(q[1]*q[2] + q[0]*q[3])
    c12 = 1 - 2*(q[0]*q[0] + q[1]*q[1])
    c31 = 2*(q[0]*q[1] + q[2]*q[3])
    c32 = 1 - 2*(q[1]*q[1] + q[2]*q[2])

    phi = np.arctan2(c11, c12)
    theta = np.arcsin(-2*(q[0]*q[2] - q[1]*q[3]))
    psi = np.arctan2(c31, c32)

    euler = np.array([phi, theta, psi])
    return euler.transpose()


def euler2q(euler):
    """ Euler angles (ZYX) to direction cosine matrix """
    if len(euler) == 3:
        c = np.cos(0.5*euler)
        s = np.sin(0.5*euler)
        q = [s[0]*c[1]*c[2] - c[0]*s[1]*s[2],
             c[0]*s[1]*c[2] + s[0]*c[1]*s[2],
             c[0]*c[1]*s[2] - s[0]*s[1]*c[2],
             c[0]*c[1]*c[2] + s[0]*s[1]*s[2]]
    else:
        c = np.cos(0.5*euler)
        s = np.sin(0.5*euler)
        q = [s[0, :]*c[1, :]*c[2, :] - c[0, :]*s[1, :]*s[2, :],
             c[0, :]*s[1, :]*c[2, :] + s[0, :]*c[1, :]*s[2, :],
             c[0, :]*c[1, :]*s[2, :] - s[0, :]*s[1, :]*c[2, :],
             c[0, :]*c[1, :]*c[2, :] + s[0, :]*s[1, :]*s[2, :]]

    return np.transpose(q)


def qVectRot(v, q):
    """ Vector rotation using quaternion """
    temp = np.array([v[0], v[1], v[2], 0])
    rotated_vect = qmulm(qconj(q), qmulm(temp.transpose(), q))
    v_rot = rotated_vect[:3]

    return v_rot


def qconj(q):
    """ Quaternion conjugate """
    q_c = np.array([-q[0], -q[1], -q[2], q[3]])

    return q_c.transpose()


def qmulm(q1, q2):
    """ Quaternion multiplication2 """
    A = np.array([[q2[3], q2[2], -q2[1], q2[0]],
                 [-q2[2], q2[3], q2[0], q2[1]],
                  [q2[1], -q2[0], q2[3], q2[2]],
                  [-q2[0], -q2[1], -q2[2], q2[3]]])
    q = np.matmul(A, q1)

    return q


def jday(date_utc):
    """ Calculate Julian date using UTC time

    Keyword arguments:
    date_utc -- ISO UTC format [string]

    NOTE: python 3.5 - datetime.datetimefromisoformat is not supported.
    """

    t_utc = datetime.fromisoformat(date_utc)

    # Day part
    jd = 367.0 * t_utc.year - \
        np.floor((7 * (t_utc.year + np.floor((t_utc.month + 9) / 12.0))) * 0.25) + \
        np.floor(275 * t_utc.month / 9.0) + t_utc.day + 1721013.5
    # use - 678987.0 to go to mjd directly

    # Fractional part
    jdfrac = (t_utc.second + t_utc.minute *
              60.0 + t_utc.hour * 3600.0) / 86400.0

    # check jdfrac
    if (jdfrac > 1.0):
        jd = jd + np.floor(jdfrac)
        jdfrac = jdfrac - np.floor(jdfrac)

    return jd, jdfrac


def qinv(q):
    """ Quaternion inverse (same as conjugate) """
    q_inv = np.zeros(4)
    q_inv[0] = -q[0]
    q_inv[1] = -q[1]
    q_inv[2] = -q[2]
    q_inv[3] = q[3]
    return q_inv


def eci2ecef(jdate):
    """ Quaternion rotation from ECI to ECEF (ADCS Software)

    Keyword arguments:
    jdate -- Julian date [float/double]

    NOTE: This attributes to levels of 1-2 deg error as it does
    not account nutation and polar motion!
    """

    # CONSTANTS FOR ECEF TRANFORM
    daysec = 86400.0
    # Earth precession [rad/sec]
    omega = 7.292115855377075e-05
    # Sidereal epoch (raf's)
    sidereal = (6.23018356e+04 / daysec + 2450448.5)
    # Time for a revolution
    T_r = (2.0 * np.pi / omega)

    N_r = np.floor((jdate) * daysec / T_r + 0.5)
    T_n = (jdate - sidereal) * daysec - N_r * T_r
    psi = np.mod(omega * T_n, 2.0 * np.pi)

    z = np.sin(psi / 2.0)
    w = np.cos(psi / 2.0)

    q = np.zeros(4)
    q[2] = np.sin(psi/2.0)
    q[3] = np.cos(psi/2.0)

    return np.transpose(q)


def gsp_frame(gsp_posvec, sat_posvec, sun_posvec):
    """ Construct Ground Sampling Points (GSP)
    +Z axis towards GSP
    +Y approx. anti-sun direction (payload)
    +X (Right hand)

    Keyword arguments:
    gsp_posvec -- GSP position vector
    sat_posvec -- Satellite position vector 
    sun_posvec -- Sun position vector
    """

    # GSP to Sat Vector
    gsp_sat_vec = gsp_posvec - sat_posvec

    # Sun - SC vector
    sat_sun_vec = -(sun_posvec - sat_posvec)  # antisun direction

    # Normalize vectors
    z_vec = gsp_sat_vec/np.linalg.norm(gsp_sat_vec)  # Z-axis
    temp_vec = sat_sun_vec/np.linalg.norm(sat_sun_vec)  # Temporary Y-axis

    # X-axis
    x_vec = np.cross(temp_vec, z_vec)
    x_vec = x_vec/np.linalg.norm(x_vec)

    # Updated Y-axis
    y_vec = np.cross(z_vec, x_vec)
    y_vec = y_vec/np.linalg.norm(y_vec)

    R = np.zeros([3, 3])

    R[0, 0] = x_vec[0]
    R[1, 0] = x_vec[1]
    R[2, 0] = x_vec[2]
    R[0, 1] = y_vec[0]
    R[1, 1] = y_vec[1]
    R[2, 1] = y_vec[2]
    R[0, 2] = z_vec[0]
    R[1, 2] = z_vec[1]
    R[2, 2] = z_vec[2]

    return R


def ref_frame(a, b):
    """ Construct reference frame
    +Z axis towards a
    +X axis approx. towards b
    +Y (Right hand)

    Keyword arguments:
    a -- Z-axis vector
    b -- Approx. X-axis vector         
    """

    # Normalize vectors
    z_vec = a/np.linalg.norm(a)  # Z-axis
    temp_yvec = b/np.linalg.norm(b)  # Temporary X-axis

    # Y-axis
    y_vec = np.cross(z_vec, temp_yvec)
    y_vec = y_vec/np.linalg.norm(y_vec)

    # Updated X-axis
    x_vec = np.cross(y_vec, z_vec)
    x_vec = x_vec/np.linalg.norm(x_vec)

    R = np.zeros([3, 3])

    R[0, 0] = x_vec[0]
    R[1, 0] = x_vec[1]
    R[2, 0] = x_vec[2]
    R[0, 1] = y_vec[0]
    R[1, 1] = y_vec[1]
    R[2, 1] = y_vec[2]
    R[0, 2] = z_vec[0]
    R[1, 2] = z_vec[1]
    R[2, 2] = z_vec[2]

    return R


def lm_frame(landmark_posvec, sat_posvec, second_vector, use_sunvector=True):
    """ Construct Landmark frame
    +Z axis towards LM
    +X axis sun-SC or SC velocity
    +Y (Right hand)

    Keyword arguments:
    landmark_posvec -- GSP position vector
    sat_posvec      -- Satellite position vector
    second_vector   -- Second vector (either SC velocity or Sun) 
    use_sunvector   -- Use second_vector as Sun-SC vector direction (TRUE)
                       or SC velocity direction (FALSE)                
    """

    # LM to Sat Vector
    lm_sat_vec = landmark_posvec - sat_posvec

    if use_sunvector:
        # Sun - SC vector
        temp_vec = second_vector - sat_posvec
    else:
        # SC velocity vector
        temp_vec = second_vector

    return ref_frame(lm_sat_vec, temp_vec)


def sun_frame(sun_posvec, sat_posvec, temp_xvec):
    """ Construct SUNPOINTING frame
    +Z SUN-SC direction
    +X temp_xvec direction
    +Y (Right hand)

    Keyword arguments:
    sun_posvec -- Sun position vector
    sat_posvec -- Satellite position vector 
    temp_xvec  -- user defined x vector 
    """

    # Sun to SC position vector
    sun_sat_vec = sun_posvec - sat_posvec

    return ref_frame(sun_sat_vec, temp_xvec)


def moon_frame(moon_posvec, sat_posvec):
    """ Construct Moon frame 
    +Z axis towards moon
    +X approx. nadir direction
    +Y (Right hand)

    Keyword arguments:
    moon_posvec -- Moon position vector
    sat_posvec  -- Satellite position vector 
    """

    # Moon to SC position vector
    moon_sat_vec = moon_posvec - sat_posvec

    return ref_frame(moon_sat_vec, -sat_posvec)


def lvlh_frame(sat_posvec, sat_velvec):
    """ Construct Local Vertical Local Horizontal (LVLH) frame
    # +Z axis towards nadir
    # +X approx. SC velocity direction
    # +Y (Right hand)

    Keyword arguments:
    sat_posvec -- Satellite position vector 
    sat_velvec -- Satellite velocity vector
    """

    return ref_frame(-sat_posvec, sat_velvec)


def get_cos_angle(a, b, c, q):
    """ Determine cosine angle in radians between vector a-b and c:

    theta = acos(a-b,c)

    Keyword arguments:
    a -- vector
    b -- vector
    c -- vector 
    q -- quaternion
    """

    # vector normalization
    v = a - b
    v = v/np.linalg.norm(v)

    # Rotate boresight vector to Inertial
    s = qVectRot(c.transpose(), q)
    s = s/np.linalg.norm(s)

    # Check cosine angle
    theta = np.arccos(np.dot(v, s))

    return theta


def moon_eclipsed(sat_posvec, moon_posvec):
    """ Check that moon is not eclipsed by Earth
    Assumed that Earth is spherical with radius 6371 km
    Added 1 degree margin 

    Keyword arguments:
    sat_posvec  -- Satellite position vector 
    moon_posvec -- Moon position vector
    """
    #

    # Moon-SAT position (ECI)
    target_sat_pos = moon_posvec - sat_posvec
    target_sat_pos = target_sat_pos/np.linalg.norm(target_sat_pos)

    # Normalization
    r_pos_n = np.linalg.norm(sat_posvec)
    sat_posvec = sat_posvec/r_pos_n

    # Earth angle
    Rearth = 6371  # km
    theta_E = np.arcsin(Rearth/r_pos_n)

    # Moon-centre Earth angle
    target_angle = np.arccos(np.dot(target_sat_pos, -sat_posvec))

    # return true when Moon is not eclipsed by Earth + 1 deg margin
    return (target_angle < (theta_E + np.deg2rad(1)))


def compute_e_bool(sunECI, sat_pECI):
    """ Compute whether satellite is in eclipse
    Assumed that Earth is spherical with radius 6371 km

    Keyword arguments:
    sunECI   -- Sun ECI position vector 
    sat_pECI -- Sat ECI position vector
    """

    # Earths equatorial radius
    Rearth = 6378137/1000.0  # km

    # Calculate sun shadow unit vector
    sunECI_norm = np.sqrt(
        np.power(sunECI[0, :], 2) + np.power(sunECI[1, :], 2) + np.power(sunECI[2, :], 2))
    sunECI_unit = sunECI/sunECI_norm

    # Calculate SC unit vector in ECI
    sat_pECI_norm = np.sqrt(
        np.power(sat_pECI[0, :], 2) + np.power(sat_pECI[1, :], 2) + np.power(sat_pECI[2, :], 2))
    sat_pECI_unit = sat_pECI/sat_pECI_norm

    # Are we on the night side of the earth? i.e. sunv*reci>0?
    coserr_sun_sat = np.sum(sunECI_unit*sat_pECI_unit, axis=0)

    # In eclipse?
    r_e = sat_pECI_norm * np.sqrt(1-np.power(coserr_sun_sat, 2))

    inde1 = (coserr_sun_sat < 0)
    inde2 = (r_e < Rearth)

    e_bool = np.logical_and(inde1, inde2)

    return e_bool.astype(int)


def DavenportQ(weights, meas, refs):
    """
    Implementation of the Davenport-q algorithm.
    See F. Markley, and D. Mortari "Quaternion Attitude Estimation using
    Vector Observations" Journal of the Astronomical Sciences v.48 pp 359-380
    2000
    @param weights: The measurement weights
    @param meas: The observed vectors (3x1 L{np.ndarray})
    @param refs: reference vector
    @return: The least squares optimal quaternion rotation
    """

    weights.shape = (len(weights), 1)
    meas.shape = (len(meas), 3)
    refs.shape = (len(refs), 3)

    B = np.zeros([3, 3])

    for a, b, r in zip(weights, meas, refs):
        b.shape = (3, 1)
        r.shape = (3, 1)
        B = B + a*np.dot(b, r.T)

    S = B + B.T

    z = np.zeros([1, 3])
    for a, b, r in zip(weights, meas, refs):
        z = z + a*np.cross(b, r)

    sigma = np.trace(B)
    K = np.zeros((4, 4))
    K[0:3, 0:3] = S-sigma*np.identity(3)
    K[0:3, 3] = z
    K[3, 0:3] = z
    K[3, 3] = sigma

    eigenValues, eigenVectors = np.linalg.eig(K)
    q = np.asarray(eigenVectors[:, np.argmax(eigenValues)])

    return q


def lla2ecef(lat, lon, alt):
    # ADCS library conversion (instead of using navpy)

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    # WGS84 ellipsoid constants:
    a = 6378137.0
    e = 8.1819190842622e-2

    # intermediate calculation
    # (prime vertical radius of curvature)
    R_oblat = a / np.sqrt(1.0 - e**2 * np.sin(lat)**2)

    r_ecef = np.zeros[2]
    r_ecef[0] = (R_oblat + alt) * np.cos(lat)*np.cos(lon)
    r_ecef[1] = (R_oblat + alt) * np.cos(lat)*np.sin(lon)
    r_ecef[2] = ((1.0 - e**2.0) * R_oblat + alt) * np.sin(lat)

    return r_ecef


def compute_elevation(sat_pos, gs_pos):
    """ Compute Elevation angle [rad] w.r.t groundstation (GS)
    Horizontal plane = 0 deg

    Keyword arguments:
    sat_pos -- Satellite position vector 
    gs_pos  -- GS position vector
    """

    # Sat to GS vector + normalization
    sat_gs_vec = sat_pos - gs_pos
    sat_gs_vec = sat_gs_vec/np.linalg.norm(sat_gs_vec)
    gs_pos = gs_pos/np.linalg.norm(gs_pos)

    # Compute angle
    # subtract 90 deg (0 deg perpendicular)
    el_angle = np.pi/2 - np.arccos(np.dot(gs_pos, sat_gs_vec))

    return el_angle


# SGP4 propagation
def sgp4_tle(tle_file, t_unix, ITRF=False,line1=None,line2=None):
    """ ASTROPY SGP4 propagator to compute satellite's position
    and velocity in GCRS frame (=ECI)

    Keyword arguments:
    tle_file -- File containing TLE in format:
                name
                line1
                line2
    t_unix   -- Time vector in UNIX
    ITRF     -- Convert final reference frame to ITRF
    line1    -- Optional input for manual TLE line1
    line2    -- Optional input for manual TLE line2
    """

    # Load TLEs
    if (line1 is not None and line2 is not None):
        sat_name = 'SAT'
        tleline1 = line1
        tleline2 = line2
    else:
        with open(tle_file) as f:
            lines = f.read().splitlines()
            sat_name = lines[0]
            tleline1 = lines[1]
            tleline2 = lines[2]
    
    # Convert to datetime and astropy TIME
    if type(t_unix) is np.ndarray:
        t_datetime = [datetime.utcfromtimestamp(
            x).strftime('%Y-%m-%d %H:%M:%S') for x in t_unix]
        t_vec = Time(list(t_datetime), scale='utc')

        # Julian dates
        jd = np.array(t_vec.jd1)
        jd_frac = np.array(t_vec.jd2)

        # Get Satellite's position/velocity vector
        satellite = Satrec.twoline2rv(tleline1, tleline2, WGS84)
        e, r, v = satellite.sgp4_array(jd, jd_frac)

        sat_teme = coord.TEME(r[:, 0]*u.km, r[:, 1]*u.km, r[:, 2]*u.km,
                          v[:, 0]*u.km/u.s, v[:, 1]*u.km/u.s, v[:, 2]*u.km/u.s,
                          obstime=t_vec)
    else:
        t_datetime = datetime.utcfromtimestamp(t_unix).strftime('%Y-%m-%d %H:%M:%S')
        t_vec = Time(t_datetime, scale='utc')

        # Julian dates
        jd = t_vec.jd1
        jd_frac = t_vec.jd2

        # Get Satellite's position/velocity vector
        satellite = Satrec.twoline2rv(tleline1, tleline2, WGS84)
        e, r, v = satellite.sgp4(jd, jd_frac)

        sat_teme = coord.TEME(r[0]*u.km, r[1]*u.km, r[2]*u.km,
                          v[0]*u.km/u.s, v[1]*u.km/u.s, v[2]*u.km/u.s,
                          obstime=t_vec)
    

    # Convert TEME to GCRS
    sat_coord = sat_teme.transform_to(coord.GCRS(obstime=t_vec))
    alt = 0
    lat = 0
    lon = 0

    if ITRF:
        # Convert to ITRS
        sat_coord = sat_coord.transform_to(coord.ITRS(obstime=t_vec))

        # GCRS to ECEF
        alt = sat_coord.sphericalcoslat.distance.value
        lat = sat_coord.sphericalcoslat.lat.value
        lon = sat_coord.sphericalcoslat.lon.value
        # wrap longitude around -180 to +180 deg
        lon = (lon + 180.0) % (360.0) - 180.0

    # to Cartesian
    sat_pos, sat_vel = sat_coord.cartesian.xyz.value, \
        sat_coord.velocity.d_xyz.value

    # TODO: this can be cleaner with a class-struct
    return_val = {'jd': jd,
                  'jd_frac': jd_frac,
                  'sat_pos': sat_pos,
                  'sat_vel': sat_vel,
                  'alt': alt,
                  'lat': lat,
                  'lon': lon}

    return return_val


def convert_gps2eci(astro_time,gps_pos, gps_vel):

    vel_coord = coord.CartesianDifferential(gps_vel[:,0],gps_vel[:,1],gps_vel[:,2],unit=u.m/u.s)
    pos_coord = coord.CartesianRepresentation(gps_pos[:,0],gps_pos[:,1],gps_pos[:,2],unit=u.m, differentials=vel_coord)
    
    itrs = coord.ITRS(pos_coord, obstime=astro_time)
    gcrs = itrs.transform_to(coord.GCRS(obstime=astro_time))

    gcrs_pos = gcrs.cartesian.xyz.value
    gcrs_vel = gcrs.cartesian.differentials['s'].d_xyz.value*1000 # [m/s]
    
    return gcrs_pos, gcrs_vel


def get_moon_sun_pos(t_vec):
    """ Computes Sun and Moon positions using JPL splice (de440s kernel)

    Keyword arguments:
    data 
    - t_vec         -- Unix time vector
    """

    # Convert to datetime and astropy TIME
    t_datetime = [datetime.utcfromtimestamp(
        x).strftime('%Y-%m-%d %H:%M:%S') for x in t_vec]
    t = Time(list(t_datetime), scale='utc')

    # Get JPL Ephemerides (GCRS)
    solar_system_ephemeris.set(
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp')

    # Get Moon and Sun in GCRS frame
    r_sun_skycoord = get_sun(t)
    r_moon_skycoord = get_moon(t)

    # Convert to numpy arrays [km]
    sun_pos_gcrs = r_sun_skycoord.cartesian.xyz.value*AU_KM
    moon_pos_gcrs = r_moon_skycoord.cartesian.xyz.value

    return sun_pos_gcrs, moon_pos_gcrs


def get_astro_body_angles(data, sc_param, ref_frame, **ref_kwargs):
    """ Computes angles from defined vector rotated by q against centre of
    - Sun
    - Moon
    - Earth

    Reference frame model is based on ref_frame function with input arguments
    and the corresponding values defined by ref_kwargs. 

    Keyword arguments:
    data 
    - t_vec         -- Unix time vector
    - sat_pos_gcrs  -- Satellite position vector in GCRS frame
    - sun_pos_gcrs  -- Sun position vector in GCRS frame
    - moon_pos_gcrs -- Moon position vector in GCRS frame

    sc_param 
    - euler_offset -- euler_offset parameteter (Euler angles in ZYX)
    - q_st_sc      -- SC to ST body rotation quaternion

    TODO: Add payload angles?
    """

    # Unpack data dictionary
    t_vec = data['t_vec']
    sat_pos_gcrs = data['sat_pos_gcrs']
    sun_pos_gcrs = data['sun_pos_gcrs']
    moon_pos_gcrs = data['moon_pos_gcrs']

    # Unpack SC parameter dictionary
    euler_offset = sc_param['euler_offset']
    q_st_sc = sc_param['q_st_sc']
    st_boresight = sc_param['st_boresight']

    # Initialize arrays
    moon_angle = np.zeros(len(t_vec))
    earth_angle = np.zeros(len(t_vec))
    sun_angle = np.zeros(len(t_vec))

    # Loop da loop
    for i in range(len(t_vec)):

        # Pass in custom reference frame with arguments
        R = ref_frame(**{name: value[:, i]
                      for name, value in ref_kwargs.items()})

        # Apply Euler offset
        q_IR = a2q(R)
        q_euler = euler2q(euler_offset)
        q_IS = qMult(q_IR, q_euler)

        # Startracker boresight
        q_IST = qMult(q_IS, qinv(q_st_sc))

        # Angles to startracker boresight
        moon_angle[i] = get_cos_angle(
            moon_pos_gcrs[:, i], sat_pos_gcrs[:, i], st_boresight, qinv(q_IST))
        earth_angle[i] = get_cos_angle(
            np.zeros(3), sat_pos_gcrs[:, i], st_boresight, qinv(q_IST))
        sun_angle[i] = get_cos_angle(
            sun_pos_gcrs[:, i], sat_pos_gcrs[:, i], st_boresight, qinv(q_IST))

    return moon_angle, earth_angle, sun_angle


def plot_astro_body_angles(path_outputfig, t_vec, **angles):

    # Unpack
    moon_angle = angles['moon_angle']
    earth_angle = angles['earth_angle']
    sun_angle = angles['sun_angle']
    e_bool = angles['e_bool']

    # Convert to datetime
    t_title = datetime.utcfromtimestamp(t_vec[0]).strftime('%Y-%m-%d')
    t_file = datetime.utcfromtimestamp(t_vec[0]).strftime('%Y%m%d_%H%M%S')
    t_datetime = [datetime.utcfromtimestamp(
        x).strftime('%Y-%m-%d %H:%M:%S') for x in t_vec]
    t_dates = dates.datestr2num(t_datetime)  # For plot x-axis

    # Delete previous figure file w/ same name
    if os.path.exists(path_outputfig):
        os.remove(path_outputfig)

    # Plot Figure
    nr_rows = 2
    fig, ax = plt.subplots(nrows=nr_rows, ncols=1, figsize=(19, 16))
    fig.subplots_adjust(hspace=0.4)
    plt.subplots_adjust(bottom=0.2)

    plot_nr = 0
    ax[plot_nr].set_title('Eclipse Boolean')
    ax[plot_nr].grid()
    ax[plot_nr].plot_date(t_dates, e_bool, linestyle='solid')
    ax[plot_nr].xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

    plot_nr = 1
    ax[plot_nr].set_title('Angles to Startracker Boresight')
    ax[plot_nr].grid()
    ax[plot_nr].plot_date(t_dates, np.rad2deg(moon_angle), linestyle='solid')
    ax[plot_nr].plot_date(t_dates, np.rad2deg(earth_angle), linestyle='solid')
    # Note: Nadir to horizon is approx 67.3 deg at 530 km altitude assuming spherical Earth.
    ax[plot_nr].plot_date(t_dates, np.rad2deg(sun_angle), linestyle='solid')
    ax[plot_nr].set_ylabel('Angle [deg]')
    ax[plot_nr].set_ylim(0, 180)
    ax[plot_nr].minorticks_on()
    ax[plot_nr].legend(['Moon', 'Earth', 'Sun'])
    ax[plot_nr].xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

    fig.savefig(path_outputfig)


def compute_igrf(t_date, lat, lon, alt):
    # lat: Latitude [deg]
    # lon: Longitude [deg]
    # alt: Altitude [km]

    # Convert time to decimal year format
    date = t_date.decimalyear

    # Interpolate the geomagnetic coefficients to the desired date(s)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    f = interpolate.interp1d(igrf.time, igrf.coeffs, fill_value='extrapolate')
    coeffs = f(date)

    # Convert Latitude into Colatitude
    colat = 90 - lat

    # Compute the main field B_r, B_theta and B_phi value for the location(s)
    Br, Bt, Bp = iut.synth_values(coeffs.T, alt, colat, lon,
                                  igrf.parameters['nmax'])

    # Result is given as north,east,nadir value - should be transformed to ECEF
    # First rotate around y-axis.

    latitude = np.deg2rad(lat)  # in radians
    longitude = np.deg2rad(270-lon)  # in radians

    # Rearrange to X, Y, Z components
    X = -Bt
    Y = Bp
    Z = -Br

    # Convert to ECEF (rot Y axis )
    xt = np.cos(latitude)*X - np.sin(latitude)*Z
    yt = Y
    zt = np.sin(latitude)*X + np.cos(latitude)*Z

    # rot around X axis
    igrf_x = xt
    igrf_y = np.cos(longitude)*yt + np.sin(longitude)*zt
    igrf_z = -np.sin(longitude)*yt + np.cos(longitude)*zt

    mag = np.zeros([3, len(t_date)])
    mag[0, :] = igrf_y
    mag[1, :] = igrf_z
    mag[2, :] = igrf_x

    return mag


# t = np.array([1.6e9, 1.6e9+1])
# t_datetime = [datetime.utcfromtimestamp(
#     x).strftime('%Y-%m-%d %H:%M:%S') for x in t]
# t_vec = Time(list(t_datetime), scale='utc')

# satpos = np.array([[3e6, 3e6], [0.9e6, 1e6], [3.1e6, 3.5e6]])
# satvel = np.array([[1e3, 1.1e3], [-2e3, -1.9e3], [6e3, 6.2e3]])

# data = {'t_astropy': t_vec, 'sat_pos_gcrs': satpos}
# sc_param = {'q_st_sc': np.array([0, 0, 0, 1]), 'euler_offset': np.array(
#     [0, 0, np.pi]), 'st_boresight': np.array([1, 0, 0])}

# moon_angle, earth_angle, sun_angle = get_astro_body_angles(
#     data, sc_param, lvlh_frame, sat_posvec=satpos, sat_velvec=satvel)
