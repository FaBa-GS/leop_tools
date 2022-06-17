#!/usr/bin/python3.7

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy.optimize import least_squares, minimize
import lib.libastro as la

def q_constraint(x):
    # Quaternion Constraint Function

    q = x[4:]
    return (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 - 1)
    
def mag_cal(B, Bref, mag_offset, mag_scale, mag_rotate):
    """ CALIBRATE MAGNETOMETER
    Use magnetometer data B to calibrate against a reference model Bref. 
    Calibration only done by looking at the norm.

    Keyword arguments:
    B          -- Measured magnetometer data
    Bref       -- Norm of the reference magnetic field data
    mag_offset -- Pre-calibration magnetometer offset parameter
    mag_scale  -- Pre-calibration magnetometer scaling parameter
    mag_rotate -- Pre-calibration magnetometer rotation matrix parameter 
    mag_type   -- Type Magnetometer, see below
    """

    # Remove calibration
    [xRaw, yRaw, zRaw] = uncalibrate_mag2(B[:, 0], B[:, 1],
                                          B[:, 2], mag_offset, mag_scale, mag_rotate)

    # Magnetic field norms igrf
    normRaw = np.sqrt(xRaw**2 + yRaw**2 + zRaw**2)
    ref_norm = np.sqrt(Bref[:, 0]**2 + Bref[:, 1]**2 + Bref[:, 2]**2)

    S = np.array([xRaw, yRaw, zRaw, normRaw]).transpose()
    R = ref_norm

    # Initial guess, and lower and upper bounds
    # Cf0 = np.array([np.mean(xRaw), np.mean(yRaw), np.mean(zRaw), 1, 1, 1, 0, 0, 0])
    # LB  = np.array([-np.Inf, -np.Inf, -np.Inf, 0.5,  0.5,  0.5, -np.pi, -np.pi, -np.pi])
    # UB  = np.array([ np.Inf,  np.Inf,  np.Inf, 2.0,  2.0,  2.0,  np.pi,  np.pi,  np.pi])
    Cf0 = [np.mean(xRaw), np.mean(yRaw), np.mean(zRaw), 1, 1, 1, 0, 0, 0]
    LB = [-np.Inf, -np.Inf, -np.Inf, 0.5,  0.5,  0.5, -np.pi, -np.pi, -np.pi]
    UB = [np.Inf,  np.Inf,  np.Inf, 2.0,  2.0,  2.0,  np.pi,  np.pi,  np.pi]

    # ,bounds=(LB,UB)
    Cf = least_squares(fun=mag_model, args=(S, R, False), x0=Cf0, bounds=(
        LB, UB), ftol=1e-15, xtol=1e-15, max_nfev=12000)

    _, xcf, ycf, zcf, af = mag_model(
        Cf.x, S, R, return_flag=True)

    return Cf, xcf, ycf, zcf, af


def mag_model(X, S, r, return_flag=False, rot_matrix=None):
    """ Model of the magnetometer for calibration
    Model composes hard-iron effects as offset parameter, and
    soft-iron effects as scale and rotation parameter.

    Keyword arguments:
    X           -- Offset, scale and rotation correction
    S           -- Magnetometer data
    r           -- Reference Magnetic Field Norm
    return_flag -- Boolean to return the residual, corrected B(X,Y,Z),  
                   and rotation matrix a, instead of the error norm(S)-r
    rot_matrix  -- Apply rotation matrix instead of Euler angles
    """

    x0 = X[0]
    y0 = X[1]
    z0 = X[2]
    sx = X[3]
    sy = X[4]
    sz = X[5]

    phi = X[6]
    theta = X[7]
    psi = X[8]

    x = S[:, 0]
    y = S[:, 1]
    z = S[:, 2]

    xOff = x + x0
    yOff = y + y0
    zOff = z + z0

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    if rot_matrix is None:
        a = np.array([
            [cpsi*cth,  -spsi*cphi+cpsi*sth*sphi,  spsi*sphi+cpsi*cphi*sth],
            [spsi*cth,  cpsi*cphi+sphi*sth*spsi,   -cpsi*sphi+sth*spsi*cphi],
            [-sth,      cth*sphi,                  cth*cphi]])
    else:
        a = rot_matrix

    A = np.matmul(a, np.array([xOff, yOff, zOff]))
    A = np.matmul(np.linalg.inv(a), np.array(
        [sx*A[0, :], sy*A[1, :], sz*A[2, :]]))

    xCorr = A[0, :].transpose()
    yCorr = A[1, :].transpose()
    zCorr = A[2, :].transpose()

    err = r - np.sqrt(xCorr**2 + yCorr**2 + zCorr**2)
    res = np.sum(err)

    if return_flag:
        return res, xCorr, yCorr, zCorr, a
    else:
        return err


def calibrate_mag(B, offset, scale, rot_m):

    B1 = B+offset
    B2 = np.matmul(rot_m, B1)
    B3 = np.zeros(3)
    for i in range(3):
        B3[i] = B2[i]*scale[i]
    B_cal = np.matmul(np.transpose(rot_m), B3)

    return B_cal


def uncalibrate_mag(B, offset, scale, rot_m):

    B1 = np.matmul(rot_m, B)
    B2 = B1/scale
    B3 = np.matmul(np.transpose(rot_m), B2)
    B_raw = B3-offset

    return B_raw


def uncalibrate_mag2(Bx, By, Bz, offset, scale, rotMatVec):
    """ Remove calibration parameter of the magnetometer data
    Does the reverse of the model of the magnetometer to 
    determine raw data values.

    Keyword arguments:
    Bx        -- Measured magnetometer X data
    By        -- Measured magnetometer Y data
    Bz        -- Measured magnetometer Z data
    offset    -- magnetometer offset parameter
    scale     -- magnetometer scaling parameter
    rotMatVec -- magnetometer rotation matrix parameter 
    """

    # Form the rotation matrix
    a = np.array([[rotMatVec[0], rotMatVec[1], rotMatVec[2]],
                  [rotMatVec[3], rotMatVec[4], rotMatVec[5]],
                  [rotMatVec[6], rotMatVec[7], rotMatVec[8]]])

    # Initialize vectors
    xRaw = np.zeros(len(Bx))
    yRaw = np.zeros(len(By))
    zRaw = np.zeros(len(Bz))

    # Rotate into soft-magnet frame
    for i in range(len(Bx)):
        # Remove offsets
        magNoOffset = np.array(
            [Bx[i] - offset[0], By[i] - offset[1], Bz[i] - offset[2]])

        # Rotate to soft-iron frame
        B_soft = np.matmul(a, magNoOffset.transpose())

        # Remove scaling
        magInSoftFrameNoScale = np.array(
            [B_soft[0]/scale[0], B_soft[1]/scale[1], B_soft[2]/scale[2]])

        # Rotate back to regular frame
        magNoScale = np.matmul(
            (np.linalg.inv(a)), magInSoftFrameNoScale.transpose())

        xRaw[i] = magNoScale[0]
        yRaw[i] = magNoScale[1]
        zRaw[i] = magNoScale[2]

    return xRaw, yRaw, zRaw


def plot_sphere(X, Y, Z, save_plot=False, savepath='./output/'):
    """ Plot unit sphere of magnetometer data """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Scale by norm
    N = np.sqrt(X**2 + Y**2 + Z**2)

    ax.scatter(X/N, Y/N, Z/N)

    ax.set_xlabel('X [-]')
    ax.set_ylabel('Y [-]')
    ax.set_zlabel('Z [-]')

    plt.title('Normalized Unit Sphere')
    plt.grid()

    if save_plot:
        fig.savefig(savepath)
    else:
        plt.show()


def plot_err_hist(err1, err2, save_plot=False, savepath='./output/',unit='mG'):
    """ Plot error histogram data """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(19, 6))
    ax[0].set_title(
        'Pre-Calibration - \u03BC: {:.2f} {:s} (\u03C3: {:.2f} {:s})'.format(np.average(err1),unit, np.std(err1),unit))
    ax[0].grid()
    ax[0].hist(err1)
    ax[0].set_xlabel('Err [{:s}]'.format(unit))
    ax[0].set_ylabel('Frequency')
    ax[1].set_title(
        'Post-Calibration - \u03BC: {:.2f} {:s} (\u03C3: {:.2f} {:s})'.format(np.average(err2),unit, np.std(err2),unit))
    ax[1].grid()
    ax[1].hist(err2)
    ax[1].set_xlabel('Err [{:s}]'.format(unit))
    ax[1].set_ylabel('Frequency')

    if save_plot:
        fig.savefig(savepath)
    else:
        plt.show()


def plot_norm(norm1, norm2, norm3, t_date=None, save_plot=False, savepath='./output'):
    """ Plot norm data """

    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 6))
    
    if t_date is not None:
        
        t_title = datetime.utcfromtimestamp(t_date[0]).strftime('%Y-%m-%d')
        plt.title('Magnetometer Norm Comparison - {:s}'.format(t_title))

        t_datetime = [datetime.utcfromtimestamp(
            x).strftime('%Y-%m-%d %H:%M:%S') for x in t_date]
        t_plot = dates.datestr2num(t_datetime)

        ax.plot_date(t_plot,norm1, linestyle='solid')
        ax.plot_date(t_plot,norm2, linestyle='solid')
        ax.plot_date(t_plot,norm3, linestyle='solid')
        ax.set_ylabel('|B| [mG]')
        ax.legend(['NO CAL', 'CAL', 'IGRF'])
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
        ax.minorticks_on()
        plt.grid()
    else:
        plt.title('Magnetometer Norm Comparison')
        ax.plot(norm1, linestyle='solid')
        ax.plot(norm2, linestyle='solid')
        ax.plot(norm3, linestyle='solid')
        ax.set_ylabel('|B| [mG]')
        ax.legend(['NO CAL', 'CAL', 'IGRF'])
        plt.grid()

    if save_plot:
        fig.savefig(savepath)
    else:
        plt.show()


def lsq_reg(x, y):
    """ LSQ linear fit function """

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    N = len(x)

    # SLOPE
    a = (N*np.sum(x*y) - sum_x*sum_y)/(N*np.sum(x**2)-sum_x**2)

    # OFFSET
    b = (sum_y - a*sum_x)/N

    return a, b


def sunvector(X, fss_raw):
    """ Calculate sun vector based on RAW FSS values
    """
    # Unpack calibration parameters
    h = X[0]  # Pinhole height
    x0 = X[1]  # x0 pos correction
    y0 = X[2]  # y0 pos correction
    theta = X[3]  # Rotational correction

    # FSS raw values
    A = fss_raw[0, :]
    B = fss_raw[1, :]
    C = fss_raw[2, :]
    D = fss_raw[3, :]

    # Convert to Sun vector
    x = (((A+B) - (C+D))/(A+B+C+D)) + x0
    y = (((A+D) - (B+C))/(A+B+C+D)) + y0

    T = np.array([[np.cos(theta), np.sin(theta)],
                 [-np.sin(theta), np.cos(theta)]])

    for i in range(len(A)):
        v = np.matmul(T, np.array([x[i], y[i]]))
        x[i] = v[0]
        y[i] = v[1]

    # Get angles phi, theta
    phi = np.arctan2(x, y)
    # if phi < 0:
    #     phi = phi + 2*np.pi
    t = (x**2 + y**2)**0.5
    theta = np.arctan2(t, h)

    # Compute FSS body vector
    R = np.array([np.cos(theta), np.sin(theta)*np.cos(phi),
                 np.sin(theta)*np.sin(phi)]).transpose()

    return R


def filter_raw(fss_raw, dark_threshold, bright_threshold, low_threshold):
    """ Filter raw fss data based on threshold data
    """
    idrm = []
    
    for i in range(len(fss_raw[0, :])):
        if np.sum(fss_raw[:, i]) < dark_threshold:
            idrm.append(i)
        if fss_raw[0, i] > bright_threshold or fss_raw[1, i] > bright_threshold or fss_raw[2, i] > bright_threshold or fss_raw[3, i] > bright_threshold:
            idrm.append(i)
        if fss_raw[0, i] < low_threshold or fss_raw[1, i] < low_threshold or fss_raw[2, i] < low_threshold or fss_raw[3, i] < low_threshold:
            idrm.append(i)

    return np.array(idrm)




def fss_model(x, fss_raw, sun_reference, mag, mag_reference, opt_mode=False):
    """ FSS Objective Function
    Difference between cosine angle between SUNREF-MAGREF and cosine angle between FSS-MAG 
    """
    q = x[4:]

    # FSS vector
    fss_vec = sunvector(x, fss_raw).transpose()

    # Normalize quaternion
    q = q/np.linalg.norm(q)

    # Rotate to SC frame q -> DCM
    DCM = la.q2a(q)
    for i in range(len(fss_vec[0,:])):
        fss_vec[:,i] = np.matmul(DCM, fss_vec[:,i]).transpose()

    # Normalize reference vectors
    s_n = np.sqrt(sun_reference[:,0]**2 +
                  sun_reference[:,1]**2 + sun_reference[:,2]**2)
    s = sun_reference.transpose()/s_n

    b_n = np.sqrt(mag_reference[:,0]**2 +
                  mag_reference[:,1]**2 + mag_reference[:,2]**2)
    b = mag_reference.transpose()/b_n

    # Normalize measurement vectors
    mag_n = np.sqrt(mag[:,0]**2 +
                  mag[:,1]**2 + mag[:,2]**2)
    b_meas = mag.transpose()/mag_n


    fss_vec_n = np.sqrt(fss_vec[0, :]**2 +
                  fss_vec[1, :]**2 + fss_vec[2, :]**2)
    fss_vec = fss_vec/fss_vec_n


    # Angle based error
    a_ref = np.zeros(len(fss_vec[:, 0]))
    a_meas = np.zeros(len(fss_vec[:, 0]))

    for i in range(len(fss_vec[:, 0])):
        a_meas[i] = np.arccos(np.dot(fss_vec[:,i], b_meas[:,i]))
        a_ref[i] = np.arccos(np.dot(s[:,i], b[:,i]))

        err = np.abs(a_meas-a_ref)

    if opt_mode:
        return np.sum(err**2)
    else:
        # a1[idrm] = np.NaN # Assign NaN to invalid numbers
        return err


def fss_cal(p, q, fss_raw, sun_reference, mag, mag_reference):
    """ CALIBRATE FSS

    Inputs:
    - fss_raw         : Raw FSS data
    Outputs:
    - cal_param       : Post-calibration SC parameters
    """

    # Initial guess, and lower and upper bounds
    optim_option = {'maxiter': 4e3}
    # Equality Constraint of quaternion norm
    con = {'type': 'eq', 'fun': q_constraint}

    x0 = np.concatenate([p, q])
    opt_bounds = ( (0.1, 1.0), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-1, 1), (-1, 1), (-1, 1), (-1, 1))

    x0  = np.concatenate([p,q])
    opt_bounds  = ( (0.1,1.0), (-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf) ,(-1,1) ,(-1,1) ,(-1,1) ,(-1,1) )
    x_opt = minimize(fun=fss_model,args=(fss_raw, sun_reference, mag, mag_reference,True), x0=x0, bounds=opt_bounds,constraints=con, method='trust-constr', jac=False, options=optim_option, tol=1e-10)

    return x_opt
