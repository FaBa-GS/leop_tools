#!/usr/bin/python3.7
import argparse
import os
import numpy as np
import navpy as nav
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.coordinates import solar_system_ephemeris, get_sun, get_moon
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

from lib import libastro as la
from lib import libdata as dt
from lib import logger as log
from lib import libconfig as lconf

MARKER_SIZE = 3

# Get JPL Ephemerides (GCRS) approx. 30 MB
# Small dataset valid till 2050 (STK 12 uses outdated 405s.bsp)
solar_system_ephemeris.set(
    'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp')


def argumentsParser():
    parser = argparse.ArgumentParser(
        description='Plot script')
    # 'test_data', 'test.csv'
    parser.add_argument('-f', metavar='file', type=str, default=os.path.join(os.getcwd(), 'bro5_st.csv'),
                        help='Specify the datafile path')

    parser.add_argument('--tle-file', type=str, action='store',
                        default=os.path.join(os.getcwd(), 'test_data', 'tle_test.txt'),
                        help='TLE file: name \\n line1 \\n line2')

    parser.add_argument('-c', metavar='config_file', type=str, default=os.path.join(os.getcwd(), 'example_param'),
                        help='Specify config_file subfolder')

    parser.add_argument('-l', '--lla', metavar='lla', nargs='+', type=float, default=[57.023, 9.978, 10],
                        help='For GS tracking plot; Latitude [deg], Longitude [deg],  Altitude [m] (LLA) in WGS84 frame. Aalborg example: 57.023 9.978 10')

    parser.add_argument('--ref-mode', type=int, default=2,
                        help='Reference modes LVLH,LMRF-VEL,SPRF {2,4,6}')

    parser.add_argument('--euler-offset', type=float, nargs='+', default=[0, 0, 0],
                        help='Euler angle offset [X, Y, Z rad] (rotation sequence Z->Y->X)')

    parser.add_argument('--enable-ukf', action='store_true', default=False,
                        help='Add UKF info')

    parser.add_argument('--only-pass', action='store_true', default=False,
                        help='Only plot elevation angles >0 deg')

    parser.add_argument('--boresight', type=int, action='store', default=[0, 1, 0],
                        nargs='+', help='Boresight vector in Satellite body frame')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = argumentsParser()
    filename = args.f
    tle_file = args.tle_file
    config_subfolder = args.c
    lla = np.array(args.lla)
    ref_mode = args.ref_mode
    only_pass = args.only_pass
    boresight_vec = np.array(args.boresight)
    enable_ukf = args.enable_ukf
    euler_angles = np.array(args.euler_offset)

    # euler_angles = np.array([-1.5708, -0.7854, 0])
    # ref_mode = 6
    filename = 'bro5_ukf_ksat.csv'
    mode = 'adcs_data'

    # Create output subfolder if not exist
    path_output = os.path.join(os.getcwd(), 'output', mode)
    os.makedirs(path_output, exist_ok=True)

    # Initiate Logging
    plog = log.init_logging("PLOT", Path('logging.ini'))
    plog.info(
        '---------------------------------------------------------------------')
    plog.notice('Plotting ADCS data')
    if ref_mode == 4:
        if only_pass:
            plog.info('- GS tracking - pass only')
        else:
            plog.info('- GS tracking')

    # Read csv datafile
    D = dt.BeaconData(filename)

    # Title date and file data
    t_title = D.t_utc[0]
    t_file = datetime.utcfromtimestamp(D.t[0]).strftime('%Y%m%d_%H%M%S')

    # Load only startracker data when necessary
    if D.status_str is None:
        print("Startracker either disabled or not present.")
    else:
        # Load STR CONFIG files
        adcs_config = lconf.AdcsConfig(
            adcs_node=4, config_subfolder=config_subfolder)

        # INTMAG table
        adcs_config.load_table(lconf.STR_TABLE)
        plog.info('Table {:d} loaded.'.format(lconf.STR_TABLE))

    if ref_mode == 4:
        ##########################################
        # GS TRACKING PLOTS
        ##########################################

        # Convert to astropy Time
        t_vec = Time(list(D.t_utc), scale='utc')

        # Convert LLA to Cartesian [m] from ECEF -> ECI
        gsp_ecef = nav.lla2ecef(lla[0], lla[1], lla[2],
                                latlon_unit='deg', alt_unit='m', model='wgs84')
        gs_eci = EarthLocation.from_geodetic(lat=lla[0], lon=lla[1], height=lla[2]).get_gcrs(
            obstime=t_vec).cartesian.xyz.value  # m

        # Normalize boresight
        boresight_vec = boresight_vec / np.linalg.norm(boresight_vec)

        # Calculate elevation angle and boresight ctrl error
        el_angle = np.zeros(len(D.t))
        boresight_err_angle = np.zeros(len(D.t))

        for i in range(len(D.t)):
            el_angle[i] = la.compute_elevation(
                sat_pos=D.ephem_reci[:, i], gs_pos=gs_eci[:, i])
            boresight_err_angle[i] = la.get_cos_angle(a=boresight_vec,
                                                      b=np.zeros(3),
                                                      c=boresight_vec,
                                                      q=la.qinv(D.ctrl_errq[:, i]))

        # AOS, LOS, Max Elevation w/ status and lock
        ind_el0 = np.where(el_angle > 0)
        if ind_el0[0].size == 0:
            raise Warning(
                "Elevation angles are below zero! Check LLA coorindates, or perhaps not groundstation tracking.")

        # Find elevation angles above 0 deg (start and end)
        if len(ind_el0[0]) == 1:
            # Only single point
            ind0a = ind_el0[0][0]
            ind0b = ind0a
        else:
            id0a = ind_el0[0][0]
            # second zero point should be  within 1H
            idtb = np.where((t_vec - t_vec[id0a]) < 3600)
            id0b = np.intersect1d(idtb, ind_el0)

        if id0b.any():
            id0b = id0b[-1]
        else:
            id0b = -1

        # Replace to whole pass
        if not only_pass:
            idel0a = id0a
            idel0b = id0b
            id0a = 0
            id0b = len(t_vec) - 1

        id_el = np.arange(id0a, id0b)

        # Lock (status 4) and validity
        id_lock = np.where(D.status_str == 4)
        id_valid = np.where(D.st_valid == 1)
        id1 = np.intersect1d(id_el, id_lock)
        id2 = np.intersect1d(id_el, id_valid)

        # Remove invalidity
        t_st = D.t_dates[id_lock]

        # Elevation angles > 0 deg with lock
        lock_el = el_angle[id1]
        t_lock = D.t_dates[id1]
        t_vec_el = t_vec[id1]

        if lock_el.size > 0:
            max_el_lock = np.nanmax(np.rad2deg(lock_el))

            # Find when MAX EL occurs
            id_maxlock = np.nanargmax(lock_el)
            t_maxlock = t_lock[id_maxlock]
        else:
            id_maxlock = None
            max_el_lock = 0
            t_maxlock = None

        # Repeat for validity
        valid_el = el_angle[id2]
        t_valid = D.t_dates[id2]
        t_vec_el2 = t_vec[id2]
        if valid_el.size > 0:
            max_el_lock2 = np.nanmax(np.rad2deg(valid_el))

            id_maxvalid = np.nanargmax(valid_el)
            t_maxvalid = t_lock[id_maxvalid]
        else:
            id_maxvalid = None
            max_el_lock2 = 0
            t_maxvalid = None

        # Find closest validity to max elevation
        idmax_el = np.nanargmax(el_angle)
        if id_valid[0].any():
            idbefore_max = np.arange(0, idmax_el)
            idafter_max = np.arange(idmax_el, len(el_angle))
            idrange0 = np.intersect1d(idbefore_max, id_valid)[-1]
            idrange1 = np.intersect1d(idafter_max, id_valid)
            t_range0 = D.t_dates[idrange0]
            if idrange1.any():
                t_range1 = D.t_dates[idrange1[0]]
            else:
                t_range1 = len(el_angle)
        else:
            t_range0 = D.t_dates[0]
            t_range1 = D.t_dates[-1]

        plog.notice("MAX ELEVATION ANGLE ")
        plog.info(
            "- ST status 4: {:f} @ {}".format(max_el_lock, t_vec_el[id_maxlock]))
        plog.info(
            "- ST valid   : {:f} @ {}".format(max_el_lock2, t_vec_el2[id_maxvalid]))

        # Plot Figure
        plot_cnt = 4

        plot_name = os.path.join(
            path_output, 'gstrackplot_' + t_file + '.png')

        fig, ax = plt.subplots(nrows=plot_cnt, ncols=1, figsize=(19, 19))
        fig.suptitle('ADCS tracking data - ' + t_title + '\n\n \
            Max. Elevation with ST Valid = {:.2f} [deg] \n \
            Max. Elevation with ST Lock  = {:.2f} [deg] \n  \
            LLA: {:.2f}, {:.2f}, {:.1f} [deg,deg,m]'.format(max_el_lock2, max_el_lock, *lla), fontsize=14)
        fig.subplots_adjust(hspace=0.2)

        # ST LOCK
        plot_nr = 0
        t = D.t_dates
        x = D.status_str
        t2 = D.t_dates
        x2 = D.st_valid

        if (x is not None or x2 is not None):
            ax[plot_nr].set_title('ST Status')
            ax[plot_nr].grid()
            if x is not None:
                ax[plot_nr].plot_date(
                    t, x, linewidth=0, markersize=MARKER_SIZE)
            if x2 is not None:
                ax[plot_nr].plot_date(
                    t2, x2, linewidth=0, markersize=MARKER_SIZE)
            ax[plot_nr].legend(['Status', 'Valid'])
            ax[plot_nr].axvline(x=t_maxlock, linestyle='--', c='k')
            if only_pass:
                ax[plot_nr].set_xlim([t[id0a], t[id0b]])
            else:
                ax[plot_nr].axvline(x=t_range0, linestyle='-.', c='b')
                ax[plot_nr].axvline(x=t_range1, linestyle='-.', c='b')
                ax[plot_nr].axvline(x=t[idel0a], linestyle='--', c='k')
                ax[plot_nr].axvline(x=t[idel0b], linestyle='--', c='k')
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))
            ax[plot_nr].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Elevation Angle [deg]
        plot_nr = plot_nr + 1
        indnan = np.isnan(x)
        t = D.t_dates
        x = np.rad2deg(el_angle)

        if (x is not None):
            ax[plot_nr].set_title('Elevation Angle')
            ax[plot_nr].grid()
            ax[plot_nr].plot_date(t, x, linewidth=0, markersize=MARKER_SIZE)
            ax[plot_nr].axvline(x=t_maxlock, linestyle='--', c='k')
            if only_pass:
                ax[plot_nr].set_xlim([t[id0a], t[id0b]])
                ax[plot_nr].set_ylim(
                    [0, np.rad2deg(np.nanmax(el_angle[id0a:id0b]))])
            else:
                ax[plot_nr].axvline(x=t_range0, linestyle='-.', c='b')
                ax[plot_nr].axvline(x=t_range1, linestyle='-.', c='b')
                ax[plot_nr].axvline(x=t[idel0a], linestyle='--', c='k')
                ax[plot_nr].axvline(x=t[idel0b], linestyle='--', c='k')
            ax[plot_nr].set_ylabel('Angle [deg]')
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))
            ax[plot_nr].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Gyro
        plot_nr = plot_nr + 1
        t = D.t_dates
        x = D.gyro
        t2 = D.t_dates
        x2 = D.extgyro

        ax[plot_nr].grid()
        for i in range(x.shape[0]):
            if x2 is not None:
                ax[plot_nr].set_title('Extgyro')
                ax[plot_nr].plot_date(
                    t, x[i, :], linewidth=0, markersize=MARKER_SIZE)
            elif (x is not None and x2 is None):
                ax[plot_nr].set_title('Intgyro')
                ax[plot_nr].plot_date(
                    t2, x2[i, :], linewidth=0, markersize=MARKER_SIZE)
        ax[plot_nr].legend(['X', 'Y', 'Z'])
        ax[plot_nr].axvline(x=t_maxlock, linestyle='--', c='k')
        if only_pass:
            ax[plot_nr].set_xlim([t[id0a], t[id0b]])
        else:
            ax[plot_nr].axvline(x=t_range0, linestyle='-.', c='b')
            ax[plot_nr].axvline(x=t_range1, linestyle='-.', c='b')
            ax[plot_nr].axvline(x=t[idel0a], linestyle='--', c='k')
            ax[plot_nr].axvline(x=t[idel0b], linestyle='--', c='k')
        ax[plot_nr].xaxis.set_major_formatter(
            dates.DateFormatter('%H:%M:%S'))
        ax[plot_nr].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Control Error
        plot_nr = plot_nr + 1
        t = D.t_dates
        x = np.rad2deg(boresight_err_angle)

        if (x is not None):
            ax[plot_nr].set_title('Boresight Control Error')
            ax[plot_nr].grid()
            ax[plot_nr].plot_date(t, x, linestyle='solid',
                                  markersize=MARKER_SIZE)
            ax[plot_nr].set_ylabel('Error [deg]')
            ax[plot_nr].axvline(x=t_maxlock, linestyle='--', c='k')
            if only_pass:
                ax[plot_nr].set_xlim([t[id0a], t[id0b]])
            else:
                ax[plot_nr].axvline(x=t_range0, linestyle='-.', c='b')
                ax[plot_nr].axvline(x=t_range1, linestyle='-.', c='b')
                ax[plot_nr].axvline(x=t[idel0a], linestyle='--', c='k')
                ax[plot_nr].axvline(x=t[idel0b], linestyle='--', c='k')
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))

        fig.savefig(plot_name)
    else:
        ##########################################
        # OTHER REF MODE PLOTS
        ##########################################

        plot_cnt = 3

        t_title = D.t_utc[0]
        plot_name = os.path.join(path_output, 'adcs_plot_' + t_file + '.png')

        # Plot Figure
        fig, ax = plt.subplots(nrows=plot_cnt, ncols=1, figsize=(19, 26))
        fig.suptitle('ADCS data - ' + t_title)
        fig.subplots_adjust(hspace=0.4)

        # ST LOCK
        plot_nr = 0
        t = D.t_dates
        x = D.status_str
        t2 = D.t_dates
        x2 = D.st_valid

        if (x is not None or x2 is not None):
            ax[plot_nr].set_title('ST Status')
            ax[plot_nr].grid()
            if x is not None:
                ax[plot_nr].plot_date(
                    t, x, linewidth=0, markersize=MARKER_SIZE)
            if x2 is not None:
                ax[plot_nr].plot_date(
                    t2, x2, linewidth=0, markersize=MARKER_SIZE)
            ax[plot_nr].legend(['Status', 'Valid'])
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))
            ax[plot_nr].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Gyro
        plot_nr = plot_nr + 1
        t = D.t_dates
        x = D.gyro
        t2 = D.t_dates
        x2 = D.extgyro

        ax[plot_nr].grid()
        for i in range(x.shape[0]):
            # if x2 is not None:
            ax[plot_nr].set_title('Ext. Gyro')
            ax[plot_nr].plot_date(t, x2[i, :], linewidth=0,
                                  markersize=MARKER_SIZE)
            # elif (x is not None and x2 is None):
            #     ax[plot_nr].set_title('Int. Gyro')
            #     ax[plot_nr].plot_date(t2, x2[i,:], linewidth=0,markersize=MARKER_SIZE)
        ax[plot_nr].legend(['X', 'Y', 'Z'])
        ax[plot_nr].xaxis.set_major_formatter(
            dates.DateFormatter('%H:%M:%S'))
        ax[plot_nr].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Control Error
        plot_nr = plot_nr + 1
        t = D.t_dates
        x = D.ctrl_err_euler

        if (x is not None):
            ax[plot_nr].set_title('Control Error')
            ax[plot_nr].grid()
            for i in range(x.shape[0]):
                ax[plot_nr].plot_date(
                    t, x[i, :], linestyle='solid', markersize=MARKER_SIZE)
            ax[plot_nr].legend(['X', 'Y', 'Z'])
            ax[plot_nr].set_ylabel('Error [deg]')
            # ax[plot_nr].set_ylim([-2,2])
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))

        if ref_mode == 4:
            plt.setp(ax, xlim=[t[id0a], t[id0b]])
        else:
            plt.setp(ax, xlim=[t[0], t[-1]])
        fig.savefig(plot_name)

    ##########################################
    # UKF PLOTS
    ##########################################

    if enable_ukf:

        plot_cnt = 4

        plot_name = os.path.join(
            path_output, 'adcs_ukf_plot_' + t_file + '.png')

        # Plot Figure
        fig, ax = plt.subplots(nrows=plot_cnt, ncols=1, figsize=(19, 26))
        fig.suptitle('ADCS-UKF data - ' + t_title)
        fig.subplots_adjust(hspace=0.4)

        # UKF - q
        plot_nr = 0
        t = D.t_dates
        x = D.ukf_X

        if (x is not None):
            ax[plot_nr].set_title('UKF - q')
            ax[plot_nr].grid()
            if x is not None:
                ax[plot_nr].plot_date(
                    t, x[0:4, :].transpose(), linewidth=0, markersize=MARKER_SIZE)
            ax[plot_nr].legend(['q0', 'q1', 'q2', 'q3'])
            ax[plot_nr].set_ylabel('q')
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))

        plot_nr = plot_nr + 1
        t = D.t_dates
        x = D.ukf_Pdiag

        if (x is not None):
            ax[plot_nr].set_title('UKF - P q error')
            ax[plot_nr].grid()
            if x is not None:
                ax[plot_nr].plot_date(
                    t, x[0:3, :].transpose(), linewidth=0, markersize=MARKER_SIZE)
            ax[plot_nr].legend(['P_errq_1', 'P_errq_2', 'P_errq_3'])
            ax[plot_nr].set_ylabel('Err Cov [-]')
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))

        # UKF - w
        plot_nr = plot_nr + 1
        t = D.t_dates
        x = D.ukf_X

        if (x is not None):
            ax[plot_nr].set_title('UKF - w')
            ax[plot_nr].grid()
            if x is not None:
                ax[plot_nr].plot_date(
                    t, np.rad2deg(x[4:7, :].transpose()), linewidth=0, markersize=MARKER_SIZE)
            ax[plot_nr].legend(['x', 'y', 'z'])
            ax[plot_nr].set_ylabel('Omega [deg/s]')
            # ax[plot_nr].set_ylim([-0.1, 0.1])
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))

        plot_nr = plot_nr + 1
        t = D.t_dates
        x = D.ukf_Pdiag

        if (x is not None):
            ax[plot_nr].set_title('UKF - P gyrobias errror')
            ax[plot_nr].grid()
            if x is not None:
                ax[plot_nr].plot_date(
                    t, np.rad2deg(x[3:6, :].transpose()), linewidth=0, markersize=MARKER_SIZE)
            ax[plot_nr].legend(['x', 'y', 'z'])
            ax[plot_nr].set_ylabel('Err Cov [deg/s]')
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))

        if ref_mode == 4:
            plt.setp(ax, xlim=[t[id0a], t[id0b]])
        else:
            plt.setp(ax, xlim=[t[0], t[-1]])
        fig.savefig(plot_name)

    ##########################################
    # Plot Planetary bodies
    ##########################################

    if (D.ephem_reci is not None) and (D.ephem_veci is not None):
        sgp4_pos = D.ephem_reci
        sgp4_vel = D.ephem_veci
    else:
        # Run SGP4 to propagate position and velocity
        sgp4_data = la.sgp4_tle(tle_file, D.t)

        sgp4_pos = sgp4_data['sat_pos']
        sgp4_vel = sgp4_data['sat_vel']

    # Get JPL Sun and Moon positions
    sun_pos_gcrs, moon_pos_gcrs = la.get_moon_sun_pos(D.t)

    # Determine eclipse condition
    e_bool = la.compute_e_bool(sun_pos_gcrs, sgp4_pos)

    # Pack data
    data = {'t_vec': D.t, 'sat_pos_gcrs': sgp4_pos,
            'sun_pos_gcrs': sun_pos_gcrs, 'moon_pos_gcrs': moon_pos_gcrs}

    plog.info("- q_st       : {:.4f} {:.4f} {:.4f} {:.4f}".format(*adcs_config.str.str_q))

    # Use onboard REF FRAME
    if (D.ctrl_refq is not None):
        plog.info("Reference frame using ctrl_refq data.")
        q_IST = np.zeros(D.ctrl_refq.shape)

        for i in range(len(D.t)):
            # q_IST = q_ISC * q_SCST
            q_IST[:, i] = la.qMult(D.ctrl_refq[:, i], adcs_config.str.str_q)

    else:
        plog.info("No ctrl_refq data provided. Generating reference based on TLE + provided input arguments")
        q_IST = np.zeros(D.ctrl_errq.shape)

        # Convert LLA to Cartesian [m]
        gsp_ecef = nav.lla2ecef(lla[0], lla[1], lla[2],
                                latlon_unit='deg', alt_unit='m', model='wgs84')

        # Print ECEF
        plog.info('GS ECEF coordinates: {:f} {:f} {:f}'.format(*gsp_ecef))

        # LLA in ECI
        t_astropy = Time(D.t_utc, scale='utc')
        gs_eci = EarthLocation.from_geodetic(lat=lla[0], lon=lla[1], height=lla[2]).get_gcrs(
            obstime=t_astropy).cartesian.xyz.value  # m

        if (ref_mode == 4):
            plog.info("Reference frame: LM TRACKING w/ Velocity vector")
        elif (ref_mode == 6):
            plog.info("Reference frame: SUNPOINTING")
        else:
            plog.info("Reference frame: LVLH")

        q_euler_offset = la.euler2q(euler_angles)

        # Generate reference frame is unavailable
        for i in range(len(D.t)):
            if ref_mode == 4:
                # TRACK
                R_IREF = la.lm_frame(
                    gs_eci[:, i], sgp4_pos[:, i], sgp4_vel[:, i], use_sunvector=False)
            elif (ref_mode == 6):
                # SUNPOINT
                R_IREF = la.sun_frame(sun_pos_gcrs[:, i], sgp4_pos[:, i], np.array([0, 0, 1]))

            else:
                # TODO: Use configuration for which reference frame to take.
                # For now the rest is using LVLH FRAME

                R_IREF = la.lvlh_frame(sgp4_pos[:, i], sgp4_vel[:, i])

            # R to q
            q_IREF = la.a2q(R_IREF)

            # Multiply euler_offset
            q_IREF2 = la.qMult(q_IREF, q_euler_offset)

            # q_IST = q_ISC * q_SCST
            q_IST[:, i] = la.qMult(q_IREF2, adcs_config.str.str_q)

    # Normalize boresight
    sc_param = {}
    sc_param['st_boresight'] = np.array([1, 0, 0])

    # Compute angles
    moon_angle, earth_angle, sun_angle = la.get_astro_body_angles2(
        data, sc_param, q_IST)

    # Compute Figure
    if ref_mode:
        angles = {'moon_angle': moon_angle, 'earth_angle': earth_angle,
                  'sun_angle': sun_angle, 'el_angle': el_angle, 'e_bool': e_bool}
    else:
        angles = {'moon_angle': moon_angle, 'earth_angle': earth_angle,
                  'sun_angle': sun_angle, 'e_bool': e_bool}

    plot_name = os.path.join(
        path_output, 'ref' + str(ref_mode) + '_st_angles_plot_' + t_file + '.png')

    path_outputfig = os.path.join(path_output, plot_name)
    if ref_mode == 4:
        plt.setp(ax, xlim=[t[id0a], t[id0b]])
    else:
        plt.setp(ax, xlim=[t[0], t[-1]])
    la.plot_astro_body_angles(path_outputfig, D.t, **angles)