#!/usr/bin/python3.7

import argparse
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from sgp4.api import Satrec, WGS84
from lib import libastro as la
from lib import libfpgen as fp
from lib import logger as log

from astropy.coordinates import solar_system_ephemeris

# TODO: move this to settings
PATH_FILE = '/flash1/'
CALIBRATION_BEACON = 25
DATA_RATE = 10
AU_KM = 149597871
CONTROL_BEACON = 23
UKF_BEACON = 22

# Limitations:
# - Not for ADCS-A3200 versions above 5.3.1
# - LVLH mode only
# - SSO only

# Get JPL Ephemerides (GCRS)
solar_system_ephemeris.set(
    'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp')


def argumentsParser():
    parser = argparse.ArgumentParser(
        description='Flightplanner generator for calibration')
    parser.add_argument('--tle-file', type=str, action='store', default=os.path.join(os.getcwd(),'test_data','tle_test.txt'),
                        help='TLE file: name \\n line1 \\n line2')

    parser.add_argument('-f', metavar='file', type=str, default='test_cal.fp',
                        help='Specify the name to the flightplanner file. File name should end with .fp e.g. <filename>.fp')

    parser.add_argument('-t', metavar='utc_time0', default="2021-08-01T00:00:00",
                        help='UTC time in ISO format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-01T00:00:00')

    parser.add_argument('-s', type=str, default=os.path.join(os.getcwd(), 'test_data', 'example_calibration.json'),
                        help='JSON file containing settings')

    parser.add_argument('-n', type=int, action='store', default=np.array([0]),
                        nargs='+', help='test number for naming beacon bin files')

    parser.add_argument('--bcn_name', '-b', type=str, action='store', default='cal',
                        help='test number for naming beacon bin files')

    parser.add_argument('--eclipse', action='store_true', default=False,
                        help='Calibration performed only in Eclipse period')

    parser.add_argument('--de', type=int, action='store', default=np.array([0,0,30]),
                        nargs='+', help='delta Euler angle offset applied per test_nr')

    parser.add_argument('--more-beacons',  action='store_true', default=False,
                        help='Add UKF and CTRL beacons (22,23)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # arguments
    args = argumentsParser()
    tle_file = args.tle_file
    setting_file = args.s
    t_utc0 = args.t
    eclipse_mode = args.eclipse
    fpfile = args.f
    bcn_fname = args.bcn_name
    test_nr = np.array(args.n)
    d_euler = args.de
    more_beacons = args.more_beacons

    mode = 'calibration'

    # Create output subfolder if not exist
    path_output = os.path.join(os.getcwd(), 'output', mode)
    os.makedirs(path_output, exist_ok=True)

    # Initiate Logging
    calfp_log = log.init_logging("FP-{:s}".format(mode.upper()), Path('logging.ini'))
    calfp_log.info(
        '---------------------------------------------------------------------')
    calfp_log.notice('Initiating generation {:s} flightplan'.format(mode))
    
    if eclipse_mode:
        calfp_log.info('Eclipse calibration activated')

    # Load settings
    calfp_log.info('Loading setting file: {:s}'.format(setting_file))

    with open(setting_file) as sf:
        settings = json.load(sf)

    with open(setting_file) as sf:
        settings = json.load(sf)
        calibration_settings = settings[mode]
        nominal_settings = settings['nominal']

    # Load TLEs
    with open(tle_file) as f:
        lines = f.read().splitlines()
        sat_name = lines[0]
        tleline1 = lines[1]
        tleline2 = lines[2]

        calfp_log.info(sat_name)
        calfp_log.info(tleline1)
        calfp_log.info(tleline2)

    # Load satellite from TLEs
    satellite = Satrec.twoline2rv(tleline1, tleline2, WGS84)

    # Orbital period
    T_orbit = 1/(satellite.no_kozai/60/(2*np.pi))  # [s]
    calfp_log.info('Orbital period: {:0.1f} [s]'.format(T_orbit))

    # Time vector to compute (UTC -> UNIX)
    t_unix0 = datetime.fromisoformat(t_utc0).replace(
        tzinfo=timezone.utc).timestamp()

    # Create time vector dependent on test number x orbit
    t_end = t_unix0 + T_orbit*(len(test_nr)+2)
    t_vec = np.arange(t_unix0, t_end, 60)  # per minute to reduce load

    # Run SGP4 to propagate position and velocity
    sgp4_data = la.sgp4_tle(tle_file, t_vec)

    # Get JPL Sun and Moon positions
    sun_pos_gcrs, moon_pos_gcrs = la.get_moon_sun_pos(t_vec)

    # Determine eclipse condition
    e_bool = la.compute_e_bool(sun_pos_gcrs, sgp4_data['sat_pos'])

    # Start time eclipse
    e_diff = np.diff(e_bool)
    ind_start_e = np.where(e_diff == 1)
    ind_end_e = np.where(e_diff == -1)

    t_start = t_vec[ind_start_e]

    if eclipse_mode:
        t_end = t_vec[ind_end_e]
    else:
        t_end = t_vec[ind_start_e] + T_orbit

    ###############################
    # Create Flightplan
    ###############################

    calfp_log.notice("Creating flightplan")
    path_fpfile = os.path.join(path_output, fpfile)
    calfp_log.info("Path to FP: {:s}".format(path_fpfile))

    cal_fp = fp.FlightPlan(filename=path_fpfile, cmd_name=bcn_fname,
                           linecount=0, t_cmd=t_unix0, test_nr=test_nr[0])

    for i in range(len(test_nr)):
        
        # Test Number
        ##############################################################
        calfp_log.notice('Test number: {:d}'.format(test_nr[i]))
        
        # Set test_nr
        cal_fp.set_test_nr(test_nr[i])

        # Set start to eclipse time 1 minute before
        cal_fp.set_time(t_start[i] - 60)

        calfp_log.info(cal_fp.get_cmd_time('utc') + ": Eclipse start -60s")

        # Calculate euler_offset
        ##############################################################
        
        for key in calibration_settings:
            if key['name'] == "MODE":
                acs_mode = key['mode']
                euler_offset = np.array(key['euler_offset'])

        for key in calibration_settings:
            if key['name'] == "STARTRACKER":
                st_power_node = key['power_node']
                st_power_chan = key['power_chan']
                enable_st = key['enable_st']
                str_q = np.array(key['str_q'])

        n_offset = test_nr[i]
        DCM_offset = la.euler2a(np.array(d_euler)*n_offset)

        euler_nominal = np.array(euler_offset)
        DCM_nominal = la.euler2a(euler_nominal)
        DCM = np.matmul(DCM_offset, DCM_nominal)
        sat_euler = la.a2euler(DCM)

        calfp_log.info(cal_fp.get_cmd_time('utc') +
                    ": Start rotating satellite ({:d}) to {:.4f} {:.4f} {:.4f} [rad]"
                    .format(acs_mode, *sat_euler))
    
        cal_fp.change_acs_mode(euler_angle=sat_euler,mode=acs_mode)

        # Plot Moon, Earth, Sun, and eclipse condition 
        ##############################################################

        ind_t0 = np.argmax(t_vec >= t_start[i])
        ind_t1 = np.argmax(t_vec >= t_end[i])
        t_plot = t_vec[ind_t0:ind_t1]

        # Pack data
        data = {'t_vec': t_plot, 'sat_pos_gcrs': sgp4_data['sat_pos'][:,ind_t0:ind_t1],
                'sun_pos_gcrs': sun_pos_gcrs[:,ind_t0:ind_t1], 'moon_pos_gcrs': moon_pos_gcrs[:,ind_t0:ind_t1]}

        # Update euler offset
        sc_param = {}
        sc_param['euler_offset'] = sat_euler
        sc_param['q_st_sc'] = str_q
        sc_param['st_boresight'] = np.array([1,0,0])

        # Compute angles
        moon_angle, earth_angle, sun_angle = la.get_astro_body_angles(
            data, sc_param, la.lvlh_frame, 
            sat_posvec=sgp4_data['sat_pos'][:,ind_t0:ind_t1], 
            sat_velvec=sgp4_data['sat_vel'][:,ind_t0:ind_t1])

        # Determine eclipse condition
        e_bool_plot = la.compute_e_bool(sun_pos_gcrs[:,ind_t0:ind_t1], sgp4_data['sat_pos'][:,ind_t0:ind_t1])

        # Compute Figure
        angles = {'moon_angle': moon_angle, 'earth_angle': earth_angle,
                  'sun_angle': sun_angle, 'e_bool': e_bool_plot}

        figname = 'astro_angle' + str(test_nr[i]) + '.png'
        path_outputfig = os.path.join(path_output, figname)
        la.plot_astro_body_angles(path_outputfig, t_plot, **angles)
        
        # Power cycle and enable ST
        #######################################
        
        cal_fp.printline('rparam download 4 89')
        cal_fp.printline('rparam set en_str 0'.format(enable_st))
        cal_fp.printline('rparam send')

        if (enable_st) and (st_power_node != 0):    
            cal_fp.delay(10)

            cal_fp.printline('power on {:d} {:d} 0 {:d}'.format(
                st_power_node, st_power_chan, int(T_orbit)-300))

            cal_fp.delay(20)

            cal_fp.printline("rgosh server 4")
            cal_fp.printline("rgosh run 'st200 init 2'")
            cal_fp.delay(3)
            cal_fp.printline("rgosh run 'st200 custom \"ping\"'")
            cal_fp.delay(3)
            cal_fp.printline("rgosh run 'st200 custom \"set keep-alive off\"'")
            cal_fp.delay(3)
            cal_fp.printline(
                "rgosh run 'st200 custom \"set param 32 1a 00 00 00 00 00 00\"'")
            cal_fp.delay(3)
            cal_fp.printline(
                "rgosh run 'st200 custom \"set param 52 1a 00 00 00 00 00 00\"'")
            cal_fp.delay(3)
            cal_fp.printline(
                "rgosh run 'st200 custom \"set param 53 80 01 00 00 00 00 00\"'")
            cal_fp.delay(3)
            cal_fp.printline('rparam download 4 89')
            cal_fp.printline('rparam set en_str 1')
            cal_fp.printline('rparam send')


        # Collect Beacons
        #######################################
        calfp_log.info(cal_fp.get_cmd_time('utc') +
                       ": Start collecting calibration beacons")
        t_hk0 = cal_fp.get_cmd_time()  # Start beacon collection

        if eclipse_mode:
            cal_fp.set_time(t_end[i] - 60)
        else:
            # Allow a bit of time to rotate 
            cal_fp.set_time(t_end[i] - 600)

        calfp_log.info(cal_fp.get_cmd_time('utc') +
                       ": End collecting calibration beacons")
        t_hk1 = cal_fp.get_cmd_time()  # End beacon collection
        
        n_samples = int(round((t_hk1 - t_hk0)/DATA_RATE))

        bcn_fname1 = PATH_FILE + bcn_fname + \
            str(CALIBRATION_BEACON) + "_" + str(test_nr[i])

        cal_fp.store_hk(bcn_fname1, CALIBRATION_BEACON, DATA_RATE, n_samples, t_hk1)
        cal_fp.zip_file(bcn_fname1)

        if more_beacons:
            # Control + UKF beacon
            n_samples2 = int(np.ceil((t_hk1 - t_hk0)/10))
            
            bcn_fname2 = PATH_FILE + bcn_fname + \
                str(UKF_BEACON) + "_" + str(test_nr[i])

            cal_fp.store_hk(bcn_fname2, UKF_BEACON, 10, n_samples2, t_hk1)
            cal_fp.zip_file(bcn_fname2)


            bcn_fname3 = PATH_FILE + bcn_fname + \
                str(CONTROL_BEACON) + "_" + str(test_nr[i])

            cal_fp.store_hk(bcn_fname3, CONTROL_BEACON, 10, n_samples2, t_hk1)
            cal_fp.zip_file(bcn_fname3)

        # Display more info
        calfp_log.info("INFO: Beacon - TEST # {:d}".format(test_nr[i]))
        calfp_log.info(" - Number of Samples: {:d}".format(n_samples))
        calfp_log.info(" - Name beacon file : {:s}.zip".format(bcn_fname1))
        if more_beacons:
            calfp_log.info(" - Number of Samples: {:d}".format(n_samples2))
            calfp_log.info(" - Name beacon file : {:s}.zip".format(bcn_fname2))
            calfp_log.info(" - Number of Samples: {:d}".format(n_samples2))
            calfp_log.info(" - Name beacon file : {:s}.zip".format(bcn_fname3))
        
    if i == len(test_nr)-1:
            
            # BACK TO NOMINAL MODE
            #######################################

            for key in nominal_settings:
                if key['name'] == "MODE":
                    acs_mode = key['mode']
                    euler_offset = np.array(key['euler_offset'])

            calfp_log.info(cal_fp.get_cmd_time('utc') +
                        ": Start rotating satellite ({:d}) to {:.4f} {:.4f} {:.4f} [rad]"
                        .format(acs_mode, *euler_offset))
            
            cal_fp.change_acs_mode(euler_angle=euler_offset,mode=acs_mode)
            
    ###############################
    # CLOSE FILE
    ###############################
    cal_fp.close()
