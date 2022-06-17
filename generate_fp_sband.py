#!/usr/bin/python3.7

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from lib import libfpgen as fp
from lib import logger as log

PATH_FILE = '/flash1/'
CALIBRATION_BEACON = 25
CONTROL_BEACON = 23
UKF_BEACON = 22

def argumentsParser():
    parser = argparse.ArgumentParser(
        description='generate_fo_sband.py: Flightplanner to generate files for magnetometer offset for Sband OPS')

    parser.add_argument('-f', metavar='file', type=str, default='sband_test.fp',
                        help='Specify the name to the flightplanner file. File name should end with .fp e.g. <filename>.fp')

    parser.add_argument('-t0', metavar='utc_time0', default="2021-08-20T09:30:00",
                        help='AOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')
    
    parser.add_argument('-t1', metavar='utc_time1', default="2021-08-20T09:40:00",
                        help='LOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')

    parser.add_argument('--bcn_name', '-b', type=str, action='store', default='sband',
                        help='test number for naming beacon bin files')

    parser.add_argument('--test_nr', '-n', type=int, action='store', default=0,
                        help='test number for naming beacon bin files')

    parser.add_argument('--mockup',  action='store_true', default=False,
                        help='Enable mockup mode')
    
    parser.add_argument('--intmag',  action='store_true', default=False,
                        help='Use internal magnetometer instead of external')

    parser.add_argument('-s', type=str, default=os.path.join(os.getcwd(),'test_data', 'example_offset_fp.json'),
                        help='JSON file containing settings')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # arguments
    args = argumentsParser()
    fpfile = args.f
    test_nr = args.test_nr
    t_utc0 = args.t0
    t_utc1 = args.t1
    
    enable_mockup = args.mockup
    bcn_fname = args.bcn_name
    setting_file = args.s
    enable_intmag = args.intmag = args.intmag

    mode = 'sband'

    # Selection Mag
    if enable_intmag:
        mag_in_use_str = 'internal - A3200'
    else:
        mag_in_use_str = 'external - M315'

    # Convert to unix
    t_aos = datetime.fromisoformat("{}+00:00".format(t_utc0)).timestamp()
    t_los = datetime.fromisoformat("{}+00:00".format(t_utc1)).timestamp()
    t_unix_start = t_aos - 20*60

    # Create output subfolder if not exist
    path_output = os.path.join(os.getcwd(), 'output', mode)
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    # Initiate Logging
    fp_log = log.init_logging("FP-{:s}".format(mode.upper()), Path('logging.ini'))
    fp_log.info(
        '---------------------------------------------------------------------')
    fp_log.notice('Initiating generation {:s} flightplan'.format(mode))

    # Load settings
    fp_log.info('Loading setting file: {:s}'.format(setting_file))
    fp_log.info('- mag      : {:s}'.format(mag_in_use_str))

    with open(setting_file) as sf:
        settings = json.load(sf)
        sband_settings = settings[mode]
        nominal_settings = settings['nominal']

    # Create Flightplan
    fp_log.notice("Creating flightplan")
    path_fpfile = os.path.join(path_output, fpfile)
    fp_log.info("Path to FP: {:s}".format(path_fpfile))

    sband_fp = fp.FlightPlan(filename=path_fpfile, cmd_name=bcn_fname,
                        linecount=0, t_cmd=t_unix_start, test_nr=test_nr)

    # Set base magnetometer offset
    #######################################
    for key in nominal_settings:
        if key['name'] == "BASE":
            base_offset = key['offset']

    sband_fp.set_magoffset(base_offset,use_intmag=enable_intmag)

    # Rotate Satellite (20 min before AOS)
    #######################################
    
    for key in sband_settings:
        if key['name'] == "MODE":
            acs_mode = key['mode']
            euler_offset = np.array(key['euler_offset'])
            location = np.array(key['location'])

    fp_log.info(sband_fp.get_cmd_time('utc') +
                ": Start rotating satellite ({:d}) to {:.4f} {:.4f} {:.4f} [rad]"
                .format(acs_mode, *euler_offset))
    
    sband_fp.change_acs_mode(euler_angle=euler_offset,location=location,mode=acs_mode)

    if enable_mockup:
        # Increase HK to 1 Hz (-5 min AOS)
        #######################################

        t_new = int(t_aos - 5*60)
        sband_fp.set_time(t_new)
        
        fp_log.info(sband_fp.get_cmd_time('utc') + ": Increase HK to 1hz")

        sband_fp.printline(
            "hk_srv beacon samplerate {:d} highest".format(CALIBRATION_BEACON))

        sband_fp.delay(10)

    # Start beacon collection (-2 min AOS)
    #######################################
    t_new = int(t_aos - 150)
    sband_fp.set_time(t_new)
    
    t_hk0 = sband_fp.get_cmd_time()
    fp_log.info(sband_fp.get_cmd_time('utc') + ": Start time hk collection") 

    t_new = int(t_aos - 120) 
    sband_fp.set_time(t_new)

    # Power on SDR SR2000 
    #######################################
    sr2000 = "SR2000"
    for key in sband_settings:
        if key['name'] == sr2000:
            sr2000_node = key['node']
            sr2000_power_node = key['power_node']
            sr2000_power_chan = key['power_chan']
            sr2000_offset = np.array(key['offset'])
        
    fp_log.info(sband_fp.get_cmd_time('utc') + ": POWER ON {:s}".format(sr2000))
    
    # Power on command
    sband_fp.printline("power on {:d} {:d} 0 900".format(sr2000_power_node,sr2000_power_chan))
    sband_fp.set_magoffset(sr2000_offset,use_intmag=enable_intmag)
    sband_fp.delay(30)

    # Clock sync
    sband_fp.printline("cmp clock sync {:d}".format(sr2000_node))
    sband_fp.delay(10)

    # Power on ANT2150 
    #######################################
    ant2150 = "ANT2150"
    for key in sband_settings:
        if key['name'] == ant2150:
            ant2150_power_node = key['power_node']
            ant2150_power_chan = np.array(key['power_chan'])
            ant2150_offset = np.array(key['offset'])
        
    fp_log.info(sband_fp.get_cmd_time('utc') + ": POWER ON {:s}".format(ant2150))

    sband_fp.printline("power on {:d} {:d} 0 850".format(ant2150_power_node,ant2150_power_chan))
    sband_fp.set_magoffset(ant2150_offset,use_intmag=enable_intmag)
    sband_fp.delay(30)

    # SR2000 LOAD GSCRIPT
    #######################################
    sr2000_load = "SR2000-LOAD"
    for key in sband_settings:
        if key['name'] == sr2000_load:
            power_node = key['power_node']
            power_chan = np.array(key['power_chan'])
            sr2000load_offset = np.array(key['offset'])
        
    fp_log.info(sband_fp.get_cmd_time('utc') + ": START {:s}".format(sr2000_load))
    
    sband_fp.printline("gscript server {:d}".format(sr2000_node))
    sband_fp.printline("gscript run /data/scripts/sband_sat_enable.gsh")
    sband_fp.set_magoffset(sr2000load_offset,use_intmag=enable_intmag)
    sband_fp.delay(30)
    
    # AOS
    #######################################
    t_new = int(t_aos)
    sband_fp.set_time(t_new)
    
    fp_log.info(sband_fp.get_cmd_time('utc') + ": AOS")

    # LOS
    #######################################
    t_new = int(t_los)
    sband_fp.set_time(t_new)
    
    fp_log.info(sband_fp.get_cmd_time('utc') + ": LOS")

    # SR2000 STOP GSCRIPT
    #######################################
    fp_log.info(sband_fp.get_cmd_time('utc') + ": STOP {:s}".format(sr2000_load))
    
    sband_fp.printline("z7000 node {:d}".format(sr2000_node))  
    sband_fp.printline("z7000 cmd exec /sbin/shutdown shutdown now")
    sband_fp.set_magoffset(-sr2000load_offset,use_intmag=enable_intmag) # Note negative sign here
    sband_fp.delay(30)

    # POWER OFF ANT2150
    #######################################
    fp_log.info(sband_fp.get_cmd_time('utc') + ": STOP {:s}".format(ant2150))
    
    sband_fp.printline("power off {:d} {:d}".format(ant2150_power_node,ant2150_power_chan))
    sband_fp.set_magoffset(-ant2150_offset,use_intmag=enable_intmag) # Note negative sign here
    sband_fp.delay(30)

    # POWER OFF SR2000
    #######################################
    fp_log.info(sband_fp.get_cmd_time('utc') + ": STOP {:s}".format(sr2000))
    
    sband_fp.printline("power off {:d} {:d}".format(sr2000_power_node,sr2000_power_chan))
    sband_fp.set_magoffset(-sr2000_offset,use_intmag=enable_intmag) # Note negative sign here
    sband_fp.delay(30)

    # BACK TO NOMINAL MODE
    #######################################

    for key in nominal_settings:
        if key['name'] == "MODE":
            acs_mode = key['mode']
            euler_offset = np.array(key['euler_offset'])

    fp_log.info(sband_fp.get_cmd_time('utc') +
                ": Start rotating satellite ({:d}) to {:.4f} {:.4f} {:.4f} [rad]"
                .format(acs_mode, *euler_offset))
    
    sband_fp.change_acs_mode(euler_angle=euler_offset,mode=acs_mode)

    # STORE and ZIP HK DATA
    #######################################
    t_hk1 = sband_fp.get_cmd_time() 
    fp_log.info(sband_fp.get_cmd_time('utc') + ": End time hk collection") 

    if enable_mockup:
        datarate = 1
    else:
        datarate = 10
    
    n_samples1 = int(np.ceil((t_hk1 - t_hk0)/datarate))
    
    fname = PATH_FILE + bcn_fname + \
                str(CALIBRATION_BEACON) + "_" + str(test_nr)
    sband_fp.store_hk(fname, CALIBRATION_BEACON, datarate, n_samples1, t_hk1)
    sband_fp.zip_file(fname)

    # Control + UKF beacon
    n_samples2 = int(np.ceil((t_hk1 - t_hk0)/10))
    
    fname2 = PATH_FILE + bcn_fname + \
                str(UKF_BEACON) + "_" + str(test_nr)
    sband_fp.store_hk(fname2, UKF_BEACON, 10, n_samples2, t_hk1)
    sband_fp.zip_file(fname2)

    fname3 = PATH_FILE + bcn_fname + \
                str(CONTROL_BEACON) + "_" + str(test_nr)
    sband_fp.store_hk(fname3, CONTROL_BEACON, 10, n_samples2, t_hk1)
    sband_fp.zip_file(fname3)

    # HK to default
    #######################################
    fp_log.info(sband_fp.get_cmd_time('utc') + ": HK to 0.1hz")

    sband_fp.printline(
        "hk_srv beacon samplerate {:d} high".format(CALIBRATION_BEACON))
    sband_fp.delay(10)

    # Display more info
    fp_log.notice("HK metadata:")
    fp_log.info(" - Number of Samples: {:d}".format(n_samples1))
    fp_log.info(" - Name beacon file : {:s}.zip".format(fname))
    fp_log.info(" - Number of Samples: {:d}".format(n_samples2))
    fp_log.info(" - Name beacon file : {:s}.zip".format(fname2))
    fp_log.info(" - Number of Samples: {:d}".format(n_samples2))
    fp_log.info(" - Name beacon file : {:s}.zip".format(fname3))

    # CLOSE FILE
    sband_fp.close()
