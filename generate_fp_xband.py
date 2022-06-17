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
        description='generate_fp_xband.py: Flightplanner to generate files for magnetometer offset for xband OPS')

    parser.add_argument('-f', metavar='file', type=str, default='xband_test.fp',
                        help='Specify the name to the flightplanner file. File name should end with .fp e.g. <filename>.fp')

    parser.add_argument('-t0', metavar='utc_time0', default="2021-08-20T09:30:00",
                        help='AOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')
    
    parser.add_argument('-t1', metavar='utc_time1', default="2021-08-20T09:40:00",
                        help='LOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')

    parser.add_argument('--bcn_name', '-b', type=str, action='store', default='xband',
                        help='test number for naming beacon bin files')

    parser.add_argument('--test_nr', '-n', type=int, action='store', default=0,
                        help='test number for naming beacon bin files')

    parser.add_argument('--mockup',  action='store_true', default=False,
                        help='Enable mockup mode')

    parser.add_argument('-s', type=str, default=os.path.join(os.getcwd(),'test_data', 'example_offset_fp.json'),
                        help='JSON file containing settings')

    parser.add_argument('--intmag',  action='store_true', default=False,
                        help='Use internal magnetometer instead of external')

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
    
    mode = 'xband'

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
        xband_settings = settings[mode]
        nominal_settings = settings['nominal']

    # Create Flightplan
    fp_log.notice("Creating flightplan")
    path_fpfile = os.path.join(path_output, fpfile)
    fp_log.info("Path to FP: {:s}".format(path_fpfile))

    xband_fp = fp.FlightPlan(filename=path_fpfile, cmd_name=bcn_fname,
                        linecount=0, t_cmd=t_unix_start, test_nr=test_nr)

    # Set base magnetometer offset
    #######################################
    for key in nominal_settings:
        if key['name'] == "BASE":
            base_offset = key['offset']

    xband_fp.set_magoffset(base_offset,use_intmag=enable_intmag)

    # Rotate Satellite (20 min before AOS)
    #######################################
    
    for key in xband_settings:
        if key['name'] == "MODE":
            acs_mode = key['mode']
            euler_offset = np.array(key['euler_offset'])
            location = np.array(key['location'])

    fp_log.info(xband_fp.get_cmd_time('utc') +
                ": Start rotating satellite ({:d}) to {:.4f} {:.4f} {:.4f} [rad]"
                .format(acs_mode, *euler_offset))
    
    xband_fp.change_acs_mode(euler_angle=euler_offset,location=location,mode=acs_mode)

    if enable_mockup:
        # Increase HK to 1 Hz (-5 min AOS)
        #######################################

        t_new = int(t_aos - 5*60)
        xband_fp.set_time(t_new)
        
        fp_log.info(xband_fp.get_cmd_time('utc') + ": Increase HK to 1hz")

        xband_fp.printline(
            "hk_srv beacon samplerate {:d} highest".format(CALIBRATION_BEACON))

        xband_fp.delay(10)

    # Start beacon collection (-3 min AOS)
    #######################################
    t_new = int(t_aos - 270)
    xband_fp.set_time(t_new)

    t_hk0 = xband_fp.get_cmd_time() 
    fp_log.info(xband_fp.get_cmd_time('utc') + ": Start time hk collection") 
    t_new = int(t_aos - 240)
    xband_fp.set_time(t_new)

    # Power on SDR SR2000 
    #######################################
    sr2000 = "SR2000"
    for key in xband_settings:
        if key['name'] == sr2000:
            sr2000_node = key['node']
            sr2000_power_node = key['power_node']
            sr2000_power_chan = key['power_chan']
            sr2000_offset = np.array(key['offset'])
        
    fp_log.info(xband_fp.get_cmd_time('utc') + ": POWER ON {:s}".format(sr2000))
    
    # Power on command
    xband_fp.printline("power on {:d} {:d} 0 1000".format(sr2000_power_node,sr2000_power_chan))
    xband_fp.set_magoffset(sr2000_offset,use_intmag=enable_intmag)
    xband_fp.delay(30)

    # Clock sync
    xband_fp.printline("cmp clock sync {:d}".format(sr2000_node[0]))
    xband_fp.delay(10)

    # Power on PDU3 
    #######################################
    pdu3 = "PDU3"
    for key in xband_settings:
        if key['name'] == pdu3:
            pdu3_power_node = key['power_node']
            pdu3_power_chan = np.array(key['power_chan'])
            pdu3_offset = np.array(key['offset'])
        
    fp_log.info(xband_fp.get_cmd_time('utc') + ": POWER ON {:s}".format(pdu3))

    xband_fp.printline("power on {:d} {:d} 0 900".format(pdu3_power_node,pdu3_power_chan))
    xband_fp.set_magoffset(pdu3_offset,use_intmag=enable_intmag)
    xband_fp.delay(30)

    # Power on X_AFE1 
    #######################################
    xafe1 = "XAFE1"
    for key in xband_settings:
        if key['name'] == xafe1:
            xafe1_power_node = key['power_node']
            xafe1_power_chan = np.array(key['power_chan'])
            xafe1_offset = np.array(key['offset'])
        
    fp_log.info(xband_fp.get_cmd_time('utc') + ": POWER ON {:s}".format(xafe1))

    xband_fp.printline("power on {:d} {:d} 0 850".format(xafe1_power_node,xafe1_power_chan))
    xband_fp.set_magoffset(xafe1_offset,use_intmag=enable_intmag)
    xband_fp.delay(30)

    # Power on X_AFE2
    #######################################
    xafe2 = "XAFE2"
    for key in xband_settings:
        if key['name'] == xafe2:
            xafe2_power_node = key['power_node']
            xafe2_power_chan = np.array(key['power_chan'])
            xafe2_offset = np.array(key['offset'])
        
    fp_log.info(xband_fp.get_cmd_time('utc') + ": POWER ON {:s}".format(xafe2))

    xband_fp.printline("power on {:d} {:d} 0 800".format(xafe2_power_node,xafe2_power_chan))
    xband_fp.set_magoffset(xafe2_offset,use_intmag=enable_intmag)
    xband_fp.delay(30)

    # SR2000 LOAD - SET XBAND SETTINGS
    #######################################
    sr2000_load = "SR2000-LOAD"
    for key in xband_settings:
        if key['name'] == sr2000_load:
            sr2000load_offset = np.array(key['offset'])
        
    fp_log.info(xband_fp.get_cmd_time('utc') + ": START {:s}".format(sr2000_load))
    
    xband_fp.printline("rparam download {:d} 11".format(sr2000_node[1]))
    xband_fp.printline("rparam set modcod 13")
    xband_fp.printline("rparam set symrate 30000000")
    xband_fp.printline("rparam set roll_off 2")
    xband_fp.printline("rparam set gain 64")
    xband_fp.printline("rparam set enable true")
    xband_fp.printline("rparam send")
    xband_fp.set_magoffset(sr2000load_offset,use_intmag=enable_intmag)
    xband_fp.delay(30)
    
    # TX
    #######################################
    tx = "TX"
    for key in xband_settings:
        if key['name'] == tx:
            tx_offset = np.array(key['offset'])
        
    fp_log.info(xband_fp.get_cmd_time('utc') + ": START {:s}".format(tx))
    
    xband_fp.printline("rparam download {:d} 20".format(sr2000_node[1]))
    xband_fp.printline("rparam set ensm_mode[1] fdd")
    xband_fp.printline("rparam set fir_equalize[1] 1")
    xband_fp.printline("rparam send")

    xband_fp.printline("rparam download {:d} 13".format(sr2000_node[1]))
    xband_fp.printline("rparam set tx_mod 32apsk")
    xband_fp.printline("rparam set tx_pwr_lvl 2")
    xband_fp.printline("rparam set tx_freq 8100000000")
    xband_fp.printline("rparam set dyn_if_adjust false")
    xband_fp.printline("rparam set enable true")
    xband_fp.printline("rparam send")

    xband_fp.printline("rparam download {:d} 20".format(sr2000_node[1]))
    xband_fp.printline("rparam set tx_gain[1] -24.5")
    xband_fp.printline("rparam send")

    xband_fp.set_magoffset(tx_offset,use_intmag=enable_intmag)
    xband_fp.delay(30)

    # AOS
    #######################################
    t_new = int(t_aos)
    xband_fp.set_time(t_new)
    
    fp_log.info(xband_fp.get_cmd_time('utc') + ": AOS")

    # LOS
    #######################################
    t_new = int(t_los)
    xband_fp.set_time(t_new)
    
    fp_log.info(xband_fp.get_cmd_time('utc') + ": LOS")

    # DISABLE TX
    #######################################
    fp_log.info(xband_fp.get_cmd_time('utc') + ": STOP {:s}".format(tx))
    
    xband_fp.printline("rparam download {:d} 13".format(sr2000_node[1]))
    xband_fp.printline("rparam set enable false")
    xband_fp.printline("rparam send")
    xband_fp.set_magoffset(-tx_offset,use_intmag=enable_intmag) # Note negative sign here
    xband_fp.delay(30)

    # POWER OFF AFE2
    #######################################
    fp_log.info(xband_fp.get_cmd_time('utc') + ": STOP {:s}".format(xafe2))
    
    xband_fp.printline("power off {:d} {:d}".format(xafe2_power_node,xafe2_power_chan))
    xband_fp.set_magoffset(-xafe2_offset,use_intmag=enable_intmag) # Note negative sign here
    xband_fp.delay(30)

    # POWER OFF AFE1
    #######################################
    fp_log.info(xband_fp.get_cmd_time('utc') + ": STOP {:s}".format(xafe1))
    
    xband_fp.printline("power off {:d} {:d}".format(xafe1_power_node,xafe1_power_chan))
    xband_fp.set_magoffset(-xafe1_offset,use_intmag=enable_intmag) # Note negative sign here
    xband_fp.delay(30)

    # SR2000 STOP XBAND 
    #######################################
    fp_log.info(xband_fp.get_cmd_time('utc') + ": STOP {:s}".format(sr2000_load))
    
    xband_fp.printline("z7000 node {:d}".format(sr2000_node[0]))  
    xband_fp.printline("z7000 cmd exec /sbin/shutdown shutdown now")
    xband_fp.set_magoffset(-sr2000load_offset,use_intmag=enable_intmag) # Note negative sign here
    xband_fp.delay(30)

    # POWER OFF PDU3
    #######################################
    fp_log.info(xband_fp.get_cmd_time('utc') + ": STOP {:s}".format(pdu3))
    
    xband_fp.printline("power off {:d} {:d}".format(pdu3_power_node,pdu3_power_chan))
    xband_fp.set_magoffset(-pdu3_offset,use_intmag=enable_intmag) # Note negative sign here
    xband_fp.delay(30)

    # POWER OFF SR2000
    #######################################
    fp_log.info(xband_fp.get_cmd_time('utc') + ": STOP {:s}".format(sr2000))
    
    xband_fp.printline("power off {:d} {:d}".format(sr2000_power_node,sr2000_power_chan))
    xband_fp.set_magoffset(-sr2000_offset,use_intmag=enable_intmag) # Note negative sign here
    xband_fp.delay(30)

    # BACK TO NOMINAL MODE
    #######################################

    for key in nominal_settings:
        if key['name'] == "MODE":
            acs_mode = key['mode']
            euler_offset = np.array(key['euler_offset'])


    fp_log.info(xband_fp.get_cmd_time('utc') +
                ": Start rotating satellite ({:d}) to {:.4f} {:.4f} {:.4f} [rad]"
                .format(acs_mode, *euler_offset))
    
    xband_fp.change_acs_mode(euler_angle=euler_offset,mode=acs_mode)

    # STORE and ZIP HK DATA
    #######################################
    t_hk1 = xband_fp.get_cmd_time() 
    fp_log.info(xband_fp.get_cmd_time('utc') + ": End time hk collection") 

    if enable_mockup:
        datarate = 1
    else:
        datarate = 10
    
    n_samples1 = int(np.ceil((t_hk1 - t_hk0)/datarate))
    
    fname = PATH_FILE + bcn_fname + \
                str(CALIBRATION_BEACON) + "_" + str(test_nr)
    xband_fp.store_hk(fname, CALIBRATION_BEACON, datarate, n_samples1, t_hk1)
    xband_fp.zip_file(fname)
    
    # Control + UKF beacon
    n_samples2 = int(np.ceil((t_hk1 - t_hk0)/10))
    
    fname2 = PATH_FILE + bcn_fname + \
                str(UKF_BEACON) + "_" + str(test_nr)
    xband_fp.store_hk(fname2, UKF_BEACON, 10, n_samples2, t_hk1)
    xband_fp.zip_file(fname2)

    fname3 = PATH_FILE + bcn_fname + \
                str(CONTROL_BEACON) + "_" + str(test_nr)
    xband_fp.store_hk(fname3, CONTROL_BEACON, 10, n_samples2, t_hk1)
    xband_fp.zip_file(fname3)

    # HK to default (-5 min AOS)
    #######################################
    fp_log.info(xband_fp.get_cmd_time('utc') + ": Increase HK to 0.1hz")

    xband_fp.printline(
        "hk_srv beacon samplerate {:d} high".format(CALIBRATION_BEACON))
    xband_fp.delay(10)

    # Display more info
    fp_log.notice("HK metadata:")
    fp_log.info(" - Number of Samples: {:d}".format(n_samples1))
    fp_log.info(" - Name beacon file : {:s}.zip".format(fname))
    fp_log.info(" - Number of Samples: {:d}".format(n_samples2))
    fp_log.info(" - Name beacon file : {:s}.zip".format(fname2))
    fp_log.info(" - Number of Samples: {:d}".format(n_samples2))
    fp_log.info(" - Name beacon file : {:s}.zip".format(fname3))
    
    # CLOSE FILE
    xband_fp.close()
