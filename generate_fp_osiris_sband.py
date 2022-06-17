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

def argumentsParser():
    parser = argparse.ArgumentParser(
        description='Sband fp generator for Osiris')

    parser.add_argument('-f', metavar='file', type=str, default='sband_osiris.fp',
                        help='Specify the name to the flightplanner file. File name should end with .fp e.g. <filename>.fp')

    parser.add_argument('-t0', metavar='utc_time0', default="2021-08-20T09:30:00",
                        help='AOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')
    
    parser.add_argument('-t1', metavar='utc_time1', default="2021-08-20T09:40:00",
                        help='LOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')

    parser.add_argument('--bcn_name', '-b', type=str, action='store', default='sband_os',
                        help='test number for naming beacon bin files')

    parser.add_argument('--test_nr', '-n', type=int, action='store', default=0,
                        help='test number for naming beacon bin files')

    parser.add_argument('--mockup',  action='store_true', default=False,
                        help='Enable mockup mode')

    args = parser.parse_args()
    return args

MAG_BASELINE = np.array([5.665000, 8.663000, 12.640000])
SR2000_OFFSET = np.array([-7.0496, 27.9088, -24.2482])
SR2000_S_OFFSET = np.array([-5.56, -3.0, -3.35])

if __name__ == '__main__':

    # arguments
    args = argumentsParser()
    fpfile = args.f
    test_nr = args.test_nr
    t_utc0 = args.t0
    t_utc1 = args.t1
    bcn_fname = args.bcn_name
    enable_mockup = args.mockup

    mode = 'osiris-sband'

    # Convert to unix
    t_aos = datetime.fromisoformat("{}+00:00".format(t_utc0)).timestamp()
    t_los = datetime.fromisoformat("{}+00:00".format(t_utc1)).timestamp()
    t_unix_start = t_aos - 20*60

    # Create output subfolder if not exist
    path_output = os.path.join(os.getcwd(), 'output', mode)
    if not os.path.exists(path_output):
        os.mkdir(path_output)


    # Create Flightplan
    print("Creating flightplan")
    path_fpfile = os.path.join(path_output, fpfile)

    sband_fp = fp.FlightPlan(filename=path_fpfile, cmd_name=bcn_fname,
                        linecount=0, t_cmd=t_unix_start, test_nr=test_nr)

    # Rotate Satellite (20 min before AOS)
    #######################################
    acs_mode = 4
    euler_offset = np.array([1.5708, 0, -1.5708])
    location = np.array([4216463.545 , 810948.835, 4701434.265 ])
    
    print(sband_fp.get_cmd_time('utc') + ": ROTATE TO TARGET GS")
    
    sband_fp.change_acs_mode(euler_angle=euler_offset,location=location,mode=acs_mode)  

    sband_fp.delay(1000)

    if enable_mockup:
        # Increase HK to 1 Hz (-5 min AOS)
        #######################################

        sband_fp.printline(
            "hk_srv beacon samplerate {:d} highest".format(CALIBRATION_BEACON))

        sband_fp.delay(10)

    # HK START
    ####################################### 
    t_hk0 = sband_fp.get_cmd_time() 

    # Power on SDR 
    ####################################### 
    print(sband_fp.get_cmd_time('utc') + ": POWER ON SDR")

    sband_fp.printline("eps node 2")
    sband_fp.printline("eps output 2 0 1000")
    sband_fp.printline("eps output 2 1 0")
    if enable_mockup == False:
        mag_offset = MAG_BASELINE + SR2000_OFFSET
        sband_fp.printline('rparam download 4 86')
        sband_fp.printline('rparam set offset {:.3f} {:.3f} {:.3f}'.format(*mag_offset))
        sband_fp.printline('rparam send')

    sband_fp.delay(30) 
    sband_fp.printline("cmp clock sync 7")
    sband_fp.delay(30)

    # Power on ANTENNA 
    ####################################### 
    print(sband_fp.get_cmd_time('utc') + ": POWER ON ANTENNA")

    sband_fp.printline("eps output 7 0 1000")
    sband_fp.printline("eps output 7 1 0")
    sband_fp.delay(30)

    # Enable RX
    ####################################### 
    print(sband_fp.get_cmd_time('utc') + ": ENABLE RX")
    sband_fp.printline('rparam download 8 1')
    sband_fp.printline('rparam set ant-rx-on 1 1 1 1')
    sband_fp.printline('rparam send')

    # Enable TX
    ####################################### 
    print(sband_fp.get_cmd_time('utc') + ": ENABLE TX")
    sband_fp.printline('rparam download 8 5')
    sband_fp.printline('rparam set ant-tx-pwr 1 1 1 1')
    sband_fp.printline('rparam set ant-tx-en 1 1 1 1')
    sband_fp.printline('rparam set ant-tx-on 1 1 1 1')
    sband_fp.printline('rparam set gain -30')
    sband_fp.printline('rparam send')
    if enable_mockup == False:
        mag_offset = MAG_BASELINE + SR2000_OFFSET + SR2000_S_OFFSET
        sband_fp.printline('rparam download 4 86')
        sband_fp.printline('rparam set offset {:.3f} {:.3f} {:.3f}'.format(*mag_offset))
        sband_fp.printline('rparam send')

    # AOS
    #######################################
    t_new = int(t_aos)
    sband_fp.set_time(t_new)
    
    print(sband_fp.get_cmd_time('utc') + ": AOS")

    # LOS
    #######################################
    t_new = int(t_los)
    sband_fp.set_time(t_new)

    print(sband_fp.get_cmd_time('utc') + ": LOS")

    # Power off SDR 
    ####################################### 
    print(sband_fp.get_cmd_time('utc') + ": POWER OF SDR")

    sband_fp.printline("eps output 2 0 0")
    if enable_mockup == False:
        mag_offset = MAG_BASELINE
        sband_fp.printline('rparam download 4 86')
        sband_fp.printline('rparam set offset {:.3f} {:.3f} {:.3f}'.format(*mag_offset))
        sband_fp.printline('rparam send')
        sband_fp.delay(30)

    # Power off ANTENNA 
    ####################################### 
    print(sband_fp.get_cmd_time('utc') + ": POWER OFF ANTENNA")

    sband_fp.printline("eps output 7 0 0")
    sband_fp.delay(30)

    # STOP HK COLLECTION TIME
    #######################################
    t_hk1 = sband_fp.get_cmd_time() 

    # BACK TO NOMINAL MODE
    #######################################
    euler_offset = np.array([1.5708, 0, -1.5708])
    acs_mode = 2
    sband_fp.change_acs_mode(euler_angle=euler_offset,mode=acs_mode)

    # STORE and ZIP HK DATA
    #######################################
    
    print(sband_fp.get_cmd_time('utc') + ": HK") 

    if enable_mockup:
        datarate = 1
    else:
        datarate = 10
    
    n_samples = int(np.ceil((t_hk1 - t_hk0)/datarate))
    n_samples2 = int(np.ceil((t_hk1 - t_hk0)/30))
    
    fname = PATH_FILE + bcn_fname + str(22) + '_' + str(test_nr)
    sband_fp.store_hk(fname, 22, datarate, n_samples, t_hk1)
    sband_fp.zip_file(fname)
    fname2 = PATH_FILE + bcn_fname + str(23) + '_' + str(test_nr)
    sband_fp.store_hk(fname2, 23, datarate, n_samples, t_hk1)
    sband_fp.zip_file(fname2)
    fname3 = PATH_FILE + bcn_fname + str(25) + '_' + str(test_nr)
    sband_fp.store_hk(fname3, 25, datarate, n_samples, t_hk1)
    sband_fp.zip_file(fname3)
    
    fname4 = PATH_FILE + bcn_fname + str(31) + '_' + str(test_nr)
    sband_fp.store_hk(fname4, 31, 30, n_samples2, t_hk1)
    sband_fp.zip_file(fname4)
    fname5 = PATH_FILE + bcn_fname + str(30) + '_' + str(test_nr)
    sband_fp.store_hk(fname5, 30, 30, n_samples2, t_hk1)
    sband_fp.zip_file(fname5)
    # HK to default (-5 min AOS)
    #######################################
    if enable_mockup:
        sband_fp.printline(
            "hk_srv beacon samplerate {:d} high".format(CALIBRATION_BEACON))
        sband_fp.delay(10)


    # Display more info
    print("HK metadata:")
    print(" - Number of Samples: {:d}".format(n_samples))
    print(" - Name beacon file : {:s}.zip".format(fname,))
    print(" - Name beacon file : {:s}.zip".format(fname2,))
    print(" - Name beacon file : {:s}.zip".format(fname3,))
    print(" - Name beacon file : {:s}.zip".format(fname4,))
    print(" - Name beacon file : {:s}.zip".format(fname5, ))
    # CLOSE FILE
    sband_fp.close()
