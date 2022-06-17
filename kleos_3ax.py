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
        description='bdot cal fp generator')

    parser.add_argument('-f', metavar='file', type=str, default='fp_3ax.fp',
                        help='Specify the name to the flightplanner file. File name should end with .fp e.g. <filename>.fp')

    parser.add_argument('-t0', metavar='utc_time0', default="2021-08-20T09:30:00",
                        help='AOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')
    
    parser.add_argument('--bcn_name', '-b', type=str, action='store', default='adcs',
                        help='test number for naming beacon bin files')

    parser.add_argument('--test_nr', '-n', type=int, action='store', default=0,
                        help='test number for naming beacon bin files')


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # arguments
    args = argumentsParser()
    fpfile = args.f
    test_nr = args.test_nr
    t_utc0 = args.t0
    bcn_fname = args.bcn_name

    mode = '3axis'

    # Convert to unix
    t_aos = datetime.fromisoformat("{}+00:00".format(t_utc0)).timestamp()
    t_unix_start = t_aos

    # Create output subfolder if not exist
    path_output = os.path.join(os.getcwd(), 'output', mode)
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    # Create Flightplan
    print("Creating flightplan")
    path_fpfile = os.path.join(path_output, fpfile)

    adcs_fp = fp.FlightPlan(filename=path_fpfile, cmd_name=bcn_fname,
                        linecount=0, t_cmd=t_unix_start, test_nr=test_nr)
    
    # 3-ax
    ####################################### 
    print(adcs_fp.get_cmd_time('utc') + ": 3AXIS")
    
    adcs_fp.printline("hk_srv beacon samplerate 21 highest")
    
    t_hk0 = adcs_fp.get_cmd_time()
    adcs_fp.delay(10)
    adcs_fp.printline("adcs server 4 20")
    adcs_fp.printline("adcs state setads 11")
    adcs_fp.delay(300)

    t_hk1 = adcs_fp.get_cmd_time()

    # STORE and ZIP HK DATA
    #######################################
    
    print(adcs_fp.get_cmd_time('utc') + ": HK") 

    datarate = 10
    
    n_samples = int(np.ceil((t_hk1 - t_hk0)/datarate))
    n_samples2 = int(np.ceil((t_hk1 - t_hk0)/1))
    
    fname = PATH_FILE + bcn_fname + str(test_nr)
    adcs_fp.store_hk(fname, 21, 1, n_samples2, t_hk1)
    adcs_fp.zip_file(fname)

    fname = PATH_FILE + bcn_fname + str(test_nr)
    adcs_fp.store_hk(fname, 23, datarate, n_samples, t_hk1)
    adcs_fp.zip_file(fname)

    # CLOSE FILE
    adcs_fp.close()
