#!/usr/bin/python3.7

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from lib import libfpgen as fp
from lib import logger as log

#PATH_FILE = '/flash1/'
#CALIBRATION_BEACON = 25

def argumentsParser():
    parser = argparse.ArgumentParser(
        description='Sband fp generator for Osiris')

    parser.add_argument('-f', metavar='file', type=str, default='Camera_osiris.fp',
                        help='Specify the name to the flightplanner file. File name should end with .fp e.g. <filename>.fp')

    parser.add_argument('-t0', metavar='utc_time0', default="2021-08-20T09:30:00",
                        help='AOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')
    
    parser.add_argument('-t1', metavar='utc_time1', default="2021-08-20T09:40:00",
                        help='LOS UTC time in format yyyy-mm-ddTHH:MM:SS e.g. 2021-08-20T09:56:18')

    parser.add_argument('--bcn_name', '-b', type=str, action='store', default='Cam_os',
                        help='test number for naming beacon bin files')

    parser.add_argument('--test_nr', '-n', type=int, action='store', default=0,
                        help='test number for naming beacon bin files')

    parser.add_argument('--mockup',  action='store_true', default=False,
                        help='Enable mockup mode')

    args = parser.parse_args()
    return args



if __name__ == '__main__':

    # arguments
    args = argumentsParser()
    fpfile = args.f
    test_nr = args.test_nr
    t_utc0 = args.t0
    t_utc1 = args.t1
    bcn_fname = args.bcn_name
    enable_mockup = args.mockup

    mode = 'osiris-camera'

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

    camera_fp = fp.FlightPlan(filename=path_fpfile, cmd_name=bcn_fname,
                        linecount=0, t_cmd=t_unix_start, test_nr=test_nr)






    # Rotate Satellite (20 min before AOS)
    #######################################


    acs_mode = 4
    euler_offset = np.array([1.5708, 0, -1.5708])
    location = np.array([4186284 , 834837, 4723172 ])
    
    print(camera_fp.get_cmd_time('utc') + ": ROTATE TO TARGET")
    
    camera_fp.change_acs_mode(euler_angle=euler_offset,location=location,mode=acs_mode)  

    camera_fp.delay(1000)

    if enable_mockup:
        # Increase HK to 1 Hz (-5 min AOS)
        #######################################

        camera_fp.printline(
            "hk_srv beacon samplerate {:d} highest".format(CALIBRATION_BEACON))

        camera_fp.delay(10)


    # Power on Camera
    ####################################### 
    print(camera_fp.get_cmd_time('utc') + ": Powering on Camera")

    camera_fp.printline("eps node 2")
    camera_fp.printline("eps output 1 0 600")
    camera_fp.printline("eps output 1 1 0")

    camera_fp.delay(30) 
    camera_fp.printline("cmp clock sync 7")
    camera_fp.printline("cmp clock sync 6")
    camera_fp.delay(30)


    # Set Camera Configurations
    #######################################
    print(camera_fp.get_cmd_time('utc') + ": Setting camera configurations")

    camera_fp.printline('cam node 6')
    camera_fp.printline('rparam download 6 1 ')
    camera_fp.printline('rparam set exposure-us 5000')
    camera_fp.printline('rparam set gain-global 30000')
    camera_fp.printline('rparam send')



    # AOS
    #######################################
    t_new = int(t_aos)
    camera_fp.set_time(t_new)
    camera_fp.printline('cam snap')
    camera_fp.delay(3)
    camera_fp.printline('cam snap')
    camera_fp.delay(2)
    camera_fp.printline('cam snap')
    camera_fp.delay(2)
    camera_fp.printline('cam snap -sti')
    camera_fp.delay(2)
    camera_fp.printline('cam snap -sti')
    camera_fp.delay(2)
    camera_fp.printline('cam snap -sti')
    print(camera_fp.get_cmd_time('utc') + ": Taking Pictures")

    # LOS
    #######################################
    t_new = int(t_los)
    camera_fp.set_time(t_new)

    print(camera_fp.get_cmd_time('utc') + ": LOS")
    camera_fp.delay(5)
    # Set Camera Configurations back
    #######################################
    print(camera_fp.get_cmd_time('utc') + ": Set Camera Configuratins Back and Switch off camera")

    camera_fp.printline('eps output 1 0 0')
    camera_fp.delay(5)


    # STOP HK COLLECTION TIME
    #######################################
    t_hk1 = camera_fp.get_cmd_time() 

    # BACK TO NOMINAL MODE
    #######################################
    euler_offset = np.array([1.5708, 0, -1.5708])
    acs_mode = 2
    camera_fp.change_acs_mode(euler_angle=euler_offset,mode=acs_mode)


    # CLOSE FILE
    camera_fp.close()


































