#!/usr/bin/python3.7

import argparse
import os
import pandas as pd
import numpy as np
from datetime import date, datetime
from astropy.time import Time

from lib import libtle as lt
from lib import libastro as la


def argumentsParser():
    parser = argparse.ArgumentParser(
        description='gps2tle.py: Generates TLE based on GPS datafile script')
    parser.add_argument('-f', metavar='file', type=str,
                        help='Specify the datafile path with gps_epoch, gps_pos, and gps_vel info.')

    parser.add_argument('-o', metavar='output_directory', type=str, default=os.path.join(os.getcwd(), 'output', 'gps_tle'),
                        help='Specify output subfolder')

    parser.add_argument('--satname', type=str, action='store', default='TEST-GPS',
                        help='Name of satellite in TLE file')

    parser.add_argument('--satnr', type=int, action='store', default=99999,
                        help='Satellite catalog number in TLE file')

    parser.add_argument('-b', type=float, action='store', default=0,
                        help='B star value of TLE, default 0')

    parser.add_argument('-v', action='store_true', default=False,
                        help='Verbose')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = argumentsParser()
    filename = args.f
    bstar = args.b
    output_subfolder = args.o
    satname = args.satname
    verbose_enable = args.v
    satnumber = args.satnr

    # Display warning
    print("this script is very experimental!")

    # Create subfolder
    os.makedirs(output_subfolder, exist_ok=True)

    # Open GPS data file
    gps_data = pd.read_table(filename, delimiter=',')

    t = np.array(gps_data['gps_epoch'])
    gps_valid = gps_data['gps_valid']

    n_col = np.arange(3)
    gps_pos = np.array([gps_data['gps_pos_' + str(
        n_col[x])] for x in n_col]).transpose()
    gps_vel = np.array([gps_data['gps_vel_' + str(n_col[x])]
                        for x in n_col]).transpose()

    # Remove invalid points
    idnon0 = np.where(t > 1483228800)  # remove all values before 1st JAN 2017
    idvalid = np.where(gps_valid == 1)
    idnonnan = np.where(~np.isnan(gps_valid))

    idselect = np.intersect1d(idnon0[0], idvalid[0])
    idselect = np.intersect1d(idselect, idnonnan[0])

    t = t[idselect]
    gps_pos = gps_pos[idselect, :]
    gps_vel = gps_vel[idselect, :]

    # Create time vectors
    t_datetime = [datetime.utcfromtimestamp(
        x).strftime('%Y-%m-%d %H:%M:%S') for x in t]
    t_vec = Time(list(t_datetime), scale='utc')

    gcrs_pos, gcrs_vel = la.convert_gps2eci(t_vec, gps_pos, gps_vel)

    line1_list, line2_list, meanElem, errEstR, errEstV = lt.RV2TLE(
        gcrs_pos, gcrs_vel, t, corrNum=2, satnumber=satnumber, bstar=bstar, verbose_enable=verbose_enable)

    # Add U classification
    str_line1 = line1_list[-1][-1]
    list_line1 = list(str_line1)
    list_line1[7] = 'U'
    str_line1 = ''.join(list_line1)

    str_line2 = line2_list[-1][-1]

    print(satname)
    print(str_line1)
    print(str_line2)

    date_str = datetime.utcfromtimestamp(t[0]).strftime('%Y_%m_%d')
    tle_file = satname + '_gps_' + date_str + '.txt'
    tle_path = os.path.join(os.getcwd(), output_subfolder, tle_file)

    with open(tle_path, 'w') as f:
        f.write(satname + '\r\n')
        f.write(str_line1)
        f.write('\r\n')
        f.write(str_line2)
