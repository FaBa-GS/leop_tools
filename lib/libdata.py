#!/usr/bin/python3.7
# Copyright (c) 2013-2020 GomSpace A/S. All rights reserved.

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.dates as dates
import lib.libastro as la


class BeaconData:
    # Class for reading beacon data file after extraction

    def __init__(self, filename):

        self.data = pd.read_table(filename, delimiter=',')

        # Data
        self.t = None
        self.mag = None
        self.extmag = None
        self.gyro = None
        self.gyro_temp = None
        self.extgyro = None
        self.fss_raw = None
        self.st_raw = None
        self.status_str = None
        self.st_valid = None
        self.ephem_suneci = None
        self.ephem_mageci = None

        self.wheel_speed = None
        self.wheel_cur = None

        self.ctrl_refq = None
        self.ctrl_errq = None
        self.ctrl_errrate = None
        self.ctrl_mwspeed = None
        self.ctrl_mwtorque = None
        self.ukf_q = None
        self.ukf_w = None
        self.ephem_reci = None
        self.ephem_veci = None

        self.ukf_X = None
        self.ukf_Z = None
        self.ukf_Zpred = None
        self.ukf_Pdiag = None
        self.ukf_Pzzdiag = None
        self.ukf_ineclipse = None

        # Times
        self.t_utc = None
        self.t_dates = None

        # Euler angles
        self.ctrl_err_euler = None

        self.load_alldata()

    def get_data(self, telem_str, telem_len, verbose=False):
        """ Get data from file

        Return the selected data, or None if not found in header
        """

        D = None
        if telem_len == 0 or telem_len == 1:
            # Single telemetry
            if telem_str in self.data.columns:
                D = np.array(self.data[telem_str])
            else:
                if verbose:
                    # Do not load the data if not found
                    print("{:s} not found.".format(telem_str))

        elif telem_len > 0:

            # Create variables: param-name_number
            n_col = list(range(telem_len))
            n_col = list(map(str, n_col))
            n_col = range(telem_len)
            telem_str_list = [telem_str + '_' + str(n_col[x]) for x in n_col]

            for item in telem_str_list[:]:
                if item in self.data.columns:
                    pass
                else:
                    telem_str_list.remove(item)
                    if verbose:
                        print("{:s} not found.".format(item))
            D = np.array([self.data[item] for item in telem_str_list])
            if D.shape[0] == 0:
                D = None
        else:
            raise Warning('Cannot put negative telemetry lengths')

        return D

    def load_alldata(self):

        # Load data (if exists in file)
        self.t = self.get_data('ts', 0)
        self.mag = self.get_data('mag', 3)
        self.extmag = self.get_data('extmag', 3)
        self.gyro = self.get_data('gyro', 3)
        self.gyro_temp = self.get_data('gyro_temp', 0)
        self.extgyro = self.get_data('extgyro', 3)
        self.fss_raw = self.get_data('fss_raw', 6 * 4)
        self.st_raw = self.get_data('st_raw', 5)
        self.status_str = self.get_data('status_str', 0)
        self.st_valid = self.get_data('st_valid', 0)
        self.ephem_suneci = self.get_data('ephem_suneci', 3)
        self.ephem_mageci = self.get_data('ephem_mageci', 3)

        self.wheel_speed = self.get_data('wheel_speed', 4)
        self.wheel_cur = self.get_data('wheel_cur', 4)

        self.ctrl_refq = self.get_data('ctrl_refq', 4)
        self.ctrl_errq = self.get_data('ctrl_errq', 4)
        self.ctrl_errrate = self.get_data('ctrl_errrate', 3)
        self.ctrl_mwspeed = self.get_data('ctrl_mwspeed', 4)
        self.ctrl_mwtorque = self.get_data('ctrl_mwtorque', 4)
        self.ukf_q = self.get_data('ukf_q', 4)
        self.ukf_w = self.get_data('ukf_w', 3)
        self.ephem_reci = self.get_data('ephem_reci', 3)
        self.ephem_veci = self.get_data('ephem_veci', 3)

        self.ukf_X = self.get_data('ukf_X', 7)
        self.ukf_Z = self.get_data('ukf_Z', 11)
        self.ukf_Zpred = self.get_data('ukf_Zpred', 11)
        self.ukf_Pdiag = self.get_data('ukf_Pdiag', 6)
        self.ukf_Pzzdiag = self.get_data('ukf_Pzzdiag', 11)
        self.ukf_ineclipse = self.get_data('ukf_ineclipse', 0)

        # Times
        self.t_utc = [datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in self.t]
        self.t_dates = dates.datestr2num(self.t_utc)

        # Euler angles
        self.ctrl_err_euler = self.q2euler(q=self.ctrl_errq)

    def select_range(self, t0, t1):
        # Find data range for selection [UNIX]
        ind0 = np.nanargmax(self.t > t0)
        ind1 = np.nanargmax(self.t > t1)

        return ind0, ind1

    def q2euler(self, q=None):
        # Convert quaternion to Euler [deg]
        euler = np.zeros([3, q.shape[1]])
        if q is not None:
            for i in range(q.shape[1]):
                euler[:, i] = la.q2euler(q[:, i])

        return np.rad2deg(euler)