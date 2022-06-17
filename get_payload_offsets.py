#!/usr/bin/python3

from cgitb import enable
import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from lib import libcal as lc
from lib import logger as log

max_estimate_window = 10

""" Estimate magnetometer offset values induced by payload
Assumptions:
- The magnetometer offset due to payload activation is constant.
- The data set is approximated as linear near payload activation and
  deactivation.

Method:
Assume dataset before, during, and after payload activation/deactivation
as linear lines; y = ax+b.
The values for a and b are determined using least-square fit. The slope
while the payload is ON is assumed to have either the same slope as
before or after payload activation (defined as offset1 and offset2).
The offset can then be determined as the difference between the values b:

offset1 = b_pre-b_target
offset2 = b_post-b_target
"""


def argumentsParser():
    parser = argparse.ArgumentParser(
        description='get_payload_offset.py: Determine offsets due to payloads/components from retrieved data')

    parser.add_argument('-f', type=str, default=os.path.join(os.getcwd(),'test_data','test_mo_data.csv'),
                        help='Specify the path to file of magnetometer offset file')

    parser.add_argument('--mode', type=str, default='xband',
                        help='Specify the mode from JSON file e.g. {sband,xband}')

    parser.add_argument('-o', type=str, default='test',
                        help='Output subfolder name')

    parser.add_argument('--submode', type=str, default=None,
                        help='Specify the specific ops of the selected mode')

    parser.add_argument('-s', type=str, default=os.path.join(os.getcwd(),'test_data', 'example_offset_test.json'),
                        help='JSON file containing settings')

    parser.add_argument('--intmag', action='store_true', default=False,
                        help='Use internal magnetometer instead of external')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # arguments
    args = argumentsParser()
    filename = args.f
    output_filename = args.o
    setting_file = args.s
    pay_mode = args.mode
    pay_submode = args.submode
    enable_intmag = args.intmag

    # Selection Mag
    if enable_intmag:
        mag_in_use_str = 'internal - A3200'
        data_str = 'mag'
    else:
        mag_in_use_str = 'external - M315'
        data_str = 'extmag'

    # Output folder
    path_output = os.path.join(os.getcwd(), 'output', 'mag_offset')
    path_output_sub = os.path.join(path_output,output_filename)
    
    # Create output subfolder if not exist
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    if not os.path.exists(path_output_sub):
        os.mkdir(path_output_sub)

    # Read Settings
    with open(setting_file) as sf:
        settings = json.load(sf)

    # Initiate Logging
    mlog = log.init_logging("MAG-OFFSET", Path('logging.ini'))
    mlog.info(
        '---------------------------------------------------------------------')
    mlog.notice('Determining Magnetometer Offsets')
    mlog.info('- file     : {:s}'.format(filename))
    mlog.info('- settings : {:s}'.format(setting_file))
    mlog.info('- mode     : {:s}'.format(pay_mode))  
    mlog.info('- mag      : {:s}'.format(mag_in_use_str))
    if pay_submode:
        mlog.info('- submode  : {:s}'.format(pay_submode))
    else:
        mlog.info('- submode  : all')

    if pay_mode not in settings:
        mlog.error(
            '{:s} mode does not exist in JSON settings file.'.format(pay_mode))
        sys.exit()

    # Loop through all if not explictly given a submode
    pay_submode_list = []
    dt_list = []
    t_on_list = []
    t_off_list = []

    for key in settings[pay_mode]:
        pay_submode_list.append(key['name'])
        dt_list.append(key['dt'])
        t_on_list.append(key['t_on'])
        t_off_list.append(key['t_off'])

    # if pay_submode is (not) specified
    if pay_submode is not None:
        subkey_exists = False

        for subkey in settings[pay_mode]:
            if subkey['name'] == pay_submode:
                subkey_exists = True

        if not subkey_exists:
            mlog.error(
                '{:s} submode does not exist in JSON settings file.'.format(pay_submode))
            sys.exit()
        else:
            ind_submode = pay_submode.index(pay_submode)

            # Overwrite list to chosen submode values
            pay_submode_list = list([pay_submode_list[ind_submode]])
            dt_list = list([dt_list[ind_submode]])
            t_on_list = list([t_on_list[ind_submode]])
            t_off_list = list([t_off_list[ind_submode]])

    # Load base magnetometer value
    if 'base' not in settings:
        mlog.error('Base setting does not exist in JSON settings file.')
        sys.exit()

    offset = np.array(settings['base'][0]['offset'])
    scale = np.array(settings['base'][0]['scale'])
    rot = np.array(settings['base'][0]['rotation'])
    rot_m = np.array([rot[0:3], rot[3:6], rot[6:9]])  # in matrix format

    mlog.notice('Magnetometer base values')
    mlog.info('- offset: {:.4f} {:.4f} {:.4f}'.format(*offset))
    mlog.info('- scale: {:.4f} {:.4f} {:.4f}'.format(*scale))
    mlog.info(
        '- rotation: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(*rot))

    # Read csv datafile
    data = pd.read_table(filename, delimiter=',')

    t = np.array(data['ts'])
    
    n_col = range(3)
    mag = np.array([data[data_str + '_' + str(n_col[x])] for x in n_col]).transpose()

    # Uncalibrate to RAW values
    uncal_magdata = np.zeros([len(t), 3])
    for i in range(len(mag[:, 0])):
        uncal_magdata[i, :] = lc.uncalibrate_mag(
            mag[i, :], np.array(offset), np.array(scale), rot_m)

    # Transpose array
    uncal_magdata = uncal_magdata.transpose()

    # Looping through all submodes
    for j in range(len(pay_submode_list)):

        """ Info on dt:
        - dt[0]:
        # of points for LSQ linear fit (during) (use 3 for high offsets, 5 or 10 for lower offsets)
        - dt[1]:
        - dt[2]:
        #  number of points for LSQ linear fit before or after device powered
        """
        current_submode = pay_submode_list[j]
        dt1 = dt_list[j][0]
        dt2 = dt_list[j][1]
        dt3 = dt_list[j][2]
        t_on = t_on_list[j]
        t_off = t_off_list[j]

        # Create subfolder per submode
        path_output_sub2 = os.path.join(path_output_sub,current_submode)
        
        # Create output subfolder if not exist
        if not os.path.exists(path_output_sub2):
            os.mkdir(path_output_sub2)

        # Skip if t_on or t_off is zero
        if (t_on == 0) or (t_off == 0):
            continue

        mlog.notice('Magnetometer offset - {:s}'.format(current_submode))

        # Check if offset times are larger than configured maximum
        # TODO: remove? it is outdated and not longer (properly) used in remainder code
        delta_offset = abs(t_off - t_on)
        if (delta_offset <= max_estimate_window):
            t_close = True
        else:
            t_close = False

        # Start time for when offset is estimated (to allow for current to settle)
        ind_t0 = np.where(t == t_on)[0] - dt1
        ind_t1 = np.where(t == t_off)[0] + dt1  # End time " "

        # Start time for when offset occurs
        ind_offset0 = np.where(t == t_on)[0]
        ind_offset1 = np.where(t == t_off)[0]-1   # End time " "

        """ Determine ranges
        note: ranges should not be too long (becomes nonlinear)
        approx. 10 seconds.
        - range0a is the period before the "power on" timepoint
        - range1b is the period after the "power off" timepoint
        - range0b is the period after the "power on" timepoint
        - range1a is the period before the "power off" timepoint
        - range_mid is the area where the payload is turned on (for plotting new offset).
        """

        range0a = np.arange(ind_t0-dt2, ind_t0+1)
        range1b = np.arange(ind_t1, ind_t1+dt2+1)

        if t_close:
            range0b = np.arange(ind_offset0, ind_offset1+1)
            range1a = np.arange(ind_offset0, ind_offset1+1)
        else:
            range0b = np.arange(ind_offset0, ind_offset0+dt3+1)
            range1a = np.arange(ind_offset1-dt3, ind_offset1+1)

        range_mid = np.arange(ind_t0, ind_t1+1)

        """ Estimate based on LSQ fit per axis
        Fit the mag readings as a linear line (LSQ) near the PAYLOAD offset:
        y = ax + b.
        Estimate b in the "offset zone". Take the same slope a in this zone using
        before or after data. Determine offset by subtracting the b value in this
        zone to the b value before or after.
        objective:
        minimize sum squared error with error: eps = y_i -a*x_i -b
        minimization problem solved through derivative sum(eps^2)=0
        note:
        MATLAB's inbuilt function polyfit/polyval can be used as well
        """

        # Number of samples
        N0a = len(range0a)
        N0b = len(range0b)
        N1a = len(range1a)
        N1b = len(range1b)

        # Vector for linear fit
        # 0a : based on pre PAYLOAD ON time
        # 0b : based on post PAYLOAD ON time
        # 1a : based on pre PAYLOAD OFF time
        # 1b : based on post PAYLOAD OFF time

        x0a = t[range0a] - t[0]
        x0b = t[range0b] - t[0]

        x1a = t[range1a] - t[0]
        x1b = t[range1b] - t[0]

        # Offsets
        b0a = np.zeros(3)
        b0b = np.zeros(3)
        b1a = np.zeros(3)
        b1b = np.zeros(3)

        # Y
        y0a = uncal_magdata[:, range0a]
        y0b = uncal_magdata[:, range0b]
        y1a = uncal_magdata[:, range1a]
        y1b = uncal_magdata[:, range1b]

        y_lsq0a = np.zeros([3, len(x0a)])
        y_lsq1b = np.zeros([3, len(x1b)])
        y_est1 = np.zeros([3, len(x0b)])
        y_est2 = np.zeros([3, len(x1a)])

        # Loop from X to Z
        for i in range(3):

            coef0a = np.zeros(2)
            coef1b = np.zeros(2)

            # LSQ fit - before PAYLOAD POWER ON
            coef0a[0], coef0a[1] = lc.lsq_reg(x0a, y0a[i, :])
            y_lsq0a[i, :] = coef0a[0]*x0a + coef0a[1]

            # LSQ fit - after PAYLOAD POWER OFF
            coef1b[0], coef1b[1] = lc.lsq_reg(x1b, y1b[i, :])
            y_lsq1b[i, :] = coef1b[0]*x1b + coef1b[1]

            # Determine b based on LSQ fit with same slope as before/after target
            b0b[i] = sum(y0b[i, :]-coef0a[0]*x0b)/N0b
            b1a[i] = sum(y1a[i, :]-coef1b[0]*x1a)/N1a

            # b value before and after
            b0a[i] = coef0a[1]
            b1b[i] = coef1b[1]

            # Estimated linear functions
            y_est1[i, :] = coef0a[0]*x0b+b0b[i]
            y_est2[i, :] = coef1b[0]*x1a+b1a[i]

        # Print offset values
        offset1 = b0a - b0b
        offset2 = b1b - b1a

        # diplay
        mlog.info("- offset1: {:.6f} {:.6f} {:.6f}".format(*offset1))
        mlog.info("- offset2: {:.6f} {:.6f} {:.6f}".format(*offset2))

        settings[pay_mode][j]['offset1'] = list(offset1)
        settings[pay_mode][j]['offset2'] = list(offset2)

        # TODO: add LSQ residual value -> best offset

        # Convert to datetime
        t_title = datetime.utcfromtimestamp(t[0]).strftime('%Y-%m-%d')
        t_file = datetime.utcfromtimestamp(t[0]).strftime('%Y%m%d_%H%M%S')
        t_datetime = [datetime.utcfromtimestamp(
            x).strftime('%Y-%m-%d %H:%M:%S') for x in t]
        t_dates = dates.datestr2num(t_datetime)  # For plot x-axis

        # Plot Figure
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(19, 26))
        str_title = 'Mag. offset estimation - ' + current_submode + ' - ' + t_title
        
        fig.suptitle(str_title, fontsize=16)
        for i in range(3):
            
            # offset estimate1
            plot_nr = i
            x = t_dates[range0a]
            x2 = t_dates[range0b]
            y = y_lsq0a[i, :]
            y2 = y_est1[i, :]

            x3 = t_dates[range0a]
            y3 = uncal_magdata[i, range0a]
            x4 = t_dates[range0b]
            y4 = uncal_magdata[i, range0b]

            if i == 0:
                ax[plot_nr, 0].set_title('Offset estimate1')
            ax[plot_nr, 0].grid()
            ax[plot_nr, 0].plot_date(x, y, linestyle='solid', ms=0)
            ax[plot_nr, 0].plot_date(x2, y2, linestyle='solid', ms=0)
            ax[plot_nr, 0].plot_date(x3, y3, color='k', ms=2)
            ax[plot_nr, 0].plot_date(x4, y4, color='k', ms=2)
            ax[plot_nr, 0].legend(['before', 'offset1'])
            ax[plot_nr, 0].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))
            ax[plot_nr, 0].minorticks_on()
            c1 = ax[plot_nr, 0].get_ylim()
            t_date1 = t_dates[ind_offset0]
            ax[plot_nr, 0].plot_date(
                [t_date1, t_date1], c1, linestyle='--', ms=0)

            # offset estimate1
            # plot_nr = i + 3
            x = t_dates[range1a]
            x2 = t_dates[range1b]
            y = y_est2[i, :]
            y2 = y_lsq1b[i, :]

            x3 = t_dates[range1a]
            y3 = uncal_magdata[i, range1a]
            x4 = t_dates[range1b]
            y4 = uncal_magdata[i, range1b]

            if i == 0:
                ax[plot_nr, 1].set_title('Offset estimate2')
            ax[plot_nr, 1].grid()
            ax[plot_nr, 1].plot_date(x2, y2, linestyle='solid', ms=0)
            ax[plot_nr, 1].plot_date(x, y, linestyle='solid', ms=0)
            ax[plot_nr, 1].plot_date(x3, y3, color='k', ms=2)
            ax[plot_nr, 1].plot_date(x4, y4, color='k', ms=2)
            ax[plot_nr, 1].legend(['after', 'offset2'])
            ax[plot_nr, 1].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))
            ax[plot_nr, 1].minorticks_on()
            c1 = ax[plot_nr, 1].get_ylim()
            t_date1 = t_dates[ind_offset1]
            ax[plot_nr, 1].plot_date([t_date1, t_date1], c1, 
                                      linestyle='--', 
                                      ms=0, color=[0.5, 0.5, 0.5])
            
        str_plot_filename = 'mag_offset_estimation_' + current_submode + '_' + t_file  + '.png'
        path_plot_file = os.path.join(path_output_sub2,str_plot_filename)
        fig.savefig(path_plot_file)
    
        # Apply new mag offsets
        # No offset
        magdata_no = mag
        magdata_n_no = np.sqrt(magdata_no[:, 0]**2 +
                            magdata_no[:, 1]**2 + magdata_no[:, 2]**2)

        # Apply - offset1
        magdata_cal = np.zeros(uncal_magdata[:, range_mid].shape)
        for i in range(3):
            magdata_cal[i, :] = uncal_magdata[i, range_mid] + offset1[i]

        # Apply - offset2
        magdata_cal2 = np.zeros(uncal_magdata[:, range_mid].shape)
        for i in range(3):
            magdata_cal2[i, :] = uncal_magdata[i, range_mid] + offset2[i]

        # Apply calibration parameters
        magdata1_postcal = np.zeros([3, len(range_mid), ])
        magdata2_postcal = np.zeros([3, len(range_mid), ])

        for i in range(len(range_mid)):
            B1 = magdata_cal[:, i]
            B2 = magdata_cal2[:, i]
            magdata1_postcal[:, i] = lc.calibrate_mag(
                B1, np.array(offset), np.array(scale), rot_m)
            magdata2_postcal[:, i] = lc.calibrate_mag(
                B2, np.array(offset), np.array(scale), rot_m)

        # NORM
        mag_cal_n1 = np.sqrt(
            magdata1_postcal[0, :]**2 + magdata1_postcal[1, :]**2 + magdata1_postcal[2, :]**2)
        mag_cal_n2 = np.sqrt(
            magdata2_postcal[0, :]**2 + magdata2_postcal[1, :]**2 + magdata2_postcal[2, :]**2)

        # Plot Figure - norm with offset
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 19))
        str_title = 'Applied mag. offset norm - ' + current_submode + ' - ' + t_title
        fig.suptitle(str_title, fontsize=16)

        x = t_dates
        y = magdata_n_no
        x2 = t_dates[range_mid]
        y2 = mag_cal_n1
        x3 = t_dates[range_mid]
        y3 = mag_cal_n2

        if i == 0:
            ax.set_title('Offset estimate1')
        ax.grid()
        ax.plot_date(x, y, linestyle='solid', ms=1)
        ax.plot_date(x2, y2, linestyle='solid', ms=1)
        ax.plot_date(x3, y3, linestyle='solid', ms=1)
        ax.set_ylabel('|B| [mG]')
        ax.legend(['no offset', 'offset1', 'offset2'])
        ax.axvline(t_dates[ind_offset0])
        ax.axvline(t_dates[ind_offset1])
        ax.xaxis.set_major_formatter(
            dates.DateFormatter('%H:%M:%S'))
        ax.minorticks_on()

        str_plot_filename = 'apply_offset_norm_' + current_submode + '_' + t_file  + '.png'
        path_plot_file = os.path.join(path_output_sub2,str_plot_filename)
        fig.savefig(path_plot_file)

    
        # Plot per axis
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(19, 26))
        str_title = 'Applied mag. offset axis - ' + current_submode + ' - ' + t_title
        fig.suptitle(str_title, fontsize=16)

        for i in range(3):

            # offset estimate1
            plot_nr = i
            x = t_dates
            y = magdata_no[:, i]
            x2 = t_dates[range_mid]
            y2 = magdata1_postcal[i, :]
            x3 = t_dates[range_mid]
            y3 = magdata2_postcal[i, :]

            if i == 0:
                ax[plot_nr].set_title('Offset estimate1')
            ax[plot_nr].grid()
            ax[plot_nr].plot_date(x, y, linestyle='solid', ms=1)
            ax[plot_nr].plot_date(x2, y2, linestyle='solid', ms=1)
            ax[plot_nr].plot_date(x3, y3, linestyle='solid', ms=1)
            ax[plot_nr].legend(['no offset', 'offset1', 'offset2'])
            ax[plot_nr].axvline(t_dates[ind_offset0])
            ax[plot_nr].axvline(t_dates[ind_offset1])
            ax[plot_nr].xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))
            ax[plot_nr].minorticks_on()

        str_plot_filename = 'apply_offset_axis_' + current_submode + '_' + t_file  + '.png'
        path_plot_file = os.path.join(path_output_sub2,str_plot_filename)
        fig.savefig(path_plot_file)

    # Store new Json file
    json_output_file = 'output_settings' + '_' + t_file + '.json'
    json_output = os.path.join(path_output_sub,json_output_file)
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)