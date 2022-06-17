#!/usr/bin/python3.7

import argparse
import os
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path
from astropy.time import Time
from lib import libcal as lc
from lib import libastro as la
from lib import libconfig as lconf
from lib import logger as log

FFS_NUM = 6

def argumentsParser():
    parser = argparse.ArgumentParser(
        description='oo_autocalib: Calibration script')
    parser.add_argument('-f', metavar='file', type=str, default=os.path.join(os.getcwd(), 'test_data', 'test.csv'),
                        help='Specify the datafile path')

    parser.add_argument('-o', metavar='output_directory', type=str, default=os.path.join(os.getcwd(), 'output/test'),
                        help='Specify output subfolder')

    parser.add_argument('-c', metavar='config_file', type=str, default=os.path.join(os.getcwd(), 'example_param'),
                        help='Specify config_file subfolder')

    parser.add_argument('--overwrite-data', action='store_true', default=False,
                        help='Overwrite ephemeris data with Astropy data using TLEs')

    parser.add_argument('--calibrate-fss', action='store_true', default=False,
                        help='Overwrite ephemeris data with Astropy data using TLEs')

    parser.add_argument('--plot-time', action='store_true', default=False,
                        help='Plot time over x-axis')

    parser.add_argument('--tle-file', type=str, action='store', default=os.path.join(os.getcwd(), 'test_data', 'tle_test.txt'),
                        help='TLE file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = argumentsParser()
    filename = args.f
    config_subfolder = args.c
    overwrite_data = args.overwrite_data
    tle_file = args.tle_file
    output_subfolder = args.o
    enable_plot_time = args.plot_time
    skip_fss = not args.calibrate_fss

    # Initiate Logging
    cal_log = log.init_logging("CAL", Path('logging.ini'))
    cal_log.info(
        '---------------------------------------------------------------------')
    cal_log.notice('Initiating sensor calibration')

    # Load CONFIG files
    adcs_config = lconf.AdcsConfig(
        adcs_node=4, config_subfolder=config_subfolder)

    # INTMAG table
    adcs_config.load_table(lconf.INTMAG_TABLE)
    cal_log.info('Table {:d} loaded.'.format(lconf.INTMAG_TABLE))

    # EXTMAB table
    adcs_config.load_table(lconf.EXTMAG_TABLE)
    cal_log.info('Table {:d} loaded.'.format(lconf.EXTMAG_TABLE))

    # SENSOR_COMMON table
    adcs_config.load_table(lconf.SENSOR_COMMON_TABLE)
    cal_log.info('Table {:d} loaded.'.format(lconf.SENSOR_COMMON_TABLE))

    # FSS_TABLE table
    for l in range(len(lconf.FSS_TABLE)):
        adcs_config.load_table(lconf.FSS_TABLE[l])
        cal_log.info('Table {:d} loaded.'.format(lconf.FSS_TABLE[l]))

    # Assign calibration config (don't load parameters)
    cal_config = lconf.AdcsConfig(
        adcs_node=4, config_subfolder=config_subfolder)

    # Create output subfolder if not exist
    if not os.path.exists(output_subfolder):
        os.mkdir(output_subfolder)

    # Read csv datafile
    data = pd.read_table(filename, delimiter=',')

    t = np.array(data['ts'])
    mag = np.array([data['mag_0'], data['mag_1'], data['mag_2']]).transpose()
    extmag = np.array([data['extmag_0'], data['extmag_1'],
                      data['extmag_2']]).transpose()

    n_col = np.arange(int(FFS_NUM*4))

    fss_raw = np.array([data['fss_raw_' + str(n_col[x])]
                       for x in n_col]).transpose()

    if overwrite_data:
        # TODO: Use SGP4 from a list of TLEs

        # Load TLEs
        cal_log.warning(
            'Overwriting IGRF data using data obtained through TLEs')
        with open(tle_file) as f:
            lines = f.read().splitlines()
            sat_name = lines[0]
            tleline1 = lines[1]
            tleline2 = lines[2]

            cal_log.info(sat_name)
            cal_log.info(tleline1)
            cal_log.info(tleline2)

        # Convert Unix to datetime
        t_datetime = [datetime.utcfromtimestamp(
            x).strftime('%Y-%m-%d %H:%M:%S') for x in t]
        t_date = Time(t_datetime, scale='utc')

        data = la.sgp4_tle(tle_file, t, ITRF=True)

        # Compute IGRF magnetic field
        igrf = la.compute_igrf(
            t_date, data['lat'], data['lon'], data['alt']).transpose()/100  # [mG]
        
        # TODO: add sun ephemeris
        cal_log.warning("Does not generate sun epehemeris yet!")
    else:
        igrf = np.array([data['ephem_mageci_0'], data['ephem_mageci_1'],
                        data['ephem_mageci_2']]).transpose()/100  # to mG

        sunref = np.array([data['ephem_suneci_0'], data['ephem_suneci_1'],
                        data['ephem_suneci_2']]).transpose()  # [m]

    # Calculate Norm
    igrf_n = np.sqrt(igrf[:, 0]**2 + igrf[:, 1]**2 + igrf[:, 2]**2)
    mag_n = np.sqrt(mag[:, 0]**2 + mag[:, 1]**2 + mag[:, 2]**2)
    extmag_n = np.sqrt(extmag[:, 0]**2 + extmag[:, 1]**2 + extmag[:, 2]**2)

    # Filter NaNs
    idrm1 = np.where(np.isnan(mag_n))
    idrm2 = np.where(np.isnan(extmag_n))
    idrm3 = np.where(np.isnan(igrf_n))
    idrm4 = np.where(np.isnan(fss_raw[:,0]))
    idrm5 = np.where(np.isnan(sunref[:,0]))

    idrm = np.unique(np.concatenate([idrm1[0], idrm2[0], idrm3[0],idrm4[0]]))

    mag = np.delete(mag, idrm, axis=0)
    mag_n = np.delete(mag_n, idrm)
    extmag = np.delete(extmag, idrm, axis=0)
    extmag_n = np.delete(extmag_n, idrm)

    igrf = np.delete(igrf, idrm, axis=0)
    igrf_n = np.delete(igrf_n, idrm)
    fss_raw = np.delete(fss_raw, idrm, axis=0)
    sunref = np.delete(sunref, idrm, axis=0)
    t = np.delete(t, idrm)
    
    # Filter reference norm of zero
    idrm1 = np.where(igrf_n == 0)
    idrm2 = np.where(extmag_n == 0)
    idrm3 = np.where(mag_n == 0)

    idrm = np.unique(np.concatenate([idrm1[0], idrm2[0], idrm3[0]]))

    mag = np.delete(mag, idrm, axis=0)
    mag_n = np.delete(mag_n, idrm)
    extmag = np.delete(extmag, idrm, axis=0)
    extmag_n = np.delete(extmag_n, idrm)
    igrf = np.delete(igrf, idrm, axis=0)
    igrf_n = np.delete(igrf_n, idrm)
    fss_raw = np.delete(fss_raw, idrm, axis=0)
    sunref = np.delete(sunref, idrm, axis=0)
    t = np.delete(t, idrm)

    if enable_plot_time:
        t_dates = t
    else:
        t_dates = None

    ###################################
    # CALIBRATE INTMAG
    ###################################

    cal_log.notice('INTMAG - PRE-CALIBRATION')

    mag_offset = adcs_config.intmag.mag_offset
    mag_scale = adcs_config.intmag.mag_scale
    mag_rotation = adcs_config.intmag.mag_rotation

    adcs_config.print_intmag(log=cal_log)

    # PLOT 3D sphere
    plot_name = os.path.join(output_subfolder, 'intmag_unitsphere.png')
    lc.plot_sphere(mag[:, 0], mag[:, 1], mag[:, 2],
                   save_plot=True, savepath=plot_name)

    # Calibrate INTMAG
    Cf, xcf, ycf, zcf, af = lc.mag_cal(
        mag, igrf, mag_offset, mag_scale, mag_rotation)

    # Set and print calibration parameters
    cal_log.notice('INTMAG - POST-CALIBRATION')
    cal_config.set_intmag(
        mag_offset=Cf.x[0:3], mag_scale=Cf.x[3:6], mag_rotation=af)
    cal_config.print_intmag(log=cal_log)

    file_name = os.path.join(output_subfolder, 'rparam_intmag.txt')
    cal_config.print_intmag(rparam=True, f=file_name)

    # Display results
    cal_mag_n = np.sqrt(xcf**2 + ycf**2 + zcf**2)
    err1 = (mag_n - igrf_n)
    err2 = (cal_mag_n - igrf_n)

    # Norm Comparison
    plot_name = os.path.join(output_subfolder, 'intmag_igrf_norm.png')
    lc.plot_norm(mag_n, cal_mag_n, igrf_n, t_date=t_dates,save_plot=True, savepath=plot_name)

    # Error between norms
    plot_name = os.path.join(output_subfolder, 'intmag_igrf_error.png')
    lc.plot_err_hist(err1, err2, save_plot=True, savepath=plot_name)

    ###################################
    # CALIBRATE EXTMAG
    ###################################

    cal_log.notice('EXTMAG - PRE-CALIBRATION')

    extmag_offset = adcs_config.extmag.offset
    extmag_scale = adcs_config.extmag.scale
    extmag_rotation = adcs_config.extmag.rotation

    adcs_config.print_extmag(log=cal_log)

    # PLOT 3D sphere
    plot_name = os.path.join(output_subfolder, 'extmag_unitsphere.png')
    lc.plot_sphere(extmag[:, 0], extmag[:, 1], extmag[:, 2],
                   save_plot=True, savepath=plot_name)

    # Calibrate INTMAG
    Cf, xcf, ycf, zcf, af = lc.mag_cal(
        extmag, igrf, extmag_offset, extmag_scale, extmag_rotation)

    # Set and print calibration parameters
    cal_log.notice('EXTMAG - POST-CALIBRATION')
    cal_config.set_extmag(offset=Cf.x[0:3], scale=Cf.x[3:6], rotation=af)
    cal_config.print_extmag(log=cal_log)

    file_name = os.path.join(output_subfolder, 'rparam_extmag.txt')
    cal_config.print_extmag(rparam=True, f=file_name)

    # Display results
    cal_extmag_n = np.sqrt(xcf**2 + ycf**2 + zcf**2)
    err1 = (extmag_n - igrf_n)
    err2 = (cal_extmag_n - igrf_n)

    # Norm Comparison
    plot_name = os.path.join(output_subfolder, 'extmag_igrf_norm.png')
    lc.plot_norm(extmag_n, cal_extmag_n, igrf_n,
                 t_date=t_dates,save_plot=True, savepath=plot_name)

    # Error between norms
    plot_name = os.path.join(output_subfolder, 'extmag_igrf_error.png')
    lc.plot_err_hist(err1, err2, save_plot=True, savepath=plot_name)

    if not skip_fss:
        ###################################
        # CALIBRATE FSS
        ###################################

        cal_log.notice('FSS - PRE-CALIBRATION')

        fss_num = adcs_config.sensor_common.fss_num
        fss_darkth = adcs_config.sensor_common.fss_darkth
        fss_idarkth = adcs_config.sensor_common.fss_idarkth
        fss_brightth = adcs_config.sensor_common.fss_brightth
        
        adcs_config.print_sensor_common(log=cal_log)

        # Calibrate FSS
        for j in range(fss_num):

            adcs_config.print_fss(nr_fss=j,log=cal_log)

            fss_q = adcs_config.fss[j].q
            fss_p = adcs_config.fss[j].p

            fss_raw_s = fss_raw[:,4*j:(4*j+4)].transpose()
            idrm = lc.filter_raw(fss_raw_s, fss_darkth, fss_brightth, fss_idarkth)
            
            if (np.array(idrm)).size != 0:
                fss_raw_s = np.delete(fss_raw_s, idrm, axis=1)
                igrf_s = np.delete(igrf, idrm, axis=0)
                sunref_s = np.delete(sunref, idrm, axis=0)
                extmag_s = np.delete(extmag, idrm, axis=0)
            else:
                igrf_s = igrf
                sunref_s = sunref
                extmag_s = extmag

            if fss_raw_s.size != 0:
                x = np.concatenate([fss_p,fss_q])
                pre_err = lc.fss_model(x, fss_raw_s, sunref_s, extmag_s, igrf_s, opt_mode=False)

                x_opt = lc.fss_cal(fss_p, fss_q, fss_raw_s, sunref_s, extmag_s, igrf_s)

                post_err = lc.fss_model(x_opt.x, fss_raw_s, sunref_s, extmag_s, igrf_s, opt_mode=False)

                # Error between norms
                plot_name = os.path.join(output_subfolder, 'fss{:d}_error.png'.format(j))
                lc.plot_err_hist(np.rad2deg(pre_err), np.rad2deg(post_err), save_plot=True, savepath=plot_name,unit='deg')
            else:
                cal_log.notice('Skipping fss {:d}, not enough valid data.'.format(j))