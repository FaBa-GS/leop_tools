#!/usr/bin/env python3
# Copyright (c) 2013-2021 GomSpace A/S. All rights reserved.
# Capgen - a tool to create capture conf's and flightplans for kleos
import argparse
import logging
import sys
import json
import time
from datetime import datetime
from datetime import date, datetime, time
from backports.datetime_fromisoformat import MonkeyPatch
MonkeyPatch.patch_fromisoformat()
#date.fromisoformat('2019-12-04')
import os
import tarfile
import shutil
import numpy as np

templates=os.getcwd() + '/templates'

class Capgen:

    SATIDS = {"A": 1, "B": 2, "C": 3, "D": 4}
    FREQS = {"A": 8315, "B": 8155, "C": 8235, "D": 8075}
    DATARATES = {"low": 12500, "medium": 25000, "high": 50000}

    def __init__(self, ctime, clength, dtime, dlength, sat, speed, dl_streamid, cap_streamid):
        #self.name = name
        self.dl_streamid = dl_streamid
        self.cap_streamid = cap_streamid

        if ctime:
            self.ctime = int(ctime) - 10  # -10 seconds for timestamps problem
            self.clength = clength + 10  # +10 seconds for timestamps problem
        else:
            self.ctime = None

        if dtime:
            self.dtime = int(dtime)
            self.dlength = dlength
        else:
            self.dtime = None

        # External magnetometer offset values
        self.extmag_origin  = np.array([0,0,0])
        self.extmag_offset0 = np.array([0,0,0])
        self.extmag_offset1 = np.array([0,0,0])

    def write_capture_conf(self, args):
        capture_conf = os.path.join(os.getcwd(), 'staging', 'captureconf.json')
        with open(os.path.join(templates, 'capture_tmpl.json')) as tf:
            template = json.load(tf)
            template['capture']['sat-id'] = self.get_sat_id(args.satellite)
            template['capture']['utc-start'] = self.ctime
            template['capture']['utc-stop'] = self.ctime + self.clength
            template['capture']['data-splitter']['stream-id'] = self.cap_streamid
            # baseband configs
            template['capture']['data-reader']['readers'][0]['channel-settings']['nco-freq-hz'] = args.chan0_nco_freq
            template['capture']['data-reader']['readers'][0]['channel-settings']['fixed-decimation-enabled'] = args.chan0_fixed_deci_enable
            template['capture']['data-reader']['readers'][0]['channel-settings']['fir-filter']['dynamic-decimation'] = args.chan0_dynamic_deci
            template['capture']['data-reader']['readers'][1]['channel-settings']['nco-freq-hz'] = args.chan1_nco_freq
            template['capture']['data-reader']['readers'][1]['channel-settings']['fixed-decimation-enabled'] = args.chan1_fixed_deci_enable
            template['capture']['data-reader']['readers'][1]['channel-settings']['fir-filter']['dynamic-decimation'] = args.chan1_dynamic_deci

            template['capture']['data-reader']['timestamp']['timestamp-enabled-chan0'] = args.chan0_timestamp_enable
            template['capture']['data-reader']['timestamp']['timestamp-interval-chan0'] = args.chan0_timestamp_interval
            template['capture']['data-reader']['timestamp']['timestamp-enabled-chan1'] = args.chan1_timestamp_enable
            template['capture']['data-reader']['timestamp']['timestamp-interval-chan1'] = args.chan1_timestamp_interval

            template['capture']['ad9361']['rx']['lo'] = args.ad9361_lo
            # maybe gain settings at some point

            with open(capture_conf, "w") as cf:
                json.dump(template, cf, indent=4)
        print("Capture config written to " + capture_conf)

    def get_sat_xband_freq(self, satellite):
        return self.FREQS[satellite] * 1000

    def get_datarate(self, speed):
        return self.DATARATES[speed]

    def get_sat_id(self, satellite):
        return self.SATIDS[satellite]

    def write_downlink_conf(self, satellite, speed):
        downlink_conf = os.path.join(os.getcwd(), 'staging', 'downlinkconf.json')
        with open(os.path.join(templates, 'downlink_tmpl.json')) as tf:
            template = json.load(tf)
            template['sat-id'] = self.get_sat_id(satellite)
            template['utc-start'] = self.dtime
            template['downlink']['x-band-transmitter']['carrier-freq'] = self.get_sat_xband_freq(satellite)
            template['downlink']['x-band-transmitter']['data-rate'] = self.get_datarate(speed)
            template['downlink']['serializer']['data-rate'] = self.get_datarate(speed)
            for stream in self.dl_streamid:
                dl = {}
                dl["stream-id"] = stream
                dl["chunks"] = [[0, -1]]
                template['downlink']['transmit'].append(dl)

            with open(downlink_conf, "w") as cf:
                json.dump(template, cf, indent=4)
        print("Downlink config written to " + downlink_conf)


    def set_extmag_offset(self,satellite):
        path_extmag_offsets = os.path.join(os.getcwd(), 'extmag_offsets', 'extmag_offset.json')

        # Get Sat ID
        sat_id = self.get_sat_id(satellite)

        # Open Json file with ext. magnetometer offsets
        with open(path_extmag_offsets) as ef:
            extmag_data = json.load(ef)

            for i in extmag_data['offset']:
                if i['sat_id'] == sat_id:

                    B_origin = np.array(i['origin'])
                    dB_xband0 = np.array(i['xband0'])
                    dB_xband1 = np.array(i['xband1'])

                    self.extmag_origin  = B_origin
                    self.extmag_offset0 = B_origin + dB_xband0
                    self.extmag_offset1 = B_origin + dB_xband0 + dB_xband1
                    break


    def write_event_entry(self, start, stop, event_text):
        with open("event.csv", "a+") as f:
            f.write("{}, {}, {}\n".format(start, stop, event_text))


    def create_new_fp(self, filename):
        open(filename, "w+").close()
        return filename

    def write_fp_entry(self, flightplan, name, command, utc_epoch):
        with open(flightplan, "a") as fp:
            fp.write("{ID},{COMMAND},0,0,0,0,{TIME},0,1\n".format(ID=name, COMMAND=command, TIME=utc_epoch))

    def write_flightplan(self, enable_hf,filenumber):
        open('event.csv', 'w').close()
        flightplan = os.path.join(os.getcwd(), 'upload', "plc.fp")
        fp = self.create_new_fp(flightplan)

        #with open(flightplan, "w") as fp:
        # Capture
        if self.ctime:
            self.write_event_entry(self.ctime-640, self.ctime+self.clength+150, "SDR active")
            self.write_event_entry(self.ctime-599, self.ctime+self.clength+120, "AIS active")
            self.write_event_entry(self.ctime-598, self.ctime+self.clength+121, "VHF_FE active")
            self.write_event_entry(self.ctime, self.ctime+self.clength, "Capture")


            self.write_fp_entry(fp, "adcscap01", "adcs server 4 20", self.ctime-1845) # ADCS pointing stuff (GPS antenna pointing directly away from earth)
            self.write_fp_entry(fp, "adcscap02", "adcs state setacs 6", self.ctime-1844)
            self.write_fp_entry(fp, "adcscap03", "rparam download 4 53", self.ctime-1843)
            self.write_fp_entry(fp, "adcscap04", "rparam set euler_offset 0 -1.5708 0", self.ctime-1842)
            self.write_fp_entry(fp, "adcscap05", "rparam send", self.ctime-1841)

            self.write_fp_entry(fp, "gps00a", "adcs server 4 20", self.ctime-1230)
            self.write_fp_entry(fp, "gps00b", "gps off", self.ctime-1229) 
            self.write_fp_entry(fp, "gps00c", "power off 14 GPS", self.ctime-1220)
            self.write_fp_entry(fp, "gps00d", "power on 14 GPS", self.ctime-1210)
            self.write_fp_entry(fp, "gps00e", "gps on", self.ctime-1208) 

            self.write_fp_entry(fp, "cap01", "power on 15 1", self.ctime-640)   # Power on SDR
            self.write_fp_entry(fp, "cap02", "power on 14 3", self.ctime-599)   # Power on AIS
            self.write_fp_entry(fp, "cap03", "power on 15 0", self.ctime-598)   # Power on VHF_FE
            self.write_fp_entry(fp, "cap04", "sdr_client capture load /mnt/data/captureconf.json", self.ctime-597)
            self.write_fp_entry(fp, "cap05", "sdr_client capture start {}".format(self.clength+120), self.ctime)
            self.write_fp_entry(fp, "cap06", "sdr_client capture stop", self.ctime + self.clength)

            if args.xor_data:
                self.write_fp_entry(fp, "cap061", "z7000 node 9", self.ctime + self.clength + 10)
                self.write_fp_entry(fp, "cap062", "z7000 cmd exec /bin/bash xor /mnt/data/xordat.sh {}".format(self.cap_streamid), self.ctime + self.clength + 11)

            self.write_fp_entry(fp, "cap07", "power off 14 3", self.ctime + self.clength + 180) # Power off AIS
            self.write_fp_entry(fp, "cap08", "power off 15 0", self.ctime + self.clength + 181) # Power off VHF_FE
            self.write_fp_entry(fp, "cap09", "power off 15 1", self.ctime + self.clength + 182) # Power off SDR

            self.write_fp_entry(fp, "adcscap06", "adcs server 4 20", self.ctime+183) # ADCS pointing stuff (resume original pointing)
            self.write_fp_entry(fp, "adcscap07", "adcs state setacs 11", self.ctime+184)
            self.write_fp_entry(fp, "adcscap08", "adcs set custom1", self.ctime+185)

        if self.dtime:
            # X-band downlink. SDR in turned on it the capture part
            self.write_event_entry(self.dtime-320, self.dtime+self.dlength+152, "GND tracking")
            self.write_event_entry(self.dtime-120, self.dtime+self.dlength+120, "Syrlinks active")
            self.write_event_entry(self.dtime, self.dtime+self.dlength, "Downlink")

            # Disable CSS
            self.write_fp_entry(fp, "css01", "rparam download 4 80", self.dtime - 60 * 15 - 12) 
            self.write_fp_entry(fp, "css02", "rparam set enable 0 0 0 0 0 0", self.dtime - 60 * 15 - 11)
            self.write_fp_entry(fp, "css03", "rparam send", self.dtime - 60 * 15 - 10)

            # ADCS ground station tracking
            self.write_fp_entry(fp, "adcsdnl01", "rparam download 4 53", self.dtime - 60 * 15)   # Start tracking 15 minutes before x-band start
            if args.dstation == 'puertollano':
                self.write_fp_entry(fp, "adcsdnl02", "rparam set gso_list 4973499 -361926 3964446", self.dtime - 60 * 15 + 1)
            if args.dstation == 'hartebeesthoek':
                self.write_fp_entry(fp, "adcsdnl02", "rparam set gso_list 5084780 2670212 -2768342", self.dtime - 60 * 15 + 1)
            if args.dstation == 'puntaarenas':
                self.write_fp_entry(fp, "adcsdnl02", "rparam set gso_list 1262457 -3639732 -5066216", self.dtime - 60 * 15 + 1)
            if args.dstation == 'awarua':
                self.write_fp_entry(fp, "adcsdnl02", "rparam set gso_list -4305873 885409 -4606038", self.dtime - 60 * 15 + 1)
            if args.dstation == 'nemea':
                self.write_fp_entry(fp, "adcsdnl02", "rparam set gso_list 4655137 1939998 3892067", self.dtime - 60 * 15 + 1)

            print("Downlink targeting {}".format(args.dstation))

            # Command to LVLH-GS TRACKING
            self.write_fp_entry(fp, "adcsdnl03", "rparam set en_gso 1 0 1", self.dtime - 60 * 15 + 2)
            self.write_fp_entry(fp, "adcsdnl04", "rparam send", self.dtime - 60 * 15 + 3)

            # Transition from SUNPOINTING to LVLH for tracking
            self.write_fp_entry(fp, "gstrack01", "adcs server 4 20", self.dtime - 60 * 15 + 4)
            self.write_fp_entry(fp, "gstrack02", "adcs state setacs 6", self.dtime - 60 * 15 + 5)

            # Set HK frequency to 1 Hz for beacon 20
            if enable_hf:
                self.write_fp_entry(fp, "hkset1", "hk_srv beacon samplerate 20 highest", self.dtime - 60 * 15 + 15)

            # Command back to SUNPOINTING
            self.write_fp_entry(fp, "adcsdnl05", "rparam download 4 53", self.dtime + self.dlength + 60 * 1)   # Stop tracking 1 minute after x-band stop
            self.write_fp_entry(fp, "adcsdnl06", "rparam set en_gso 0 0 1", self.dtime + self.dlength + 60 * 1 + 1)
            self.write_fp_entry(fp, "adcsdnl07", "rparam send", self.dtime + self.dlength + 60 * 1 + 2)

            # Transition from LVLH to SUNPOINTING
            self.write_fp_entry(fp, "gstrack03", "adcs server 4 20", self.dtime + self.dlength + 60 * 1 + 3)
            self.write_fp_entry(fp, "gstrack04", "adcs state setacs 11", self.dtime + self.dlength + 60 * 1 + 4)

            # ADCS HK collection
            self.write_fp_entry(fp, "hkget01a", "rm /flash1/ksat1_adcs{}a.dat".format(filenumber), self.dtime + self.dlength + 5 * 60 - 30)
            self.write_fp_entry(fp, "hkget01b", "rm /flash1/ksat1_adcs{}b.dat".format(filenumber), self.dtime + self.dlength + 5 * 60 - 25)
            self.write_fp_entry(fp, "hkget01c", "rm /flash1/ksat1_adcs{}c.dat".format(filenumber), self.dtime + self.dlength + 5 * 60 - 20)
            self.write_fp_entry(fp, "hkget01d", "rm /flash1/ksat1_adcs{}a.zip".format(filenumber), self.dtime + self.dlength + 5 * 60 - 15)
            self.write_fp_entry(fp, "hkget01e", "rm /flash1/ksat1_adcs{}b.zip".format(filenumber), self.dtime + self.dlength + 5 * 60 - 10)
            self.write_fp_entry(fp, "hkget01f", "rm /flash1/ksat1_adcs{}c.zip".format(filenumber), self.dtime + self.dlength + 5 * 60 - 5)
            self.write_fp_entry(fp, "hkget02", "hk server 1", self.dtime + self.dlength + 5 * 60)
            hf_delay = 0
            if enable_hf:
                self.write_fp_entry(fp, "hkget03", "hk get 20 1 1050 {} /flash1/ksat1_adcs%da.dat".format(self.dtime + self.dlength + 3 * 60) % filenumber, self.dtime + self.dlength + 5 * 60 + 5)
                hf_delay = 60
            else:
                self.write_fp_entry(fp, "hkget03", "hk get 20 30 35 {} /flash1/ksat1_adcs%da.dat".format(self.dtime + self.dlength + 3 * 60,) % filenumber, self.dtime + self.dlength + 5 * 60 + 5)
            self.write_fp_entry(fp, "hkget04", "hk get 22 30 35 {} /flash1/ksat1_adcs%db.dat".format(self.dtime + self.dlength + 3 * 60)  % filenumber, self.dtime + self.dlength + 5 * 60 + 25 + hf_delay)
            self.write_fp_entry(fp, "hkget05", "hk get 23 30 35 {} /flash1/ksat1_adcs%dc.dat".format(self.dtime + self.dlength + 3 * 60)  % filenumber, self.dtime + self.dlength + 5 * 60 + 45 + hf_delay)
            self.write_fp_entry(fp, "hkget06", "ftp server 1".format(self.dtime + self.dlength + 2 * 60), self.dtime + self.dlength + 5 * 60 + 65 + hf_delay)
            self.write_fp_entry(fp, "hkget07", "ftp zip /flash1/ksat1_adcs%da.dat /flash1/ksat1_adcs%da.zip".format(self.dtime + self.dlength + 2 * 60) % (filenumber,filenumber), self.dtime + self.dlength + 5 * 60 + 66 + hf_delay)
            self.write_fp_entry(fp, "hkget08", "ftp zip /flash1/ksat1_adcs%db.dat /flash1/ksat1_adcs%db.zip".format(self.dtime + self.dlength + 2 * 60) % (filenumber,filenumber), self.dtime + self.dlength + 5 * 60 + 126 + hf_delay)
            self.write_fp_entry(fp, "hkget09", "ftp zip /flash1/ksat1_adcs%dc.dat /flash1/ksat1_adcs%dc.zip".format(self.dtime + self.dlength + 2 * 60) % (filenumber,filenumber), self.dtime + self.dlength + 5 * 60 + 156 + hf_delay)

            # Set HK frequency back to 0.1Hz
            if enable_hf:
                self.write_fp_entry(fp, "hkset2", "hk_srv beacon samplerate 20 high", self.dtime + self.dlength + 5 * 60 + 180 + hf_delay)

            # Start x-band transmission
            self.write_fp_entry(fp, "dnl00", "rparam autosend 1", self.dtime - 202)
            self.write_fp_entry(fp, "dnl01", "rparam download 4 86", self.dtime - 201)    # Download ADCS extmag table
            self.write_fp_entry(fp, "dnl02", "power on 15 1", self.dtime - 200) # Power on SDR
            self.write_fp_entry(fp, "dnl03", "rparam set offset {:.4f} {:.4f} {:.4f}".format(*self.extmag_offset0), self.dtime - 199) # Apply SDR mag offset
            self.write_fp_entry(fp, "dnl04", "power on 14 6", self.dtime - 140)   # Power on Syrlinks
            self.write_fp_entry(fp, "dnl05", "sdr_client downlink load /mnt/data/downlinkconf.json", self.dtime - 119)
            self.write_fp_entry(fp, "dnl06", "sdr_client downlink start", self.dtime)
            self.write_fp_entry(fp, "dnl07", "rparam download 4 86", self.dtime + 2)
            self.write_fp_entry(fp, "dnl08", "rparam set offset {:.4f} {:.4f} {:.4f}".format(*self.extmag_offset1), self.dtime + 3) # Apply Xband transmission mag offset
            self.write_fp_entry(fp, "dnl09", "sdr_client downlink stop", self.dtime + self.dlength)
            self.write_fp_entry(fp, "dnl10", "power off 14 6", self.dtime + self.dlength + 120)   # Power off Syrlinks
            self.write_fp_entry(fp, "dnl11", "power off 15 1", self.dtime + self.dlength + 150)   # Power off SDR
            self.write_fp_entry(fp, "dnl12", "rparam download 4 86", self.dtime + self.dlength + 151)
            self.write_fp_entry(fp, "dnl13", "rparam set offset {:.4f} {:.4f} {:.4f}".format(*self.extmag_origin), self.dtime + self.dlength + 152) # Apply original mag offset
            self.write_fp_entry(fp, "dnl14", "rparam autosend 0", self.dtime + self.dlength + 153)

            # Enable CSS
            self.write_fp_entry(fp, "css04", "rparam download 4 80", self.dtime + self.dlength + 153 + 10) 
            self.write_fp_entry(fp, "css05", "rparam set enable 1 1 1 1 1 1", self.dtime + self.dlength + 153 + 11)
            self.write_fp_entry(fp, "css06", "rparam send", self.dtime + self.dlength + 153 + 12)

#        print("Flightplan written to " + flightplan)

    def write_commands(self, args):
        commandsfile = os.path.join(os.getcwd(), 'upload', 'commands.txt')
        with open(commandsfile, "w+") as f:
            f.write("##############################\n")
            f.write("### Upload payload tarball ###\n")
            f.write("##############################\n")
            f.write("\n")
            f.write("rdpopt 1 10000 2000 0 1000 1\n")
            f.write("# Power on SDR\n")
            f.write("power on 15 1 0 700\n")
            f.write("\n")
            f.write("# wait 30 seconds\n")
            f.write("\n")
            f.write("ping 9\n")
            f.write("\n")
            f.write("# Upload tarball\n")
            f.write("ftp server 9 180\n")
            f.write("ftp upload /ksm/sdr/{}/{}/plc.tar.gz /mnt/data/plc.tar.gz\n".format(args.destination_directory, args.satellite.lower()))
            f.write("\n")
            f.write("# unpack tarball\n")
            f.write("z7000 node 9\n")
            f.write("z7000 cmd exec /bin/bash upk /mnt/data/unpacktar.sh\n")
            f.write("\n")
            f.write("#########################\n")
            f.write("### Upload flightplan ###\n")
            f.write("#########################\n")
            f.write("\n")
            f.write("ftp server 1 180\n")
            f.write("ftp rm /flash/plc.fp\n")
            f.write("ftp rm /flash/plc.fp.map\n")
            f.write("ftp upload /ksm/sdr/{}/{}/plc.fp /flash/plc.fp\n".format(args.destination_directory, args.satellite.lower()))
            f.write("fp server 1\n")
            f.write("fp load /flash/plc.fp\n")
            f.write("\n")
            f.write("##################################\n")
            f.write("### Request GPS data few times ###\n")
            f.write("##################################\n")
            f.write("\n")
            f.write("hk server 1\n")
            f.write("hk get 24 60 100 0\n")
            f.write("hk get 25 60 100 0\n")
            f.write("\n")

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true','1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='KLEOS capture Generator')
    parser.add_argument('--logging', action='store_true', help='Adds logging.json from the templates folder to the tarball')
    parser.add_argument('--dl_streamid', type=int, action='store', help='Downlink stream ID. IDs can be a list', nargs="+")
    parser.add_argument('--cap_streamid', type=int, action='store', help='Capture stream ID')
    parser.add_argument('-sat', '--satellite', type=str, action='store', required=True,
                        choices=['A', 'B', 'C', 'D'], help='Set name of capture config')
    parser.add_argument('-rate', '--datarate', type=str, action='store', default='low',
                        choices=['low', 'medium', 'high'], help='Select downlink data rate')
    parser.add_argument('--ctime', type=str, action='store',
            help='Specify capture time in ISO format like this 2021-05-25T15:30:45. No timezone offset is added to the specifed time')
    parser.add_argument('--clength', type=int, action='store',
                        help='Length of capture in seconds')
    parser.add_argument('--dtime', type=str, action='store',
                        help='Specify download time in ISO format like this 2021-05-25T15:30:45. No timezone offset is added to the specifed time')
    parser.add_argument('--dlength', type=int, action='store',
                        help='Length of downlink in seconds')
    parser.add_argument('--dstation', type=str, action='store', choices=['puertollano', 'hartebeesthoek', 'puntaarenas', 'awarua', 'nemea'],
                        default='puertollano', help='Name of the X-band ground station')
    parser.add_argument('--enable-hf', action='store_true', help='Enable high frequency data on ADCS bcn 22')
    parser.add_argument('-n','--test_number', type=int, action='store',default=0, help='Set test number to beacon filename')

    parser.add_argument('--xor-data', action='store_true', default=False, help='Xor the data after capture, required if data is captured with the GomSpace data handler')

    parser.add_argument('--chan0-nco-freq', type=int, action='store', default=0, help='Baseband channel 0 NCO frequency in Hz')
    parser.add_argument('--chan0-fixed-deci-enable', type=str2bool, default=True, help='Baseband channel 0 fixed decimation enable (bool)')
    parser.add_argument('--chan0-dynamic-deci', type=int, action='store', default=11, help='Baseband channel 0 dynamic decimation')
    parser.add_argument('--chan1-nco-freq', type=int, action='store', default=0, help='Baseband channel 1 NCO frequency in Hz')
    parser.add_argument('--chan1-fixed-deci-enable', type=str2bool, default=True, help='Baseband channel 1 fixed decimation enable (bool)')
    parser.add_argument('--chan1-dynamic-deci', type=int, action='store', default=10, help='Baseband channel 1 dynamic decimation')
    parser.add_argument('--chan0-timestamp-enable', type=str2bool, default=True, help='Enables interval generated timestamps for baseband channel 0')
    parser.add_argument('--chan0-timestamp-interval', type=int, action='store', default=1000000, help='Number of samples between timestamps for baseband channel 0')
    parser.add_argument('--chan1-timestamp-enable', type=str2bool, default=True, help='Enables interval generated timestamps for baseband channel 1')
    parser.add_argument('--chan1-timestamp-interval', type=int, action='store', default=1000000, help='Number of samples between timestamps for baseband channel 1')
    parser.add_argument('--ad9361-lo', type=int, action='store', default=145175000, help='AD9361 LO')

    parser.add_argument('--destination-directory', type=str, action='store', default='<directory to upload a file>', help='Name of directory in /ksm/sdr/ to upload a file')

    args = parser.parse_args(sys.argv[1:])

    if args.dtime and (args.dl_streamid is None):
        parser.error("if --dtime is given --dl_streamid is needed")

    if args.ctime and (args.cap_streamid is None):
        parser.error("if --ctime is given --cap_streamid is needed")

    if args.dtime:
        tmp = datetime.fromisoformat("{}+00:00".format(args.dtime))  # +00:00 makes it UTC
        args.dtime = int(tmp.timestamp())
    if args.ctime:
        tmp = datetime.fromisoformat("{}+00:00".format(args.ctime))  # +00:00 makes it UTC
        args.ctime = int(tmp.timestamp())

    capgen = Capgen(args.ctime, args.clength, args.dtime, args.dlength, args.satellite, args.datarate, args.dl_streamid, args.cap_streamid)

    #  remove old staging folder if it exists and prepare new folder
    try:
        shutil.rmtree(os.path.join(os.getcwd(), 'staging'))
    except OSError:
        pass
    os.mkdir(os.path.join(os.getcwd(), 'staging'))
    #  remove old upload folder if it exists and prepare new folder
    try:
        shutil.rmtree(os.path.join(os.getcwd(), 'upload'))
    except OSError:
        pass
    os.mkdir(os.path.join(os.getcwd(), 'upload'))

    if capgen.ctime:
        capgen.write_capture_conf(args)
    if capgen.dtime:
        capgen.write_downlink_conf(args.satellite, args.datarate)
    capgen.set_extmag_offset(args.satellite)
    capgen.write_flightplan(args.enable_hf,args.test_number)
    capgen.write_commands(args)
    # move empty shell script to staging folder
    shutil.copyfile(os.path.join(os.getcwd(), 'templates', 'commands.sh'), os.path.join(os.getcwd(), 'staging', 'commands.sh'))
    # move logging.json from templates to staging folder
    if args.logging:
        shutil.copyfile(os.path.join(os.getcwd(), 'templates', 'logging.json'), os.path.join(os.getcwd(), 'staging', 'logging.json'))

    input("Modify files in the 'staging' folder if needed, press <Enter> when ready to compress files")

    # create a tarball of the relevant files
    print("Compressing files")

    tarfiles = []
    tarfiles.append(os.path.join(os.getcwd(), 'staging', 'commands.sh'))
    if args.logging:
        tarfiles.append(os.path.join(os.getcwd(), 'staging', 'logging.json'))
    if args.ctime:
        tarfiles.append(os.path.join(os.getcwd(), 'staging', 'captureconf.json'))
    if args.dtime:
        tarfiles.append(os.path.join(os.getcwd(), 'staging', 'downlinkconf.json'))
    with tarfile.open(os.path.join(os.getcwd(), 'upload', "plc.tar.gz"), "w:gz") as tar:
        for file in tarfiles:
            tar.add(file, arcname=os.path.basename(file))
