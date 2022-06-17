#!/usr/bin/python3.7
# Copyright (c) 2013-2020 GomSpace A/S. All rights reserved.

from cgitb import enable
import os
import numpy as np
from datetime import datetime

LVLH = 2 # Local vertical local horizontal
LPRF = 4 # Landmark tracking
SPRF = 6 # Sunpointing

class FlightPlan:

    def __init__(self, filename='test.fp', cmd_name='cmd', linecount=0, t_cmd=0, test_nr=0, enable_rel=False):

        self.cmd_name = cmd_name        # Command Name
        self.linecount = int(linecount) # Command Line Number
        self.set_test_nr(test_nr)
        self.relative_mode = enable_rel
        
        if isinstance(t_cmd,str):
            tmp = datetime.fromisoformat("{}+00:00".format(t_cmd)) # +00:00 makes it UTC
            self.t_cmd = int(tmp.timestamp())
        else:
            self.t_cmd = int(t_cmd)

        if self.t_cmd > 86400 and self.relative_mode:
            raise Warning("Time set is larger than 1 day ({:d}) in relative FP mode.".format(int(self.t_cmd)))

        # Current directoy
        dir_path = os.getcwd()

        # Create output folder if not exist
        output_path = os.path.join(dir_path, 'output')

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Delete previous FP file w/ same name
        output_file = os.path.join(output_path, filename)

        if os.path.exists(output_file):
            os.remove(output_file)
        
        self.f = open(output_file, 'w')

        # Base magnetometer offset 
        self.current_offset = np.zeros(3)

    def close(self,):
        # Close Flightplanner File
        if self.f:
            self.f.close()
            self.f = None

    def delay(self, delay_s):
        # Add a delay in the command time
        self.t_cmd = self.t_cmd + delay_s

    def set_time(self,t):
        # Change the time on the FP
        if isinstance(t,str):
            tmp = datetime.fromisoformat("{}+00:00".format(t)) # +00:00 makes it UTC
            t = tmp.timestamp()
        
        if (self.t_cmd > t):
            dt = self.t_cmd - t
            raise Warning("New set time is before the current FP time by {:d} s.".format(int(dt)))
        
        if self.t_cmd > 86400 and self.relative_mode:
            raise Warning("Time set is larger than 1 day ({:d}) in relative FP mode.".format(int(self.t_cmd)))

        self.t_cmd = int(t)

    def set_test_nr(self,test_nr):
        if type(test_nr) is np.ndarray:
            self.test_nr = test_nr[0]
        else:
            self.test_nr = test_nr   

    def get_cmd_time(self,time_format='unix'):
        # Get current time in unix or utc
        if (time_format == 'utc') or (time_format == 'UTC'):
            t = datetime.utcfromtimestamp(self.t_cmd).strftime('%Y-%m-%d %H:%M:%S')
        else:
            t = int(self.t_cmd)

        return t

    def printline(self, cmd=""):
        # Flight Plan command format:
        # name,command,state,basis,last_sec,last_nsec,when_sec,when_nsec,repeat
        #
        # name: Unique id of the timer.
        # state: Specifies whether the timer is active (0) or dormant (1).
        # basis: Specifies whether the timer is an absolute (0) or relative (1) timer.
        # last_sec: The last execution timestamp seconds part. Ignored for absolute timers.
        # last_nsec: The last execution timestamp nanoseconds part. Ignored for absolute timers.
        # when_sec: Execution timestamp seconds part.
        # when_nsec: Execution timestamp nanoseconds part.
        # repeat: Number of repeats. Only for relative timers.

        # Increment Linecount
        self.linecount += 1

        command_id = self.cmd_name + str(self.test_nr) + "_" + str(self.linecount)

        # First part of command for alignment
        str_cmd = command_id + "," + cmd

        # Enable/disable relative mode print
        if self.relative_mode:
            rel_flag = 1
        else:
            rel_flag = 0

        # print
        print(str_cmd.ljust(60) + ",0,{:d},0,0,".format(rel_flag) + format(int(self.t_cmd)) + ",0,0", file=self.f)

        self.delay(1)


    def store_hk(self, bcn_fname, bcn_number, sample_rate, n_samples, t_hk, short_delay=False):
        # Store beacon to file
        self.delay(1)
        self.printline('hk server 1')

        # Store beacon
        self.printline('hk get {:d} {:d} {:d} {:d} {:s}.bin'.format(bcn_number,
                       sample_rate, n_samples, t_hk, bcn_fname))

        if short_delay:
            # Wait 10 seconds until next command
            self.delay(10)
        else:
            # Wait 30 seconds until next command
            self.delay(30)


    def zip_file(self, bcn_fname, short_delay=False):
        # Zip file
        self.delay(1)
        self.printline('ftp server 1')
        self.printline('ftp zip {:s}.bin {:s}.zip'.format(bcn_fname,bcn_fname))

        if short_delay:
            # Wait 10 seconds until next command
            self.delay(10)
        else:
            # Wait 30 seconds until next command
            self.delay(30)

    def set_magoffset(self,offset,use_intmag=False):
        
        # check offset value do not contain zeros
        if np.any(offset):

            # Add new offset to current
            applied_offset = self.current_offset + offset
            
            if use_intmag:
                # internal magnetometer
                self.printline('rparam download 4 82')
                self.printline('rparam set mag_offset {:.6f} {:.6f} {:.6f}'.format(*applied_offset))
                self.printline('rparam send')

            else:
                # external magnetometer
                self.printline('rparam download 4 86')
                self.printline('rparam set offset {:.6f} {:.6f} {:.6f}'.format(*applied_offset))
                self.printline('rparam send')

            # Update current offset
            self.current_offset = applied_offset
    
    def change_acs_mode(self, euler_angle, location=None, mode='LVLH'):
        """ Change to different mode
        SPRF: sunpointing
        LM: landmark tracking
        LVLH: local vertical local horizontal
        """ 

        if mode == LVLH:
            acs_mode   = 6
            en_gso     = 0
            euler_offset_str = 'euler_offset'
        elif mode == LPRF:
            acs_mode   = 6
            en_gso     = 1
            euler_offset_str = 'gso_ant'
        elif mode == SPRF:
            acs_mode   = 11
            en_gso     = 0
            euler_offset_str = 'euler_offset'
        else:
            raise Exception('Illegal acs mode selected')

        # TODO: make configurable and this is not compatible with versions >=5.3.2
        self.printline('rparam download 4 53')
        self.printline('rparam set {:s} {:.4f} {:.4f} {:.4f}'.format(euler_offset_str,*euler_angle))
        self.printline('rparam set en_gso {:d} 0 1'.format(en_gso))
        if location is not None:
            self.printline('rparam set gso_list {:.1f} {:.1f} {:.1f}'.format(*location))
        self.printline('rparam send')

        self.printline('adcs server 4 20')
        self.printline('adcs state setacs {:d}'.format(acs_mode))