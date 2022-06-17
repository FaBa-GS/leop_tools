#!/usr/bin/python3.7

import os
import json
from dataclasses import dataclass
import numpy as np

STATUS_NOT_LOADED = 0
STATUS_LOADED = 1
STATUS_PARTIALLY_LOADED = 2

INTMAG_TABLE = 82
EXTMAG_TABLE = 86
SENSOR_COMMON_TABLE = 70
FSS_TABLE = [71,72,73,74,75,76,77,78]
STR_TABLE = 89

@dataclass
class IntMag:
    mag_offset: np.ndarray = np.zeros(3)
    mag_scale: np.ndarray = np.zeros(3)
    mag_rotation: np.ndarray = np.zeros(9)
    mag_rot_t: np.ndarray = np.zeros(9)
    status: int = STATUS_NOT_LOADED


@dataclass
class ExtMag:
    offset: np.ndarray = np.zeros(3)
    scale: np.ndarray = np.zeros(3)
    rotation: np.ndarray = np.zeros(9)
    rot_trans: np.ndarray = np.zeros(9)
    status: int = STATUS_NOT_LOADED

@dataclass
class FineSunSensor:
    q: np.ndarray = np.array([0,0,0,1])
    p: np.ndarray = np.array([0.5,0,0,0])
    status: int = STATUS_NOT_LOADED

@dataclass
class SensorCommon:
    fss_num: int = 0
    fss_darkth: float = 0
    fss_idarkth: float = 0
    fss_brightth: float = 0
    status: int = STATUS_NOT_LOADED

@dataclass
class Startracker:
    str_q: np.ndarray = np.array([0,0,0,1])
    status: int = STATUS_NOT_LOADED


class AdcsConfig:
    # Class for configuration read out
    def __init__(self, adcs_node=4, config_subfolder=None):

        self._adcs_node = adcs_node
        self._config_subfolder = config_subfolder

        # Current Parameters
        self.intmag = IntMag()
        self.extmag = ExtMag()
        self.sensor_common = SensorCommon()
        self.fss = [FineSunSensor() for _ in range(8)]
        self.str = Startracker()

    def load_table(self, adcs_table_number):

        file_name = 'node_' + str(self._adcs_node) + \
            '_table_' + str(adcs_table_number) + '.json'

        if self._config_subfolder is not None:
            fpath = os.path.join(self._config_subfolder, file_name)

        if not os.path.isfile(fpath):
            print("[!] Warning: ADCS TABLE {:d} not found. "
                  "This table will be skipped loading its "
                  "parameters.".format(adcs_table_number))
            self.intmag.status = STATUS_NOT_LOADED
            return

        # Open file and load as JSON file
        with open(fpath, 'r') as f:
            sc_store = json.load(f)

            if isinstance(sc_store, list):
                # Loop through tables (adcs-persistent and adcs-protected)
                for j in range(len(sc_store)):
                    if sc_store[j]['store'] == 'adcs-persistent':
                        # 0 to access adcs-persistent parameters
                        sc_param = sc_store[j]['parameters']
            else:
                sc_param = sc_store['parameters']  # Take the only available

            # Counter to check if all parameters are loaded
            load_counter = 0
            for key in sc_param:

                # TODO: create a function instead of repeating below
                if adcs_table_number == INTMAG_TABLE:
                    # Internal Magnetometer Config Param
                    # TODO: renaming this by updating our adcs software could be a hassle

                    if key['name'] == 'mag_offset':
                        load_counter = load_counter + 1
                        self.set_intmag(mag_offset=np.array(key['value']))
                    elif key['name'] == 'mag_scale':
                        load_counter = load_counter + 1
                        self.set_intmag(mag_scale=np.array(key['value']))
                    elif key['name'] == 'mag_rotation':
                        load_counter = load_counter + 1
                        self.set_intmag(mag_rotation=np.array(key['value']))

                    # Update if all or partially loaded
                    if load_counter == 3:
                        self.intmag.status = STATUS_LOADED
                    elif load_counter > 0:
                        self.intmag.status = STATUS_PARTIALLY_LOADED

                elif adcs_table_number == EXTMAG_TABLE:
                    # External Magnetometer Config Param

                    if key['name'] == 'offset':
                        load_counter = load_counter + 1
                        self.set_extmag(offset=np.array(key['value']))
                    elif key['name'] == 'scale':
                        load_counter = load_counter + 1
                        self.set_extmag(scale=np.array(key['value']))
                    elif key['name'] == 'rotation':
                        load_counter = load_counter + 1
                        self.set_extmag(rotation=np.array(key['value']))

                    # Update if all or partially loaded
                    if load_counter == 3:
                        self.extmag.status = STATUS_LOADED
                    elif load_counter > 0:
                        self.extmag.status = STATUS_PARTIALLY_LOADED

                elif adcs_table_number == SENSOR_COMMON_TABLE:
                    # Sensor Common Config Param
                    # TODO: renaming this by updating our adcs software could be a hassle

                    if key['name'] == 'fss_num':
                        load_counter = load_counter + 1
                        self.set_sensor_common(fss_num=key['value'])
                    elif key['name'] == 'fss_darkth':
                        load_counter = load_counter + 1
                        self.set_sensor_common(fss_darkth=key['value'])
                    elif key['name'] == 'fss_idarkth':
                        load_counter = load_counter + 1
                        self.set_sensor_common(fss_idarkth=key['value'])
                    elif key['name'] == 'fss_brightth':
                        load_counter = load_counter + 1
                        self.set_sensor_common(fss_brightth=key['value'])

                    # Update if all or partially loaded
                    if load_counter == 4:
                        self.sensor_common.status = STATUS_LOADED
                    elif (load_counter > 0 and load_counter < 4):
                        self.sensor_common.status = STATUS_PARTIALLY_LOADED
                
                elif adcs_table_number in FSS_TABLE:
                    # FSS Config Param
                    # TODO: renaming this by updating our adcs software could be a hassle

                    # 71 is first table number
                    n_fss = int(adcs_table_number - 71)

                    if key['name'] == 'q':
                        load_counter = load_counter + 1                        
                        self.set_fss(i=n_fss,q=np.array(key['value']))

                    elif key['name'] == 'p':
                        load_counter = load_counter + 1
                        self.set_fss(i=n_fss,p=np.array(key['value']))

                    # Update if all or partially loaded
                    if load_counter == 2:
                        self.fss[n_fss].status = STATUS_LOADED
                    elif (load_counter > 0 and load_counter < 2):
                        self.fss[n_fss].status = STATUS_PARTIALLY_LOADED

                elif adcs_table_number in STR_TABLE:
                    # STR Config Param
                    # TODO: renaming this by updating our adcs software could be a hassle

                    if key['name'] == 'str_q':
                        load_counter = load_counter + 1                        
                        self.set_str(str_q=np.array(key['value']))

                    # Update if all or partially loaded
                    if load_counter == 1:
                        self.fss[n_fss].status = STATUS_LOADED
                    elif (load_counter > 0 and load_counter < 1):
                        self.fss[n_fss].status = STATUS_PARTIALLY_LOADED
                
    
    def load_all_tables(self):
        self.load_table(INTMAG_TABLE)  # INTMAG
        self.load_table(EXTMAG_TABLE)  # EXTMAG
        self.load_table(SENSOR_COMMON_TABLE)  # SENSOR COMMON
        for i in range(len(FSS_TABLE)): 
            self.load_table(FSS_TABLE[i]) # FSS
        self.load_table(STR_TABLE) # STR

    def _transpose_M(self, B):
        # Transpose matrix put in a single 1D array

        if len(B) is 9:
            # reshape to 3x3 matrix
            B = B.reshape((3, 3))

            # transpose and rehape to 1D array
            B_transpose = (B.transpose()).reshape(9)
            return B_transpose
        else:
            Exception("Input matrix does not have length 9.")

    def set_intmag(self, mag_offset=None, mag_scale=None, mag_rotation=None):
        # Set manually intmag parameters (TODO: maybe overload functions?)

        if mag_offset is not None:
            self.intmag.mag_offset = np.array(mag_offset)
        if mag_scale is not None:
            self.intmag.mag_scale = np.array(mag_scale)
        if mag_rotation is not None:

            mag_rotation = np.array(mag_rotation)

            if len(mag_rotation) is not 9:
                # reshape to 1D array
                mag_rotation = mag_rotation.reshape(9)

            self.intmag.mag_rotation = mag_rotation

            # Get Transposed
            self.intmag.mag_rot_t = self._transpose_M(self.intmag.mag_rotation)

    def set_extmag(self, offset=None, scale=None, rotation=None):
        # Set manually extmag parameters

        if offset is not None:
            self.extmag.offset = np.array(offset)
        if scale is not None:
            self.extmag.scale = np.array(scale)
        if rotation is not None:

            rotation = np.array(rotation)

            if len(rotation) is not 9:
                # reshape to 1D array
                rotation = rotation.reshape(9)

            self.extmag.rotation = rotation

            # Get Transposed
            self.extmag.rot_trans = self._transpose_M(self.extmag.rotation)

    def set_sensor_common(self,fss_num=None,fss_darkth=None,fss_idarkth=None, fss_brightth=None):

        if fss_num is not None:
            self.sensor_common.fss_num = fss_num
        if fss_darkth is not None:
            self.sensor_common.fss_darkth = fss_darkth
        if fss_idarkth is not None:
            self.sensor_common.fss_idarkth = fss_idarkth
        if fss_brightth is not None:
            self.sensor_common.fss_brightth = fss_brightth

    def set_fss(self,i=0,q=None,p=None):

        if q is not None:
            self.fss[i].q = np.array(q)
        if p is not None:
            self.fss[i].p = np.array(p)

    def set_str(self,str_q=None):
        if str_q is not None:
            self.str.str_q = np.array(str_q)

    def print_intmag(self, rparam=False, f=None, log=None):
        # Display INTMAG parameters

        if rparam:
            str_download = ("rparam download {:d} {:d}".format(
                self._adcs_node, INTMAG_TABLE))
            str_send = "rparam send"
            rparam_str = "rparam set "
        else:
            rparam_str = ""

        str_mag_offset = rparam_str + \
            "mag_offset {:.5f} {:.5f} {:.5f}".format(*self.intmag.mag_offset)
        str_mag_scale = rparam_str + \
            "mag_scale {:.5f} {:.5f} {:.5f}".format(*self.intmag.mag_scale)
        str_rot = " ".join("{:.5f}".format(n)
                           for n in self.intmag.mag_rotation)
        str_mag_rot = rparam_str + "mag_rotation " + str_rot

        # rotation array too long, use index instead
        str_mag_rot_i = ""
        for i in range(9):
            str_rot_i = "mag_rotation[{:d}] {:.5f}".format(i, self.intmag.mag_rotation[i])
            str_mag_rot_i = str_mag_rot_i + rparam_str + "".format(i) + str_rot_i + '\n'

        str_rot_t = " ".join("{:.5f}".format(n) for n in self.intmag.mag_rot_t)
        str_mag_rot_t = rparam_str + "mag_rot_t " + str_rot_t


        if f is None:
            str_title = 'A3200 Magnetometer Parameters:'

            str_print = str_mag_offset + '\n' + str_mag_scale + \
                '\n' + str_mag_rot + '\n' + str_mag_rot_t

            if log:
                log.info(str_title)
                log.info(str_mag_offset)
                log.info(str_mag_scale)
                log.info(str_mag_rot)
                log.info(str_mag_rot_t)
            else:
                print(str_title)
                print(str_print)
        else:
            str_print = str_mag_offset + '\n' + str_mag_scale + \
                '\n' + str_mag_rot_i

            str_print = str_download + '\n' + str_print + '\n' + str_send + '\n'

            with open(f, "w") as text_file:
                text_file.write(str_print)

    def print_extmag(self, rparam=False, f=None, log=None):
        # Display EXTMAG parameters

        if rparam:
            str_download = ("rparam download {:d} {:d}".format(
                self._adcs_node, EXTMAG_TABLE))
            str_send = "rparam send"
            rparam_str = "rparam set "
        else:

            rparam_str = ""

        str_mag_offset = rparam_str + \
            "offset {:.5f} {:.5f} {:.5f}".format(*self.extmag.offset)
        str_mag_scale = rparam_str + \
            "scale {:.5f} {:.5f} {:.5f}".format(*self.extmag.scale)
        str_rot = " ".join("{:.5f}".format(n) for n in self.extmag.rotation)
        str_mag_rot = rparam_str + "rotation " + str_rot

        # rotation array too long, use index instead
        str_mag_rot_i = ""
        for i in range(9):
            str_rot_i = "rotation[{:d}] {:.5f}".format(i, self.extmag.rotation[i])
            str_mag_rot_i = str_mag_rot_i + rparam_str + str_rot_i + '\n'

        str_rot_t = " ".join("{:.5f}".format(n) for n in self.extmag.rot_trans)
        str_mag_rot_t = rparam_str + "rot_trans " + str_rot_t

        if f is None:
            str_title = 'M315 Magnetometer Parameters:'

            str_print = str_mag_offset + '\n' + str_mag_scale + \
                '\n' + str_mag_rot + '\n' + str_mag_rot_t

            if log:
                log.info(str_title)
                log.info(str_mag_offset)
                log.info(str_mag_scale)
                log.info(str_mag_rot)
                log.info(str_mag_rot_t)
            else:
                print(str_title)
                print(str_print)
        else:
            str_print = str_mag_offset + '\n' + str_mag_scale + \
                '\n' + str_mag_rot_i

            str_print = str_download + '\n' + str_print + '\n' + str_send + '\n'

            with open(f, "w") as text_file:
                text_file.write(str_print)

    def print_sensor_common(self, rparam=False, f=None, log=None):
        # Display SENSOR COMMON parameters

        if rparam:
            str_download = ("rparam download {:d} {:d}".format(
                self._adcs_node, SENSOR_COMMON_TABLE))
            str_send = "rparam send"
            rparam_str = "rparam set "
        else:
            rparam_str = ""

        str_fss_num = rparam_str + \
            "fss_num {:d}".format(self.sensor_common.fss_num)
        str_fss_darkth = rparam_str + \
            "fss_darkth {:.0f}".format(self.sensor_common.fss_darkth)
        str_fss_idarkth = rparam_str + \
            "fss_idarkth {:.0f}".format(self.sensor_common.fss_idarkth)
        str_fss_brightth = rparam_str + \
            "fss_brightth {:.0f}".format(self.sensor_common.fss_brightth)

        str_print = str_fss_num + '\n' + str_fss_darkth + \
                '\n' + str_fss_idarkth + '\n' + str_fss_brightth

        if f is None:
            str_title = 'Sensor Common Parameters:'

            if log:
                log.info(str_title)
                log.info(str_fss_num)
                log.info(str_fss_darkth)
                log.info(str_fss_idarkth)
                log.info(str_fss_brightth)
            else:
                print(str_title)
                print(str_print)
        else:
            str_print = str_download + '\n' + str_print + '\n' + str_send + '\n'

            with open(f, "w") as text_file:
                text_file.write(str_print)

    def print_fss(self, nr_fss, rparam=False, f=None, log=None):
        # Display SENSOR COMMON parameters

        if rparam:
            str_download = ("rparam download {:d} {:d}".format(
                self._adcs_node, FSS_TABLE[nr_fss]))
            str_send = "rparam send"
            rparam_str = "rparam set "
        else:
            rparam_str = ""

        str_q = rparam_str + \
            "q {:.5f} {:.5f} {:.5f} {:.5f}".format(*self.fss[nr_fss].q)
        str_p = rparam_str + \
            "p {:.5f} {:.5f} {:.5f} {:.5f}".format(*self.fss[nr_fss].p)

        str_print = str_q + '\n' + str_p + '\n'

        if f is None:
            str_title = 'FSS {:d}:'.format(nr_fss)

            if log:
                log.info(str_title)
                log.info(str_q)
                log.info(str_p)

            else:
                print(str_title)
                print(str_print)
        else:
            str_print = str_download + '\n' + str_print + '\n' + str_send + '\n'

            with open(f, "w") as text_file:
                text_file.write(str_print)
    
    def print_str(self, rparam=False, f=None, log=None):
        # Display SENSOR COMMON parameters

        if rparam:
            str_download = ("rparam download {:d} {:d}".format(
                self._adcs_node, STR_TABLE))
            str_send = "rparam send"
            rparam_str = "rparam set "
        else:
            rparam_str = ""

        str_str_q = rparam_str + \
            "str_q {:.5f} {:.5f} {:.5f} {:.5f}".format(*self.str.str_q)
        
        str_print = str_str_q + '\n'

        if f is None:
            str_title = 'STR:'

            if log:
                log.info(str_title)
                log.info(str_str_q)

            else:
                print(str_title)
                print(str_print)
        else:
            str_print = str_download + '\n' + str_print + '\n' + str_send + '\n'

            with open(f, "w") as text_file:
                text_file.write(str_print)