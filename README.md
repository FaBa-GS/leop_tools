# leop-tools

These tools are to be used for LEOP. Note that these scripts may be project specific, and new projects should take into account for current commands. To highlight, the reference frame modes are not applied in these script, and missions with ADCS image post 5.3.2 require an update of these script to be compatible.

# General Outline
To use the scripts during LEOP, it is recommended to follow the following sequence after the satellite has been confirmed to be SAFE.

1. Transition the satellite in COARSE 3-axis control mode. Include the external GYRO if possible in the loop.
2. Perform mockup Sband test. 
    1. Copy *example_offset_fp.json* and fill in to match with the satellite's configuration.
    2. Execute `generate_fp_sband.py` using the AOS and LOS times of the desired pass. Add the *--mockup* flag.
    3. Upload FP to satellite, download the data, and load it into the database.
1. Determine magnetometer offset of Sband related components.
    1. Copy *example_offset_test.json* and fill in to match with the satellite's configuration, and the flightplan timestamps defined in the previous step.
    1. Extract the data from the database using `gsweb_extractor.py` (Example json file of json_config_file can be found as *example_cal_bcn.json* in test_data subfolder):

        ````python
        python3 gsweb_extractor.py --host <ip_address> --db <database_name> --node <adcs_csp_node> --satellite <sat_number> --from_ts <unixtime_start> --to_ts <unixtime_end> --config <json_config_file> --output <filename> --resolution 1
        ````
    
    1. Execute 'get_payload_offsets.py using the extracted file and the .json file to estimate the new offsets. In most cases, offset2 provides a more accurate offset than offset1.
    1. Copy the offset retrieved to the copy of *example_offset_fp.json* file in the second step. 
    1. Execute `generate_fp_sband.py` using the AOS and LOS times of the desired pass. Without the *--mockup* flag.
    1. Upload FP to satellite, download the data, and load it into the database.
    1. Verify that no discrete jumps in the magneteometer is observed, and that Sband link has been established.
1. Perform calibration (power on relevant ADCS components if required also relevant PDUs)
    1. Copy *example_calibration.json* and fill in to match with the satellite's configuration.
    1. Add the most up-to-date satellite's TLE file similar to *tle_test.txt*
    1. Execute `generate_fp_cal.py` add a desired time, and amount of tests to performed using -n.
    1. Upload FP to satellite, download the data, and load it into the database.
    1. Extract the data from the database using `gsweb_extractor.py`
    1. Execute `oo_autocalib.py` with the extracted file and the .json file. 
    1. Verify that calibration figures
    1. Upload calibration parameters to satellite. DO NOT SAVE YET.
    1. Rerun the ADCS (NO REBOOT)
    1. Check if satellite stablizes.
    1. Save the parameters after satellite has stabilized.
    1. Redo above substeps 2-6 to verify that calibration has been applied correctly.
1. Repeat same procedure for Xband as done for Sband if applicable.

# [EXPERIMENTAL] gps2tle.py

This tool converts a file with gps_epoch, gps_pos, and gps_vel data to TLE. Note that a gps data file must be specified. 

# generate_fp_cal_coarse.py
Coase calibration FP generator, calibration maneouver used in case the magnetometers do not provide the performance to transition coarse 3-axis control mode.