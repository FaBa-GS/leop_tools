#!/usr/bin/python
# Copyright (c) 2013-2019 GomSpace A/S. All rights reserved.

import verboselogs  # sudo pip install verboselogs
import coloredlogs  # sudo pip install coloredlogs
import logging
from logging.handlers import RotatingFileHandler
import logging.config
import time


def init_logging(name, config_file):
    logging.config.fileConfig(config_file)
    verboselogs.install()
    logger = logging.getLogger(name)
    logging.Formatter.converter = time.localtime
    # printing to console via coloredlogs, using level settings from logging.ini, but setup of log is done separately
    coloredlogs.DEFAULT_DATE_FORMAT = '%Y%m%d-%H:%M:%S'
    coloredlogs.DEFAULT_LEVEL_STYLES = {
        'spam': {'color': 'magenta'},
        'debug': {'color': 'white'},
        'verbose': {'color': 'cyan'},
        'info': {},
        'notice': {'color': 'green'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'color': 'red', 'bold': True}
    }
    coloredlogs.install(level=logger.getEffectiveLevel(),
                        fmt='%(asctime)s.%(msecs)03dZ %(levelname)-8s %(name)s %(message)s')
    return logger