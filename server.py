#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging as logger
from os.path import dirname, realpath

import dssg
from dssg.webapp import app

# Load the application file
config_file = dirname(realpath(__file__)) + '/dssg/config/dssg.ini'
dssg.load_config(app, config_file)

# Import the API endpoints
from dssg.webapp.rest_api import *


if __name__ == "__main__":
    app.debug = True
    app.run()
