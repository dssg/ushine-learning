#!/usr/bin/python
# -*- coding: utf-8 -*-
import ConfigParser
import logging as logger
from os.path import dirname, realpath

from flask.ext.sqlalchemy import SQLAlchemy

import dssg
from dssg.webapp import app

def init_db(config):
    """Initializes the database connection"""
    try:
        database_uri = config.get('database', 'sqlalchemy.url')
        pool_size = config.get('database', 'sqlalchemy.pool_size')

        # SQLAlchemy configuration
        app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
        app.config['SQLALCHEMY_POOL_SIZE'] = pool_size
    except ConfigParser.NoSectionError, e:
        logger.error("The specified section does not exist", e)
        
    dssg.db = SQLAlchemy(app)

# Load the database configuration
config = ConfigParser.SafeConfigParser()
try:
    config.readfp(open(dirname(realpath(__file__)) + '/dssg/config/dssg.ini'))
except IOError, e:
    logger.error("Error opening file", e)

init_db(config)

# Initialze the db
app.secret_key = 'YOUR SECRET KEY'


if __name__ == "__main__":
    app.debug = True
    app.run()
