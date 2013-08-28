import ConfigParser
import os

from flask.ext.sqlalchemy import SQLAlchemy

import util as util
from classifier import DssgCategoryClassifier
from machine import Machine

db = None
machine = None
category_classifier = None


def load_config(app, config_file):
    """Loads the configuration from the specified file and sets the
    properties of ```app```, ```db``` and ```machine``` application objects

    :param app: the flask application object
    :param config_file: the absolute path to the configuration file
    """
    global db, machine, category_classifier

    config = ConfigParser.SafeConfigParser()

    try:
        config.readfp(open(config_file))
    except IOError as e:
        app.logger.error("An error while reading '%s': %s" %
                        (config_file, e.strerror))

    # Initialize the database
    try:
        database_uri = config.get('database', 'sqlalchemy.url')
        pool_size = config.get('database', 'sqlalchemy.pool_size')

        # SQLAlchemy configuration
        app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
        app.config['SQLALCHEMY_POOL_SIZE'] = int(pool_size)
    except ConfigParser.NoSectionError as e:
        logger.error("The specified section does not exist", e)

    db = SQLAlchemy(app)

    # Intialize the machine
    classifier_file = config.get("classifier", "classifier.file")
    if not classifier_file is None:
        if os.path.exists(classifier_file):
            _dict = util.load_pickle(classifier_file)
            category_classifier = _dict['categoryClassifier']
            if not isinstance(category_classifier, DssgCategoryClassifier):
                app.logger.error("Invalid classifier object type: %s" %
                                 type(category_classifier))
                category_classifier = None
                return
            # Proceed
            machine = Machine(category_classifier)
        else:
            app.logger.info("The classifier file '%s' does not exist" %
                            classifier_file)
