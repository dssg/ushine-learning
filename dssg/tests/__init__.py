from flask.ext.sqlalchemy import SQLAlchemy

import dssg
from dssg.webapp import app

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
dssg.db = SQLAlchemy(app)

# Import the models after initializing SQLAlchemy
import dssg.model as _model


def setup_module():
    """Tests the model mapping and subsequent table creation"""
    dssg.db.create_all()


def teardown_module():
    """Drop all the schema tables"""
    dssg.db.drop_all()


def create_deployment(name, url):
    deployment = _model.Deployment(name=name, url=url)
    deployment.save()

    return deployment
