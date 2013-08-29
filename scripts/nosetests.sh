#!/bin/sh

# NOTE: requires autopep8 is installed from pip, e.g. via `pip install -r requirements-dev.txt`

# navigate from `/ushine-learning/scripts` to `/ushine-learning/`
cd ..
nosetests dssg/tests test