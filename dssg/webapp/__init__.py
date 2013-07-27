from flask import Flask

# Flask application instance
app = Flask(__name__)

from main import home, machine, main_menu
from rest_api import *
