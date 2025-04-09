"""
All routing should go in this file - it's a route blueprint that
the app factory will use to create a Flask app that can 
be replicated for both testing and running the app
"""
from flask import Blueprint

routes = Blueprint("routes", __name__)


@routes.route("/")
def index():
    return "Hello World!"
