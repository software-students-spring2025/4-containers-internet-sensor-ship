"""
WSGI entry point for the Flask application.
This file is needed to run the app with Gunicorn.
"""
from src.app import create_app

# This should be imported by Gunicorn
application = create_app()

if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=True)
