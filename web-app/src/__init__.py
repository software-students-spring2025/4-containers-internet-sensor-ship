"""
Flask app package for the Cat Feeder application
"""
# Import and re-export the create_app function
from .app import create_app, User, get_utc_time

# This enables: from src import create_app
__all__ = ["create_app", "User", "get_utc_time"]
