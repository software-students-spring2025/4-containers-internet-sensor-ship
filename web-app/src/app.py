'''
The main flask app factory
'''
from flask import Flask
from flask_login import LoginManager, UserMixin
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz

# Load environment variables from .env file at the project root
# Assuming .env is in the root directory, adjust path if needed
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Define User class (can be here or in a models.py)
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        # Add other user attributes if needed

# Helper function for UTC time
def get_utc_time():
    """Get current time in UTC"""
    return datetime.utcnow().replace(tzinfo=pytz.utc) # Make timezone aware

def create_app(test_config=None):
    '''
    The app factory for creating instances of the app
    '''
    app = Flask(__name__, instance_relative_config=True)

    # --- Configuration ---
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'), # Provide a default for dev
        MONGODB_URI=os.getenv('MONGODB_URI'),
        MONGODB_DBNAME=os.getenv('MONGODB_DBNAME')
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass # Already exists

    if not app.config['MONGODB_URI'] or not app.config['MONGODB_DBNAME']:
        raise ValueError("MONGODB_URI and MONGODB_DBNAME must be set in environment variables or config.")

    # --- Extensions ---
    mongo_client = MongoClient(app.config['MONGODB_URI'])
    app.db = mongo_client[app.config['MONGODB_DBNAME']] # Attach db client to app context

    login_manager = LoginManager()
    login_manager.init_app(app)
    # Specify the login view using the blueprint name 'routes' and the function name 'login'
    login_manager.login_view = 'routes.login'

    # --- User Loader ---
    @login_manager.user_loader
    def load_user(user_id):
        try:
            # Access db via the app context
            user_data = app.db.users.find_one({'_id': ObjectId(user_id)})
            if user_data:
                return User(user_data)
        except Exception as e:
            app.logger.error(f"Error loading user {user_id}: {e}") # Log errors
        return None

    # --- Jinja Filter ---
    @app.template_filter('strftime')
    def _jinja2_filter_datetime(date, fmt='%Y-%m-%d %H:%M:%S'):
        # Ensure date is timezone-aware if possible, otherwise assume UTC
        if isinstance(date, datetime):
            if date.tzinfo is None:
                date = date.replace(tzinfo=pytz.utc) # Assume UTC if naive
            # Example: Convert to a specific timezone if needed for display
            # ny_timezone = pytz.timezone('America/New_York')
            # date_local = date.astimezone(ny_timezone)
            # return date_local.strftime(fmt)
            return date.strftime(fmt) # Keep as UTC for now unless specified otherwise
        return date # Return as is if not a datetime object

    # --- Blueprints ---
    # Import and register the blueprint from routes.py
    from . import routes # Use relative import
    app.register_blueprint(routes.routes) # Assuming the blueprint instance is named 'routes' in routes.py

    # --- App Context ---
    # You might want to push an application context for certain operations
    # Or use Flask-PyMongo extension which handles context better

    return app

# Remove the old direct run block:
# if __name__ == "__main__":
#    app = create_app()
#    app.run(host='0.0.0.0', debug=True) # Running via 'flask run' or WSGI server is preferred


'''Just changing something silly here to see what happens'''