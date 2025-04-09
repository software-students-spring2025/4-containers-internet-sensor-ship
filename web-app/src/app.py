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

# Load environment variables from .env file
# Try multiple possible locations for .env
possible_dotenv_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),  # web-app/.env
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'),  # project root .env
    '.env'  # current directory
]

for dotenv_path in possible_dotenv_paths:
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded environment from {dotenv_path}")
        break
else:
    print("Warning: No .env file found. Using environment variables directly.")

# Define User class (can be here or in a models.py)
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        # Add other user attributes if needed

# Helper function for UTC time
def get_utc_time():
    """Get current time in UTC"""
    return datetime.utcnow() # Keep as naive datetime for MongoDB compatibility

def create_app(test_config=None):
    '''
    The app factory for creating instances of the app
    '''
    # Determine the template folder path - it should be in the parent directory of the src folder
    template_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    static_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    
    # Create app with explicit template path
    app = Flask(__name__, 
                template_folder=template_folder,
                static_folder=static_folder,
                instance_relative_config=True)

    # Print paths for debugging
    print(f"Template folder path: {template_folder}")
    print(f"Static folder path: {static_folder}")
    print(f"Working directory: {os.getcwd()}")

    # --- Configuration ---
    # Use environment variables with fallbacks
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
        MONGODB_URI=os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'),
        MONGODB_DBNAME=os.getenv('MONGODB_DBNAME', 'cat_feeder')
    )

    # Print config for debugging (excluding sensitive info)
    print(f"MongoDB URI configured (showing only prefix): {app.config['MONGODB_URI'].split('@')[0]}...")
    print(f"MongoDB DB name: {app.config['MONGODB_DBNAME']}")

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

    try:
        # --- Extensions ---
        mongo_client = MongoClient(app.config['MONGODB_URI'])
        # Test the connection
        mongo_client.admin.command('ping')
        print("MongoDB connection successful!")
        app.db = mongo_client[app.config['MONGODB_DBNAME']]
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        # Don't raise here - provide a more helpful error but allow app to start
        # This helps with debugging in Docker

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'routes.login'

    # --- User Loader ---
    @login_manager.user_loader
    def load_user(user_id):
        try:
            user_data = app.db.users.find_one({'_id': ObjectId(user_id)})
            if user_data:
                return User(user_data)
        except Exception as e:
            app.logger.error(f"Error loading user {user_id}: {e}")
        return None

    # --- Jinja Filter ---
    @app.template_filter('strftime')
    def _jinja2_filter_datetime(date, fmt='%Y-%m-%d %H:%M:%S'):
        if isinstance(date, datetime):
            return date.strftime(fmt) 
        return date

    # --- Blueprints ---
    try:
        from . import routes
        app.register_blueprint(routes.routes)
        print("Routes blueprint registered successfully")
    except Exception as e:
        print(f"Error registering routes blueprint: {e}")

    # Add a basic route for testing
    @app.route('/health')
    def health_check():
        return {"status": "ok", "message": "App is running"}

    return app

# Remove the old direct run block:
# if __name__ == "__main__":
#    app = create_app()
#    app.run(host='0.0.0.0', debug=True) # Running via 'flask run' or WSGI server is preferred


'''Just changing something silly here to see what happens'''