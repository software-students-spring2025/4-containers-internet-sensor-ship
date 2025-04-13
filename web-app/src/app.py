"""
The main flask app factory
"""

from datetime import datetime
import os

from flask import Flask, jsonify, request, render_template, redirect, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from pymongo import MongoClient
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin):
    """User class for flask-login"""
    def __init__(self, user_data):
        if isinstance(user_data, dict):
            self.id = str(user_data.get("_id", ""))
            self.username = user_data.get("username", "")
        else:
            self.id = str(user_data) 
            self.username = "Unknown"

def get_utc_time():
    """Get current time in UTC"""
    return datetime.utcnow()  

def create_app():
    """
    The app factory for creating instances of the app
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')
    app.config['MONGO_URI'] = os.getenv('MONGODB_URI', 'mongodb://mongodb:27017/')
    app.config['MONGODB_DBNAME'] = os.getenv('MONGODB_DBNAME', 'cat_monitor')
    client = MongoClient(app.config['MONGO_URI'], serverSelectionTimeoutMS=5000)
    app.db = client[app.config['MONGODB_DBNAME']]
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'routes.login'
    
    @login_manager.user_loader
    def load_user(user_id):
        try:
            app.logger.info(f"Loading user with ID: {user_id}")
            if not user_id:
                app.logger.warning("No user_id provided to user_loader")
                return None
                
            user = app.db.users.find_one({'_id': ObjectId(user_id)})
            if user:
                app.logger.info(f"User found: {user.get('username', 'unknown')}")
                return User(user)
            else:
                app.logger.warning(f"No user found with ID: {user_id}")
                return None
        except Exception as e:
            app.logger.error(f"Error loading user {user_id}: {e}")
            return None
    
    from .routes import bp as routes_bp
    app.register_blueprint(routes_bp)
    
    @app.route('/health')
    def health_check():
        return {"status": "ok", "message": "App is running"}
    
    @app.route('/api/register', methods=['POST'])
    def register():
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return jsonify({'error': 'Username and password are required'}), 400
                
            if app.db.users.find_one({'username': username}):
                return jsonify({'error': 'Username already exists'}), 400
                
            hashed_password = generate_password_hash(password)
            user_id = app.db.users.insert_one({
                'username': username,
                'password': hashed_password,
                'created_at': datetime.utcnow()
            }).inserted_id
            
            return jsonify({
                'message': 'User registered successfully',
                'user_id': str(user_id)
            }), 201
        except Exception as e:
            app.logger.error('Registration error: %s', e)
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/direct-login', methods=['GET', 'POST'])
    def direct_login():
        if request.method == 'POST':
            try:
                username = request.form.get('username')
                password = request.form.get('password')
                
                if not username or not password:
                    flash('Username and password are required')
                    return render_template('login.html')
                    
                user_data = app.db.users.find_one({'username': username})
                if user_data and check_password_hash(user_data['password'], password):
                    user = User(user_data)
                    login_user(user)
                    return redirect('/')
                    
                flash('Invalid username or password')
            except Exception as e:
                app.logger.error('Direct login error: %s', e)
                flash('An error occurred during login')
                
        return render_template('login.html')
    
    @app.route('/direct-logout')
    def direct_logout():
        logout_user()
        return redirect('/direct-login')
    
    @app.route('/')
    def home():
        try:
            app.logger.info("Home route accessed")
            if current_user.is_authenticated:
                app.logger.info(f"User is authenticated: {current_user.username}")
                return render_template('index.html', 
                                      feeding_events=[], 
                                      total_feedings=0, 
                                      last_feeding=None,
                                      message="Welcome! The application is running.")
            else:
                app.logger.info("User is not authenticated")
                return redirect('/direct-login')
        except Exception as e:
            app.logger.error(f"Error in home route: {e}")
            return "Welcome to the IoT Pet Feeder! <a href='/direct-login'>Login</a>"
    
    return app


# Remove the old direct run block:
# if __name__ == "__main__":
#    app = create_app()
#    app.run(host='0.0.0.0', debug=True) # Running via 'flask run' or WSGI server is preferred


"""Just changing something silly here to see what happens"""
