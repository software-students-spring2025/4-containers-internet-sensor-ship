from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Initialize MongoDB connection
mongo_client = MongoClient(os.getenv('MONGODB_URI'))
db = mongo_client[os.getenv('MONGODB_DBNAME')]

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Add custom filter for datetime formatting
@app.template_filter('strftime')
def _jinja2_filter_datetime(date, fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S'
    return date.strftime(fmt)

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = db.users.find_one({'_id': ObjectId(user_id)})
        if user_data:
            return User(user_data)
    except:
        pass
    return None

@app.route('/')
@login_required
def index():
    # Get today's feeding events
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    feeding_events = list(db.feeding_events.find({
        'timestamp': {'$gte': today}
    }).sort('timestamp', -1))
    
    # Calculate statistics
    total_feedings = len(feeding_events)
    last_feeding = feeding_events[0]['timestamp'] if feeding_events else None
    
    return render_template('index.html',
                         feeding_events=feeding_events,
                         total_feedings=total_feedings,
                         last_feeding=last_feeding)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_data = db.users.find_one({
            'username': username,
            'password': password  # In a real app, use proper password hashing
        })
        
        if user_data:
            user = User(user_data)
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username already exists
        if db.users.find_one({'username': username}):
            flash('Username already exists')
            return redirect(url_for('register'))
        
        # Create new user
        result = db.users.insert_one({
            'username': username,
            'password': password  # In a real app, use proper password hashing
        })
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 