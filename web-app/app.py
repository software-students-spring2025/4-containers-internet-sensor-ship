from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId
import os
from dotenv import load_dotenv
import pytz  # Add pytz for timezone handling

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

# Use UTC for consistency
def get_utc_time():
    """Get current time in UTC"""
    return datetime.utcnow()

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
    today = get_utc_time().replace(hour=0, minute=0, second=0, microsecond=0)
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

@app.route('/api/detect-cat', methods=['POST'])
@login_required
def detect_cat():
    """API endpoint to forward browser camera images to ML container"""
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get image data from browser
    image_data = request.json['image']
    
    # Forward the image to the machine learning container for processing
    # The ML container runs on the Docker network with hostname 'machine-learning-client'
    try:
        # Import requests module if not already imported
        import requests
        
        # Send the image to the ML container for processing
        ml_response = requests.post(
            'http://machine-learning-client:5000/detect',  # ML container endpoint
            json={'image': image_data}
        )
        
        if ml_response.status_code != 200:
            return jsonify({'error': 'ML service error'}), 500
            
        # Get detection result from ML container
        detection_result = ml_response.json()
        
        # Only save to database if cat detected AND the ML service didn't log it
        if detection_result.get('detected', False) and not detection_result.get('logged', True):
            event = {
                'timestamp': get_utc_time(),
                'type': 'feeding',
                'confidence': detection_result.get('confidence', 0.8),
                'image': image_data
            }
            db.feeding_events.insert_one(event)
            detection_result['logged'] = True
        
        return jsonify(detection_result)
        
    except Exception as e:
        print(f"Error communicating with ML container: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/api/feeding-events')
@login_required
def get_feeding_events():
    # Define target timezone
    ny_timezone = pytz.timezone('America/New_York')
    utc_timezone = pytz.utc

    # Get today's start time in UTC, based on NY timezone midnight
    now_ny = datetime.now(ny_timezone)
    today_start_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_ny.astimezone(utc_timezone)

    # Query events in UTC
    feeding_events_utc = list(db.feeding_events.find({
        'timestamp': {'$gte': today_start_utc}
    }).sort('timestamp', -1))

    # Convert timestamps to NY time for the response
    events = []
    for event in feeding_events_utc:
        timestamp_utc = event['timestamp'].replace(tzinfo=utc_timezone)
        timestamp_ny = timestamp_utc.astimezone(ny_timezone)
        
        timestamp_iso = timestamp_ny.strftime('%Y-%m-%dT%H:%M:%S%z')

        events.append({
            'timestamp': timestamp_iso,  # Send NY time in ISO format
            'readable_time': timestamp_ny.strftime('%I:%M:%S %p'), # 12-hour format with AM/PM
            'type': event['type'],
            'confidence': event['confidence'],
            'image': event.get('image', None)
        })
    
    # Calculate statistics
    total_feedings = len(events)
    
    # Convert last feeding time to NY time ISO format
    last_feeding = None
    if feeding_events_utc:
        last_feeding_utc = feeding_events_utc[0]['timestamp'].replace(tzinfo=utc_timezone)
        last_feeding_ny = last_feeding_utc.astimezone(ny_timezone)
        last_feeding = last_feeding_ny.strftime('%Y-%m-%dT%H:%M:%S%z')

    # Get chart data with NY time ISO format timestamps
    timestamps = []
    confidences = []
    for event in feeding_events_utc:
        timestamp_utc = event['timestamp'].replace(tzinfo=utc_timezone)
        timestamp_ny = timestamp_utc.astimezone(ny_timezone)
        timestamps.append(timestamp_ny.strftime('%Y-%m-%dT%H:%M:%S%z'))
        confidences.append(event['confidence'])

    return jsonify({
        'feeding_events': events,
        'total_feedings': total_feedings,
        'last_feeding': last_feeding,
        'timestamps': timestamps,
        'confidences': confidences
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 