"""
All routing should go in this file - it's a route blueprint that
the app factory will use to create a Flask app that can 
be replicated for both testing and running the app
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_user, login_required, logout_user
from datetime import datetime, timedelta
import pytz
import requests

# Import User class and get_utc_time helper from the app module
from .app import User, get_utc_time

routes = Blueprint("routes", __name__, template_folder='../templates') # Point to templates folder

@routes.route("/")
@login_required
def index():
    # Get today's feeding events
    # Access db via current_app proxy
    today = get_utc_time().replace(hour=0, minute=0, second=0, microsecond=0)
    feeding_events = list(current_app.db.feeding_events.find({
        'timestamp': {'$gte': today}
    }).sort('timestamp', -1))
    
    # Calculate statistics
    total_feedings = len(feeding_events)
    last_feeding = feeding_events[0]['timestamp'] if feeding_events else None
    
    return render_template('index.html',
                         feeding_events=feeding_events,
                         total_feedings=total_feedings,
                         last_feeding=last_feeding)

@routes.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Access db via current_app proxy
        user_data = current_app.db.users.find_one({
            'username': username,
            'password': password  # WARNING: Still insecure password handling!
        })
        
        if user_data:
            user = User(user_data) # Use imported User class
            login_user(user)
            # Use blueprint name in url_for
            return redirect(url_for('routes.index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@routes.route('/logout')
@login_required
def logout():
    logout_user()
    # Use blueprint name in url_for
    return redirect(url_for('routes.login'))

@routes.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Access db via current_app proxy
        if current_app.db.users.find_one({'username': username}):
            flash('Username already exists')
            # Use blueprint name in url_for
            return redirect(url_for('routes.register'))
        
        # Create new user
        result = current_app.db.users.insert_one({
            'username': username,
            'password': password  # WARNING: Still insecure password handling!
        })
        
        flash('Registration successful! Please login.')
        # Use blueprint name in url_for
        return redirect(url_for('routes.login'))
    
    return render_template('register.html')

@routes.route('/api/detect-cat', methods=['POST'])
@login_required
def detect_cat():
    """API endpoint to forward browser camera images to ML container"""
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get image data from browser
    image_data = request.json['image']
    
    # Forward the image to the machine learning container
    try:
        ml_response = requests.post(
            'http://machine-learning-client:5000/detect',  # ML container endpoint
            json={'image': image_data}
        )
        
        ml_response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            
        # Simply pass through the ML container's response
        # The ML container is solely responsible for logging detections to the database
        return jsonify(ml_response.json())
        
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Error communicating with ML container: {e}")
        return jsonify({'error': 'Failed to connect to ML service'}), 500
    except Exception as e:
        current_app.logger.error(f"Error processing image detection: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

@routes.route('/api/feeding-events')
@login_required
def get_feeding_events():
    # Define target timezone
    try:
        ny_timezone = pytz.timezone('America/New_York')
        utc_timezone = pytz.utc
    except pytz.UnknownTimeZoneError:
        current_app.logger.error("Could not find timezone 'America/New_York'")
        return jsonify({'error': 'Server timezone configuration error'}), 500

    # Get today's start time in UTC, based on NY timezone midnight
    now_ny = datetime.now(ny_timezone)
    today_start_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_ny.astimezone(utc_timezone)

    # Query events in UTC using current_app.db
    feeding_events_utc = list(current_app.db.feeding_events.find({
        'timestamp': {'$gte': today_start_utc}
    }).sort('timestamp', -1))

    # Convert timestamps to NY time for the response
    events = []
    for event in feeding_events_utc:
        timestamp_utc = event['timestamp']
        # Ensure timestamp from DB is timezone-aware (UTC)
        if timestamp_utc.tzinfo is None:
            timestamp_utc = utc_timezone.localize(timestamp_utc)
        
        timestamp_ny = timestamp_utc.astimezone(ny_timezone)
        timestamp_iso = timestamp_ny.isoformat() # Use ISO format

        events.append({
            'timestamp': timestamp_iso,  # Send NY time in ISO format
            'readable_time': timestamp_ny.strftime('%I:%M:%S %p'), # 12-hour format with AM/PM
            'type': event['type'],
            'confidence': event.get('confidence', None), # Use .get for safety
            'image': event.get('image', None)
        })
    
    total_feedings = len(events)
    
    last_feeding_iso = None
    if feeding_events_utc:
        last_feeding_utc = feeding_events_utc[0]['timestamp']
        if last_feeding_utc.tzinfo is None:
            last_feeding_utc = utc_timezone.localize(last_feeding_utc)
        last_feeding_ny = last_feeding_utc.astimezone(ny_timezone)
        last_feeding_iso = last_feeding_ny.isoformat()

    timestamps_iso = []
    confidences = []
    for event in feeding_events_utc:
        timestamp_utc = event['timestamp']
        if timestamp_utc.tzinfo is None:
            timestamp_utc = utc_timezone.localize(timestamp_utc)
        timestamp_ny = timestamp_utc.astimezone(ny_timezone)
        timestamps_iso.append(timestamp_ny.isoformat())
        confidences.append(event.get('confidence', None))

    return jsonify({
        'feeding_events': events,
        'total_feedings': total_feedings,
        'last_feeding': last_feeding_iso,
        'timestamps': timestamps_iso,
        'confidences': confidences
    })
