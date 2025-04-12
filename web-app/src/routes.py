"""
All routing should go in this file - it's a route blueprint that
the app factory will use to create a Flask app that can
be replicated for both testing and running the app
"""

from datetime import datetime

import pytz
import requests
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    current_app,
)
from flask_login import login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash

from .app import User, get_utc_time

bp = Blueprint("routes", __name__, template_folder="../templates")  

@bp.route("/")
@login_required
def index():
    try:
        current_app.logger.info("Index route accessed")
        
        collections = current_app.db.list_collection_names()
        if "feeding_events" not in collections:
            current_app.logger.warning("feeding_events collection does not exist yet")
            return render_template(
                "index.html",
                feeding_events=[],
                total_feedings=0,
                last_feeding=None,
            )
        
        all_events = list(current_app.db.feeding_events.find().sort("timestamp", -1))
        current_app.logger.info(f"Found {len(all_events)} total feeding events")
        today_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        feeding_events = []
        for event in all_events:
            try:
                timestamp = event.get("timestamp")
                if timestamp and isinstance(timestamp, datetime):
                    if timestamp.date() >= today_date.date():
                        # Add this event to the list for today
                        feeding_events.append(event)
            except Exception as e:
                current_app.logger.error(f"Error processing event in index: {e}")
        total_feedings = len(feeding_events)
        last_feeding = feeding_events[0]["timestamp"] if feeding_events else None
        
        current_app.logger.info(f"Returning template with {total_feedings} events")
        
        return render_template(
            "index.html",
            feeding_events=feeding_events,
            total_feedings=total_feedings,
            last_feeding=last_feeding,
        )
    except Exception as e:
        current_app.logger.error("Error in index route: %s", e)
        return render_template(
            "index.html",
            feeding_events=[],
            total_feedings=0,
            last_feeding=None,
            error=str(e)
        )


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        current_app.logger.info("Login POST request received")
        try:
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            
            if not username or not password:
                current_app.logger.warning("Missing username or password")
                flash("Username and password are required")
                return render_template("login.html")
            current_app.logger.info("Attempting to find user: %s", username)
            user_data = current_app.db.users.find_one({"username": username})
            
            if user_data:
                current_app.logger.info("User found, checking password")
                if check_password_hash(user_data["password"], password):
                    current_app.logger.info("Password verified, creating user object")
                    user = User(user_data)  # Use imported User class
                    current_app.logger.info("Logging in user")
                    login_user(user)
                    current_app.logger.info("User logged in, redirecting to index")
                    # The blueprint name is "routes" but we're using the variable "bp"
                    return redirect(url_for("routes.index"))
                else:
                    current_app.logger.info("Invalid password for user: %s", username)
            else:
                current_app.logger.info("User not found: %s", username)

            flash("Invalid username or password")
        except Exception as e:
            current_app.logger.error("Error during login: %s", str(e))
            flash("An error occurred during login. Please try again.")

    return render_template("login.html")


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("routes.login"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if current_app.db.users.find_one({"username": username}):
            flash("Username already exists")
            return redirect(url_for("routes.register"))

        current_app.db.users.insert_one(
            {
                "username": username,
                "password": generate_password_hash(password),
            }
        )

        flash("Registration successful! Please login.")
        return redirect(url_for("routes.login"))

    return render_template("register.html")


@bp.route("/api/detect-cat", methods=["POST"])
@login_required
def detect_cat():
    """API endpoint to forward browser camera images to ML container"""
    if "image" not in request.json:
        return jsonify({"error": "No image provided"}), 400

    image_data = request.json["image"]
    try:
        ml_response = requests.post(
            "http://machine-learning-client:5000/detect", 
            json={"image": image_data},
            timeout=5
        )

        ml_response.raise_for_status()
        return jsonify(ml_response.json())

    except requests.exceptions.Timeout:
        current_app.logger.error("Request to ML service timed out")
        return jsonify({"error": "Machine learning service timed out"}), 504
    except requests.exceptions.ConnectionError:
        current_app.logger.error("Could not connect to ML service")
        return jsonify({"error": "Could not connect to machine learning service"}), 503
    except Exception as e:
        current_app.logger.error('Error processing image detection: %s', e)
        return jsonify({"error": "Failed to process image"}), 500


@bp.route("/api/feeding-events")
@login_required
def get_feeding_events():
    try:
        current_app.logger.info("Fetching feeding events")
        ny_timezone = pytz.timezone("America/New_York")
        utc_timezone = pytz.utc
    except pytz.UnknownTimeZoneError:
        current_app.logger.error("Could not find timezone 'America/New_York'")
        return jsonify({"error": "Server timezone configuration error"}), 500
    now_ny = datetime.now(ny_timezone)
    today_start_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_ny.astimezone(utc_timezone)
    
    current_app.logger.info(f"Looking for events since: {today_start_utc.isoformat()}")
    
    collections = current_app.db.list_collection_names()
    current_app.logger.info(f"Available collections: {collections}")
    
    if "feeding_events" not in collections:
        current_app.logger.warning("feeding_events collection does not exist")
        return jsonify({
            "feeding_events": [],
            "total_feedings": 0,
            "last_feeding": None,
            "timestamps": [],
            "confidences": []
        })
    
    try:
        current_app.logger.info("Fetching all feeding events")
        all_events = list(current_app.db.feeding_events.find().sort("timestamp", -1))
        current_app.logger.info(f"Found {len(all_events)} total feeding events")
        
        feeding_events_utc = []
        for event in all_events:
            try:
                timestamp = event.get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    
                    if isinstance(timestamp, datetime):
                        today_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                        
                        if timestamp.date() >= today_date.date():
                            feeding_events_utc.append(event)
            except Exception as e:
                current_app.logger.error(f"Error processing event timestamp: {e}")
        
        current_app.logger.info(f"Filtered to {len(feeding_events_utc)} events from today")
        
        if feeding_events_utc:
            event = feeding_events_utc[0]
            current_app.logger.info(f"Sample event: timestamp={event.get('timestamp')}, type={event.get('type')}")
    except Exception as e:
        current_app.logger.error(f"Error querying feeding_events: {e}")
        feeding_events_utc = []

    events = []
    timestamps_iso = []
    confidences = []
    
    for event in feeding_events_utc:
        try:
            timestamp_utc = event.get("timestamp")
            if not timestamp_utc:
                current_app.logger.warning(f"Event missing timestamp: {event}")
                continue
                
            if isinstance(timestamp_utc, datetime) and timestamp_utc.tzinfo is None:
                timestamp_utc = utc_timezone.localize(timestamp_utc)
            elif not isinstance(timestamp_utc, datetime):
                current_app.logger.warning(f"Invalid timestamp format: {timestamp_utc}")
                continue
            timestamp_ny = timestamp_utc.astimezone(ny_timezone)
            timestamp_iso = timestamp_ny.isoformat()  # Use ISO format
            confidence = event.get("confidence", 0)
            if not isinstance(confidence, (int, float)):
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0
            
            events.append({
                "timestamp": timestamp_iso,
                "readable_time": timestamp_ny.strftime("%I:%M:%S %p"),
                "type": event.get("type", "unknown"),
                "confidence": confidence,
                "image": event.get("image"),
            })
            
            timestamps_iso.append(timestamp_iso)
            confidences.append(confidence)
            
        except Exception as e:
            current_app.logger.error(f"Error processing event {event.get('_id')}: {e}")
            continue

    total_feedings = len(events)
    last_feeding_iso = timestamps_iso[0] if timestamps_iso else None
    
    current_app.logger.info(f"Returning {total_feedings} events, last feeding: {last_feeding_iso}")
    
    return jsonify({
        "feeding_events": events,
        "total_feedings": total_feedings,
        "last_feeding": last_feeding_iso,
        "timestamps": timestamps_iso,
        "confidences": confidences,
    })
