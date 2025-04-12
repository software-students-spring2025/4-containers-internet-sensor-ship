"""
Machine learning client blueprint for cat detection in images.
This module provides REST API endpoints for detecting cats in images using OpenCV.
"""

# Standard library imports
import base64
import os
from datetime import datetime, timedelta

# Third-party imports
import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client_blueprint = Blueprint("client_blueprint", __name__)

# Set up MongoDB connection with proper parameters
try:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://mongodb:27017/")
    db_name = os.getenv("MONGODB_DBNAME", "sensor_ship")
    print(f"ML Client connecting to MongoDB at URI: {mongo_uri} with DB: {db_name}")
    
    mongo_client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000
    )
    db = mongo_client[db_name]
    
    # Test connection
    mongo_client.admin.command('ping')
    print("MongoDB connection successful!")
    
    # Ensure feeding_events collection exists
    if "feeding_events" not in db.list_collection_names():
        print("Creating feeding_events collection")
        # Create collection with a simple document that we'll remove
        db.feeding_events.insert_one({
            "timestamp": datetime.utcnow(),
            "type": "system",
            "message": "Collection initialization"
        })
        # Remove the initialization document
        db.feeding_events.delete_many({"type": "system"})
    else:
        print("feeding_events collection already exists")
        # Count documents for debugging
        count = db.feeding_events.count_documents({})
        print(f"Found {count} existing feeding events")
        
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    mongo_client = None
    db = None

CASCADE_PATH = "models/cascades"
CONFIDENCE_THRESHOLD = 0.5
DETECTION_COOLDOWN = 30  # Cooldown period in seconds


def get_utc_time():
    """Get current time in UTC"""
    return datetime.utcnow()


def load_cat_detector():
    """
    Load cat detection cascade classifiers from files.

    Returns:
        tuple: A pair of cascade classifiers or (None, None) if loading fails
    """
    print("Loading cat detection cascades...")

    cat_cascade_path = os.path.join(CASCADE_PATH, "haarcascade_frontalcatface.xml")
    cat_cascade_ext_path = os.path.join(
        CASCADE_PATH, "haarcascade_frontalcatface_extended.xml"
    )

    if not os.path.exists(cat_cascade_path) or not os.path.exists(cat_cascade_ext_path):
        print("Cascade files not found")
        print(f"Working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")
        if os.path.exists(CASCADE_PATH):
            print(f"Cascade directory contents: {os.listdir(CASCADE_PATH)}")
        return None, None

    try:
        cat_cascade_classifier = cv2.CascadeClassifier(cat_cascade_path)
        cat_cascade_ext_classifier = cv2.CascadeClassifier(cat_cascade_ext_path)

        if cat_cascade_classifier.empty() or cat_cascade_ext_classifier.empty():
            print("Failed to load cascade classifiers (empty)")
            return None, None

        print("Cascade classifiers loaded successfully")
        return cat_cascade_classifier, cat_cascade_ext_classifier
    except (cv2.error, IOError, FileNotFoundError) as e:
        print(f"Error loading cascade classifiers: {e}")
        return None, None


cat_cascade, cat_cascade_ext = load_cat_detector()


# Key Parameters documentation and explanation
DETECTION_PARAMETERS_DOCUMENTATION = """
Key Parameters to Adjust:
- scaleFactor: This parameter specifies how much the image size is reduced at each image scale. 
  A smaller value (e.g., 1.05) means the algorithm will be more sensitive to smaller changes 
  in scale, potentially increasing detection sensitivity but also false positives.
- minNeighbors: This parameter affects the quality of the detected faces. Higher values result 
  in fewer detections but with higher quality. Increasing this value will reduce false positives.
- minSize: This parameter sets the minimum possible object size. Objects smaller than this are 
  ignored. Adjusting this can help in focusing on larger objects, potentially reducing false positives.

Steps to Change Certainty:
- Increase minNeighbors: This will make the detection stricter, reducing false positives but 
  potentially missing some true positives.
- Adjust scaleFactor: A smaller scale factor can increase sensitivity but may also increase 
  false positives.
- Modify minSize: If you want to ignore smaller detections, increase this size.
"""


def detect_cat_with_cascade(image):
    """
    Detect cats using Haar cascade classifiers with stricter parameters.

    Args:
        image: OpenCV image in BGR format

    Returns:
        tuple: (is_detected, confidence_score)
    """
    if cat_cascade is None or cat_cascade_ext is None:
        # No valid detector available, return false instead of random
        print("No valid cat detector available - detection will always fail")
        return False, 0.0

    # Convert to grayscale for cascade classifier
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try both cascades with consistent parameters
    scale_factor = 1.1
    min_neighbors = 5
    min_size = (40, 40)

    # Try primary cascade
    cats_primary = cat_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
    )

    # Try extended cascade
    cats_extended = cat_cascade_ext.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
    )

    # Combine detections
    cats = list(cats_primary) + list(cats_extended)
    cat_count = len(cats)
    verified_count = 0

    # Verification threshold
    variation_threshold = 25

    # Secondary verification: check color profile of detected regions
    if cat_count > 0:
        for x, y, w, h in cats:
            # Expand region slightly to get more context
            x_exp = max(0, x - 10)
            y_exp = max(0, y - 10)
            w_exp = min(image.shape[1] - x_exp, w + 20)
            h_exp = min(image.shape[0] - y_exp, h + 20)

            # Get the region
            roi = image[y_exp : y_exp + h_exp, x_exp : x_exp + w_exp]

            # Simple heuristic: check if region has enough variation
            if roi.size > 0 and np.std(roi) > variation_threshold:
                verified_count += 1

    # Only consider it a cat if at least one detection passed verification
    if verified_count > 0:
        # More verified detections = higher confidence, but cap it lower than before
        confidence = min(0.5 + (verified_count * 0.05), 0.8)
        print(
            f"Detected {verified_count} verified cat features with confidence: {confidence:.2f}"
        )
        return True, confidence

    # No cats detected with high confidence
    return False, 0.1


def preprocess_image(image_data):
    """
    Convert base64 image to numpy array and preprocess.

    Args:
        image_data: Base64 encoded image data

    Returns:
        numpy.ndarray: Decoded image as OpenCV array

    Raises:
        ValueError: If image decoding fails
    """
    # Remove data URL prefix if present
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # Decode base64 image
    image_bytes = base64.b64decode(image_data)

    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    return img


@client_blueprint.route("/detect", methods=["POST"])
def detect():
    """Endpoint to detect cats in images"""
    if "image" not in request.json:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Get image data
        image_data = request.json["image"]

        # Process the image
        img = preprocess_image(image_data)

        # Detect cat in the image
        is_cat_detected, confidence = detect_cat_with_cascade(img)

        # If cat is detected with sufficient confidence, check cooldown before saving
        if is_cat_detected:
            # Check if MongoDB is available
            if db is None or mongo_client is None:
                print("Cannot log detection: MongoDB connection unavailable")
                return jsonify({
                    "detected": is_cat_detected,
                    "confidence": float(confidence),
                    "logged": False,
                    "message": "Detection not logged due to database unavailability"
                })
                
            current_time = get_utc_time()
            cooldown_time = current_time - timedelta(seconds=DETECTION_COOLDOWN)

            # Check for recent detections within cooldown period
            recent_detection = db.feeding_events.find_one(
                {"timestamp": {"$gte": cooldown_time}}
            )

            if recent_detection:
                # Detection within cooldown period exists, don't save a new one
                print(
                    "Cat detected, but within cooldown period. Skipping event logging."
                )
                return jsonify(
                    {
                        "detected": is_cat_detected,
                        "confidence": float(confidence),
                        "logged": False,
                        "message": "Detection not logged due to cooldown",
                    }
                )

            # No recent detection, save the event
            try:
                # Get current time and ensure it's in the proper format
                current_time = get_utc_time()
                print(f"Current time (UTC): {current_time.isoformat()}")
                
                # Create a simplified event - don't include the full image which is large
                # Keep a smaller version for display purposes
                image_for_storage = None
                try:
                    # If the image is a data URL, create a proper thumbnail
                    if image_data and image_data.startswith('data:image'):
                        # Extract the base64 data and image type
                        image_parts = image_data.split(',')
                        image_prefix = image_parts[0] + ','  # Keep the data:image/jpeg;base64, part
                        image_base64 = image_parts[1]
                        
                        # Decode the base64 image
                        image_bytes = base64.b64decode(image_base64)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            # Resize image to a small thumbnail (100x75)
                            thumbnail = cv2.resize(img, (100, 75), interpolation=cv2.INTER_AREA)
                            
                            # Encode to jpeg with lower quality
                            _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            
                            # Convert back to base64
                            thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Create the full data URL
                            image_for_storage = f"data:image/jpeg;base64,{thumbnail_base64}"
                            print("Created proper thumbnail image")
                        else:
                            print("Failed to decode image for thumbnail creation")
                    else:
                        print("No valid image data to save")
                except Exception as img_error:
                    print(f"Error processing image for storage: {img_error}")
                
                # Prepare event document with proper data types
                event = {
                    "timestamp": current_time,           # Use Python datetime object
                    "type": "feeding",                   # String
                    "confidence": float(confidence),     # Float
                    "image": image_for_storage           # String or None
                }
                
                # Log the event we're about to save
                print(f"Saving detection to MongoDB collection: feeding_events in database {db.name}")
                print(f"Event timestamp: {event['timestamp'].isoformat()}, type: {event['type']}")
                
                # Insert the document
                result = db.feeding_events.insert_one(event)
                inserted_id = result.inserted_id
                print(f"Detection saved with ID: {inserted_id}")
                
                # Verify the document was saved by retrieving it
                saved_event = db.feeding_events.find_one({"_id": inserted_id})
                if saved_event:
                    print(f"Successfully verified saved event with ID: {inserted_id}")
                    saved_time = saved_event.get("timestamp")
                    print(f"Saved timestamp: {saved_time}")
                else:
                    print(f"Warning: Could not verify saved event with ID: {inserted_id}")
                
                # Log the detection with confidence
                print(f"Cat detected with {confidence:.2f} confidence! "
                      f"Event saved at {event['timestamp'].strftime('%H:%M:%S')}")
                
                # Return success response
                return jsonify({
                    "detected": True,
                    "confidence": float(confidence),
                    "logged": True,
                    "message": "Detection logged successfully",
                    "timestamp": current_time.isoformat(),
                    "event_id": str(inserted_id)
                })
            except Exception as e:
                print(f"Error saving detection to MongoDB: {e}")
                return jsonify(
                    {
                        "detected": is_cat_detected,
                        "confidence": float(confidence),
                        "logged": False,
                        "error": str(e)
                    }
                )

        return jsonify({"detected": is_cat_detected, "confidence": float(confidence)})

    except ValueError as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Image processing error: {str(e)}"}), 400
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return jsonify({"error": f"Image analysis error: {str(e)}"}), 500
    except (
        MongoClient.ServerSelectionTimeoutError,
        MongoClient.ConnectionFailure,
    ) as e:
        print(f"Database connection error: {e}")
        return jsonify({"error": "Database connection error"}), 503


@client_blueprint.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify(
        {
            "status": "ok",
            "ml_model": "active" if cat_cascade is not None else "fallback",
        }
    )
