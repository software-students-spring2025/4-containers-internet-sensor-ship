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

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client[os.getenv("MONGODB_DBNAME")]

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
            roi = image[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp]

            # Simple heuristic: check if region has enough variation
            if roi.size > 0 and np.std(roi) > variation_threshold:
                verified_count += 1

    # Only consider it a cat if at least one detection passed verification
    if verified_count > 0:
        # More verified detections = higher confidence, but cap it lower than before
        confidence = min(0.5 + (verified_count * 0.05), 0.8)
        print(f"Detected {verified_count} verified cat features with confidence: {confidence:.2f}")
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
            current_time = get_utc_time()
            cooldown_time = current_time - timedelta(seconds=DETECTION_COOLDOWN)

            # Check for recent detections within cooldown period
            recent_detection = db.feeding_events.find_one(
                {"timestamp": {"$gte": cooldown_time}}
            )

            if recent_detection:
                # Detection within cooldown period exists, don't save a new one
                print("Cat detected, but within cooldown period. Skipping event logging.")
                return jsonify({
                    "detected": is_cat_detected,
                    "confidence": float(confidence),
                    "logged": False,
                    "message": "Detection not logged due to cooldown",
                })

            # No recent detection, save the event
            event = {
                "timestamp": current_time,
                "type": "feeding",
                "confidence": confidence,
                "image": image_data,
            }
            db.feeding_events.insert_one(event)
            print(
                f"Cat detected with {confidence:.2f} confidence! "
                f"Event saved at {event['timestamp'].strftime('%H:%M:%S')}"
            )

            return jsonify({
                "detected": is_cat_detected,
                "confidence": float(confidence),
                "logged": True,
            })

        return jsonify({"detected": is_cat_detected, "confidence": float(confidence)})

    except ValueError as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Image processing error: {str(e)}"}), 400
    except (cv2.error, np.Error) as e:
        print(f"OpenCV/NumPy error: {e}")
        return jsonify({"error": f"Image analysis error: {str(e)}"}), 500
    except (MongoClient.ServerSelectionTimeoutError, MongoClient.ConnectionFailure) as e:
        print(f"Database connection error: {e}")
        return jsonify({"error": "Database connection error"}), 503


@client_blueprint.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "ml_model": "active" if cat_cascade is not None else "fallback",
    })
