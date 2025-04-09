from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import os
import random
from datetime import datetime, timedelta
import pytz
from pymongo import MongoClient
from dotenv import load_dotenv
import sys

load_dotenv()

app = Flask(__name__)

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client[os.getenv("MONGODB_DBNAME")]

CASCADE_PATH = "models/cascades"
CONFIDENCE_THRESHOLD = 0.5
DETECTION_COOLDOWN = 30  # Cooldown period in seconds


def get_utc_time():
    """Get current time in UTC"""
    return datetime.utcnow()


def load_cat_detector():
    print("Loading cat detection cascades...")

    cat_cascade_path = os.path.join(CASCADE_PATH, "haarcascade_frontalcatface.xml")
    cat_cascade_ext_path = os.path.join(
        CASCADE_PATH, "haarcascade_frontalcatface_extended.xml"
    )

    if not os.path.exists(cat_cascade_path) or not os.path.exists(cat_cascade_ext_path):
        print(f"Cascade files not found")
        print(f"Working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")
        if os.path.exists(CASCADE_PATH):
            print(f"Cascade directory contents: {os.listdir(CASCADE_PATH)}")
        return None, None

    try:
        cat_cascade = cv2.CascadeClassifier(cat_cascade_path)
        cat_cascade_ext = cv2.CascadeClassifier(cat_cascade_ext_path)

        if cat_cascade.empty() or cat_cascade_ext.empty():
            print("Failed to load cascade classifiers (empty)")
            return None, None

        print("Cascade classifiers loaded successfully")
        return cat_cascade, cat_cascade_ext
    except Exception as e:
        print(f"Error loading cascade classifiers: {e}")
        return None, None


cat_cascade, cat_cascade_ext = load_cat_detector()


"""
Key Parameters to Adjust:
scaleFactor: This parameter specifies how much the image size is reduced at each image scale. A smaller value (e.g., 1.05) means the algorithm will be more sensitive to smaller changes in scale, potentially increasing detection sensitivity but also false positives.
minNeighbors: This parameter affects the quality of the detected faces. Higher values result in fewer detections but with higher quality. Increasing this value will reduce false positives.
minSize: This parameter sets the minimum possible object size. Objects smaller than this are ignored. Adjusting this can help in focusing on larger objects, potentially reducing false positives.
Steps to Change Certainty:
Increase minNeighbors: This will make the detection stricter, reducing false positives but potentially missing some true positives.
Adjust scaleFactor: A smaller scale factor can increase sensitivity but may also increase false positives.
Modify minSize: If you want to ignore smaller detections, increase this size.
You can experiment with these parameters to find the right balance between sensitivity and specificity for your application
"""


def detect_cat_with_cascade(image):
    """Detect cats using Haar cascade classifiers with stricter parameters"""
    if cat_cascade is None or cat_cascade_ext is None:
        # No valid detector available, return false instead of random
        print("No valid cat detector available - detection will always fail")
        return False, 0.0

    # Convert to grayscale for cascade classifier
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try the primary cascade first with increased sensitivity (lower minNeighbors)
    cats = cat_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    # If no cats detected with the first cascade, try the extended one with increased sensitivity
    if len(cats) == 0:
        cats = cat_cascade_ext.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

    cat_count = len(cats)

    # Secondary verification: check color profile of detected regions
    # Cats often have distinct color patterns
    verified_count = 0
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
            # False positives often have little color/texture variation
            if roi.size > 0:
                std_dev = np.std(roi)
                if std_dev > 25:  # Threshold for variation
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
    """Convert base64 image to numpy array and preprocess"""
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


@app.route("/detect", methods=["POST"])
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
                print(
                    f"Cat detected, but within cooldown period. Skipping event logging."
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
            event = {
                "timestamp": current_time,
                "type": "feeding",
                "confidence": confidence,
                "image": image_data,
            }
            db.feeding_events.insert_one(event)
            print(
                f"Cat detected with {confidence:.2f} confidence! Event saved at {event['timestamp'].strftime('%H:%M:%S')}"
            )

            return jsonify(
                {
                    "detected": is_cat_detected,
                    "confidence": float(confidence),
                    "logged": True,
                }
            )

        return jsonify({"detected": is_cat_detected, "confidence": float(confidence)})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify(
        {
            "status": "ok",
            "ml_model": "active" if cat_cascade is not None else "fallback",
        }
    )


if __name__ == "__main__":
    print("Starting Cat Detection API server...")
    app.run(host="0.0.0.0", port=5000)
