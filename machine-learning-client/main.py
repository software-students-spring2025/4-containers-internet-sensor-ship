import os
import time
import cv2
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

# Load environment variables
load_dotenv()

class CatDetector:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv('MONGODB_URI'))
        self.db = self.mongo_client[os.getenv('MONGODB_DBNAME')]
        self.use_test_mode = os.getenv('USE_TEST_MODE', 'false').lower() == 'true'
        self.camera_id = int(os.getenv('CAMERA_ID', '0'))
        
        if not self.use_test_mode:
            try:
                self.camera = cv2.VideoCapture(self.camera_id)
                if not self.camera.isOpened():
                    print(f"Warning: Could not open camera {self.camera_id}. Falling back to test mode.")
                    self.use_test_mode = True
            except Exception as e:
                print(f"Error opening camera {self.camera_id}: {e}. Falling back to test mode.")
                self.use_test_mode = True
        
        self.model = self._load_model()
        self.last_detection_time = None
        self.detection_cooldown = 60  # seconds between detections
        
        if self.use_test_mode:
            print("Running in test mode - will simulate cat detections")
        else:
            print(f"Using camera ID: {self.camera_id}")

    def _load_model(self):
        """Load the pre-trained cat detection model"""
        try:
            return load_model('models/cat_detector.h5')
        except:
            # If no model exists, create a simple one for demonstration
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

    def preprocess_image(self, frame):
        """Preprocess the image for the model"""
        resized = cv2.resize(frame, (224, 224))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def detect_cat(self, frame=None):
        """Detect if a cat is present in the frame"""
        if self.use_test_mode:
            # In test mode, randomly detect a cat with 20% probability
            return random.random() < 0.2
            
        processed = self.preprocess_image(frame)
        prediction = self.model.predict(processed)[0][0]
        return prediction > 0.5

    def save_detection(self):
        """Save the detection event to MongoDB"""
        event = {
            'timestamp': datetime.now(),
            'type': 'feeding',
            'confidence': 1.0 if self.use_test_mode else 0.8  # In a real implementation, this would be the model's confidence
        }
        self.db.feeding_events.insert_one(event)
        print(f"Cat detection saved at {event['timestamp'].strftime('%H:%M:%S')}")

    def run(self):
        """Main detection loop"""
        print("Starting cat detection...")
        
        while True:
            if self.use_test_mode:
                # In test mode, simulate camera and detections
                time.sleep(5)  # Check every 5 seconds
                current_time = time.time()
                
                # Check if enough time has passed since last detection
                if (self.last_detection_time is None or 
                    current_time - self.last_detection_time > self.detection_cooldown):
                    
                    if self.detect_cat():
                        print("Cat detected! (Test Mode)")
                        self.save_detection()
                        self.last_detection_time = current_time
                        
            else:
                # Normal camera mode
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    time.sleep(1)
                    continue

                current_time = time.time()
                
                # Check if enough time has passed since last detection
                if (self.last_detection_time is None or 
                    current_time - self.last_detection_time > self.detection_cooldown):
                    
                    if self.detect_cat(frame):
                        print("Cat detected!")
                        self.save_detection()
                        self.last_detection_time = current_time

                # Display the frame (for debugging)
                cv2.imshow('Cat Monitor', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Clean up
        if not self.use_test_mode and hasattr(self, 'camera'):
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CatDetector()
    detector.run() 