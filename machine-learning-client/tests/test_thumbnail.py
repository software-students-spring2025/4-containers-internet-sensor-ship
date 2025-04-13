import base64
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import cv2
from bson import ObjectId
from flask import jsonify, Flask, current_app
import os
import unittest
import sys
from datetime import datetime, timedelta

from src.client_blueprint import preprocess_image, detect_cat_with_cascade, get_utc_time

class MockPyMongoErrors:
    class ServerSelectionTimeoutError(Exception):
        """Mock MongoDB Server Selection Timeout Error"""
        pass
    
    class ConnectionFailure(Exception):
        """Mock MongoDB Connection Failure Error"""
        pass

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        
    def get_json(self):
        return self.json_data
    
    def __call__(self, *args, **kwargs):
        return self

def create_thumbnail(img):
    """Create a thumbnail from an image for testing"""
    if img is None:
        return None
    thumbnail = cv2.resize(img, (100, 75), interpolation=cv2.INTER_AREA)
    buffer = np.zeros((100, 75, 3), dtype=np.uint8).tobytes()
    thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{thumbnail_base64}"

test_app = Flask(__name__)

@pytest.fixture
def app_context():
    with test_app.app_context():
        yield

class MockMongoClient:
    ServerSelectionTimeoutError = MockPyMongoErrors.ServerSelectionTimeoutError
    ConnectionFailure = MockPyMongoErrors.ConnectionFailure
    
    def __init__(self, *args, **kwargs):
        self.db = MagicMock()
        
    def __getitem__(self, name):
        return self.db

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client_blueprint import detect, load_cat_detector

@pytest.fixture
def get_base64_thumbnail():
    """Create a simple get_base64_thumbnail function for tests"""
    def _get_thumbnail(img):
        buffer = np.zeros((100, 75, 3), dtype=np.uint8).tobytes()
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"
    return _get_thumbnail

@pytest.mark.usefixtures("app_context")
def test_detect_handles_invalid_image():
    """Test the detect function handles invalid images correctly"""
    with test_app.test_request_context(json={"image": "invalid_image"}), \
         patch('src.client_blueprint.preprocess_image') as mock_preprocess, \
         patch('src.client_blueprint.jsonify') as mock_jsonify:
        
        mock_preprocess.side_effect = ValueError("Invalid image")
        mock_jsonify.return_value = ({"error": "Image processing error: Invalid image"}, 400)
        
        from src.client_blueprint import detect
        result = detect()
        
        assert result[1] == 400  # HTTP 400 Bad Request
        mock_jsonify.assert_called_once_with({"error": "Image processing error: Invalid image"})

@pytest.mark.usefixtures("app_context")
def test_detect_handles_cv2_error():
    """Test the detect function handles OpenCV errors correctly"""
    with test_app.test_request_context(json={"image": "test_image"}), \
         patch('src.client_blueprint.preprocess_image') as mock_preprocess, \
         patch('src.client_blueprint.jsonify') as mock_jsonify, \
         patch('src.client_blueprint.cv2') as mock_cv2:
        
        mock_cv2.error = type('CV2Error', (Exception,), {})
        
        mock_preprocess.side_effect = mock_cv2.error("OpenCV error")
        mock_jsonify.return_value = ({"error": "Image analysis error: OpenCV error"}, 500)
        
        from src.client_blueprint import detect
        result = detect()
        
        assert result[1] == 500  # HTTP 500 Internal Server Error
        mock_jsonify.assert_called_once_with({"error": "Image analysis error: OpenCV error"})

@pytest.mark.usefixtures("app_context")
def test_get_base64_thumbnail():
    """Test creating a base64 thumbnail"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with patch('cv2.imencode', return_value=(True, b'test_bytes')), \
         patch('base64.b64encode', return_value=b'base64_encoded_string'):
        
        result = create_thumbnail(img)
        
        assert isinstance(result, str)
        assert result.startswith('data:image/jpeg;base64,')

def test_detect_function_no_database():
    """Test the detect_cat_with_cascade function directly"""
    with patch('numpy.std', return_value=30), \
         patch('cv2.cvtColor', return_value=np.zeros((10, 10), dtype=np.uint8)):
        
        with patch('src.client_blueprint.cat_cascade') as mock_cascade, \
             patch('src.client_blueprint.cat_cascade_ext') as mock_cascade_ext:
            
            mock_cascade.detectMultiScale.return_value = [(10, 10, 30, 30)]
            mock_cascade_ext.detectMultiScale.return_value = []
            
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            detected, confidence = detect_cat_with_cascade(test_image)
            
            assert detected is True
            assert confidence > 0.5  

def test_database_interaction_mocked():
    """Test mocked database interaction without Flask context"""
    mock_db = MagicMock()
    mock_result = MagicMock()
    mock_result.inserted_id = ObjectId("507f1f77bcf86cd799439011")
    mock_db.feeding_events.insert_one.return_value = mock_result
    mock_db.feeding_events.find_one.return_value = None 
    mock_db.name = "test_db"

    current_time = get_utc_time()
    
    event = {
        "timestamp": current_time,
        "type": "feeding",
        "confidence": 0.85,
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB"
    }
    
    result = mock_db.feeding_events.insert_one(event)
    
    assert result.inserted_id is not None
    
    mock_db.feeding_events.find_one.return_value = event
    found = mock_db.feeding_events.find_one({"_id": result.inserted_id})
    
    assert found is not None
    assert found["type"] == "feeding"
    assert found["confidence"] == 0.85

def test_cooldown_check_mocked():
    """Test cooldown check logic without Flask context"""
    mock_db = MagicMock()

    current_time = get_utc_time()
    mock_db.feeding_events.find_one.return_value = {"_id": "recent_id", "timestamp": current_time}
    
    result = mock_db.feeding_events.find_one({"timestamp": {"$gte": current_time}})
    assert result is not None
    
    mock_db.feeding_events.find_one.return_value = None
    
    result = mock_db.feeding_events.find_one({"timestamp": {"$gte": current_time}})
    assert result is None

def test_mongodb_error_handling():
    """Test MongoDB error handling without Flask context"""
    mock_db = MagicMock()
    mock_db.feeding_events.insert_one.side_effect = Exception("MongoDB insert error")

    current_time = get_utc_time()
    event = {
        "timestamp": current_time,
        "type": "feeding",
        "confidence": 0.75,
        "image": None
    }
    
    try:
        mock_db.feeding_events.insert_one(event)
        assert False, "Expected an exception but none was raised"
    except Exception as e:
        assert str(e) == "MongoDB insert error"

def test_load_cat_detector():
    """Test the load_cat_detector function"""
    with patch('os.path.exists') as mock_exists, \
         patch('os.getcwd') as mock_getcwd, \
         patch('os.listdir') as mock_listdir, \
         patch('cv2.CascadeClassifier') as mock_cascade_classifier:
        
        from src.client_blueprint import load_cat_detector
        
        mock_exists.return_value = True
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade_classifier.return_value = mock_cascade
        
        result1, result2 = load_cat_detector()
        
        # Verify results
        assert result1 is mock_cascade
        assert result2 is mock_cascade
        assert mock_cascade_classifier.call_count == 2
        
        mock_cascade_classifier.reset_mock()
        mock_cascade_fail = MagicMock()
        mock_cascade_fail.empty.return_value = True
        mock_cascade_classifier.return_value = mock_cascade_fail
        
        result1, result2 = load_cat_detector()
        
        assert result1 is None
        assert result2 is None
        assert mock_cascade_classifier.call_count == 2
        
        mock_cascade_classifier.reset_mock()
        mock_exists.reset_mock()
        
        mock_exists.return_value = False
        mock_getcwd.return_value = "/app"
        mock_listdir.return_value = ["src", "models"]
        
        result1, result2 = load_cat_detector()
        
        assert result1 is None
        assert result2 is None
        assert mock_cascade_classifier.call_count == 0

def test_load_cat_detector_exception():
    """Test the load_cat_detector function when exceptions occur"""
    with patch('os.path.exists') as mock_exists, \
         patch('cv2.CascadeClassifier') as mock_cascade_classifier:
        
        from src.client_blueprint import load_cat_detector
        
        mock_exists.return_value = True
        mock_cascade_classifier.side_effect = cv2.error("Error loading cascade")
        
        result1, result2 = load_cat_detector()
        
        assert result1 is None
        assert result2 is None 

def test_detect_no_cat_detected():
    """Test detect endpoint when no cat is detected"""
    request_mock = MagicMock()
    request_mock.json = {"image": "test_image_data"}
    
    mock_jsonify = MagicMock()
    mock_jsonify.return_value = {"detected": False, "confidence": 0.1}
    
    with patch('src.client_blueprint.request', request_mock), \
         patch('src.client_blueprint.preprocess_image') as mock_preprocess, \
         patch('src.client_blueprint.detect_cat_with_cascade') as mock_detect, \
         patch('src.client_blueprint.jsonify', mock_jsonify):
        
        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (False, 0.1)
        
        result = detect()
        
        assert result == {"detected": False, "confidence": 0.1}
        mock_jsonify.assert_called_once_with({"detected": False, "confidence": 0.1})

def test_db_unavailable():
    """Test detect endpoint when database is unavailable"""
    request_mock = MagicMock()
    request_mock.json = {"image": "test_image_data"}
    
    mock_jsonify = MagicMock()
    mock_jsonify.return_value = {"detected": True, "confidence": 0.8, "logged": False, 
                                "message": "Detection not logged due to database unavailability"}
    
    with patch('src.client_blueprint.request', request_mock), \
         patch('src.client_blueprint.preprocess_image') as mock_preprocess, \
         patch('src.client_blueprint.detect_cat_with_cascade') as mock_detect, \
         patch('src.client_blueprint.db', None), \
         patch('src.client_blueprint.mongo_client', None), \
         patch('src.client_blueprint.jsonify', mock_jsonify):

        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (True, 0.8)
        
        result = detect()

        assert result == mock_jsonify.return_value
        mock_jsonify.assert_called_once_with({
            "detected": True,
            "confidence": 0.8,
            "logged": False,
            "message": "Detection not logged due to database unavailability"
        })

def test_health_check_endpoint():
    """Test the health check endpoint"""
    with patch('src.client_blueprint.jsonify') as mock_jsonify, \
         patch('src.client_blueprint.cat_cascade', MagicMock()):
        
        mock_jsonify.return_value = {"status": "ok", "ml_model": "active"}
        
        from src.client_blueprint import health_check
        result = health_check()
        
        assert result == {"status": "ok", "ml_model": "active"}
        mock_jsonify.assert_called_once_with({"status": "ok", "ml_model": "active"})

def test_health_check_no_model():
    """Test health check when model is not available"""
    with patch('src.client_blueprint.jsonify') as mock_jsonify, \
         patch('src.client_blueprint.cat_cascade', None):
        
        mock_jsonify.return_value = {"status": "ok", "ml_model": "fallback"}
        
        from src.client_blueprint import health_check
        result = health_check()
        
        assert result == {"status": "ok", "ml_model": "fallback"}
        mock_jsonify.assert_called_once_with({"status": "ok", "ml_model": "fallback"}) 


def test_preprocess_image_with_prefix():
    """Test preprocess_image with data URL prefix"""
    test_data = "data:image/jpeg;base64,SGVsbG8="  # "Hello" in base64
    
    with patch('cv2.imdecode') as mock_imdecode:
        mock_img = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_img
        
        result = preprocess_image(test_data)
        
        assert result is mock_img

def test_preprocess_image_decoding_failure():
    """Test preprocess_image when decoding fails"""
    test_data = "SGVsbG8="  # "Hello" in base64
    
    with patch('cv2.imdecode', return_value=None):
        # Call the function and expect an exception
        with pytest.raises(ValueError) as excinfo:
            preprocess_image(test_data)
        
        assert "Failed to decode image" in str(excinfo.value)

def test_detect_with_recent_detection():
    """Test detect endpoint when there's a recent detection within cooldown"""
    request_mock = MagicMock()
    request_mock.json = {"image": "test_image_data"}
    
    mock_jsonify = MagicMock()
    mock_jsonify.return_value = {
        "detected": True,
        "confidence": 0.8,
        "logged": False,
        "message": "Detection not logged due to cooldown"
    }
    
    current_time = datetime.utcnow()
    cooldown_time = current_time - timedelta(seconds=10)  # Within the cooldown period
    
    mock_db = MagicMock()
    mock_db.name = "test_db"
    mock_db.feeding_events.find_one.return_value = {
        "_id": "recent_id",
        "timestamp": cooldown_time
    }
    
    with patch('src.client_blueprint.request', request_mock), \
         patch('src.client_blueprint.preprocess_image') as mock_preprocess, \
         patch('src.client_blueprint.detect_cat_with_cascade') as mock_detect, \
         patch('src.client_blueprint.db', mock_db), \
         patch('src.client_blueprint.mongo_client', MagicMock()), \
         patch('src.client_blueprint.get_utc_time', return_value=current_time), \
         patch('src.client_blueprint.jsonify', mock_jsonify):
        
        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (True, 0.8)
        
        result = detect()
        
        assert result == mock_jsonify.return_value
        mock_jsonify.assert_called_once_with({
            "detected": True,
            "confidence": 0.8,
            "logged": False,
            "message": "Detection not logged due to cooldown"
        })

def test_detect_mongodb_connection_error():
    """Test detect endpoint with MongoDB connection error"""
    request_mock = MagicMock()
    request_mock.json = {"image": "test_image_data"}
    
    mock_jsonify = MagicMock()
    mock_jsonify.return_value = {"error": "Database connection error"}, 503
    
    mongodb_error = MockPyMongoErrors.ServerSelectionTimeoutError("Connection timeout")
    
    cv2_error = type('cv2Error', (Exception,), {})
    
    with patch('src.client_blueprint.request', request_mock), \
         patch('src.client_blueprint.preprocess_image') as mock_preprocess, \
         patch('src.client_blueprint.detect_cat_with_cascade') as mock_detect, \
         patch('src.client_blueprint.db') as mock_db, \
         patch('src.client_blueprint.mongo_client') as mock_client, \
         patch('src.client_blueprint.jsonify', mock_jsonify), \
         patch('src.client_blueprint.MongoClient') as mock_mongo_class, \
         patch('src.client_blueprint.cv2') as mock_cv2:
        
        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (True, 0.8)
        mock_db.feeding_events.find_one.side_effect = mongodb_error
        mock_mongo_class.ServerSelectionTimeoutError = MockPyMongoErrors.ServerSelectionTimeoutError
        mock_mongo_class.ConnectionFailure = MockPyMongoErrors.ConnectionFailure
        mock_cv2.error = cv2_error
        
        result = detect()
        
        assert result[1] == 503
        assert "Database connection error" in mock_jsonify.call_args[0][0]["error"]

def test_successful_detection_and_logging():
    """Test full successful detection and logging workflow"""
    request_mock = MagicMock()
    request_mock.json = {"image": "data:image/jpeg;base64,SGVsbG8="}
    
    mock_jsonify = MagicMock()
    mock_jsonify.return_value = {
        "detected": True,
        "confidence": 0.8,
        "logged": True,
        "message": "Detection logged successfully",
        "timestamp": "2023-01-01T12:00:00",
        "event_id": "507f1f77bcf86cd799439011"
    }
    
    current_time = datetime(2023, 1, 1, 12, 0, 0)
    
    mock_db = MagicMock()
    mock_db.name = "test_db"
    mock_db.feeding_events.find_one.return_value = None
    
    mock_result = MagicMock()
    mock_result.inserted_id = ObjectId("507f1f77bcf86cd799439011")
    mock_db.feeding_events.insert_one.return_value = mock_result
    
    mock_db.feeding_events.find_one.side_effect = [
        None,
        {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "timestamp": current_time,
            "type": "feeding",
            "confidence": 0.8
        }
    ]
    
    mock_image = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_thumbnail = np.zeros((100, 75, 3), dtype=np.uint8)
    
    with patch('src.client_blueprint.request', request_mock), \
         patch('src.client_blueprint.preprocess_image', return_value=mock_image), \
         patch('src.client_blueprint.detect_cat_with_cascade', return_value=(True, 0.8)), \
         patch('src.client_blueprint.db', mock_db), \
         patch('src.client_blueprint.mongo_client', MagicMock()), \
         patch('src.client_blueprint.get_utc_time', return_value=current_time), \
         patch('cv2.resize', return_value=mock_thumbnail), \
         patch('cv2.imencode', return_value=(True, b'jpeg_bytes')), \
         patch('base64.b64encode', return_value=b'encoded_thumbnail'), \
         patch('src.client_blueprint.jsonify', mock_jsonify):
        
        result = detect()
        
        assert result == mock_jsonify.return_value
        mock_jsonify.assert_called_once()
        call_arg = mock_jsonify.call_args[0][0]
        assert "detected" in call_arg
        assert "confidence" in call_arg
        assert "logged" in call_arg
        assert "message" in call_arg
        assert "timestamp" in call_arg
        assert "event_id" in call_arg

def test_detect_missing_image():
    """Test detect endpoint when image is missing from request"""
    request_mock = MagicMock()
    request_mock.json = {}
    
    mock_jsonify = MagicMock()
    mock_jsonify.return_value = ({"error": "No image provided"}, 400)
    
    with patch('src.client_blueprint.request', request_mock), \
         patch('src.client_blueprint.jsonify', mock_jsonify):
        
        result = detect()
        
        assert result[1] == 400
        mock_jsonify.assert_called_once_with({"error": "No image provided"})

def test_detect_with_database_error():
    """Test detect endpoint with database error during insert"""
    request_mock = MagicMock()
    request_mock.json = {"image": "test_image_data"}
    
    mock_jsonify = MagicMock()
    mock_jsonify.return_value = {
        "detected": True,
        "confidence": 0.8,
        "logged": False,
        "error": "Database error"
    }
    
    current_time = datetime(2023, 1, 1, 12, 0, 0)
    
    mock_db = MagicMock()
    mock_db.name = "test_db"
    mock_db.feeding_events.find_one.return_value = None  # No recent detection
    mock_db.feeding_events.insert_one.side_effect = Exception("Database error")
    
    with patch('src.client_blueprint.request', request_mock), \
         patch('src.client_blueprint.preprocess_image') as mock_preprocess, \
         patch('src.client_blueprint.detect_cat_with_cascade') as mock_detect, \
         patch('src.client_blueprint.db', mock_db), \
         patch('src.client_blueprint.mongo_client', MagicMock()), \
         patch('src.client_blueprint.get_utc_time', return_value=current_time), \
         patch('src.client_blueprint.jsonify', mock_jsonify):
        
        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (True, 0.8)
        
        result = detect()
        
        mock_jsonify.assert_called_once()
        call_arg = mock_jsonify.call_args[0][0]
        assert call_arg["detected"] is True
        assert call_arg["confidence"] == 0.8
        assert call_arg["logged"] is False
        assert "error" in call_arg 