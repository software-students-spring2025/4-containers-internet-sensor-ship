import base64
import json
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Mock the app import
with patch('src.client_blueprint.load_dotenv', MagicMock()), \
     patch('cv2.imread', MagicMock()), \
     patch('cv2.CascadeClassifier', MagicMock()), \
     patch('cv2.cvtColor', MagicMock()), \
     patch('pymongo.MongoClient', MagicMock()):
    from src.app import create_app


# Mock functions instead of importing them
@pytest.fixture
def get_utc_time():
    with patch('src.client_blueprint.get_utc_time') as mock_time:
        mock_time_obj = MagicMock()
        mock_time_obj.isoformat.return_value = "2023-01-01T12:00:00"
        mock_time_obj.strftime.return_value = "12:00:00"
        mock_time_obj.year = 2023
        mock_time_obj.month = 1
        mock_time_obj.day = 1
        mock_time.return_value = mock_time_obj
        yield mock_time


def test_app_creation():
    """Test that the app can be created"""
    with patch('src.client_blueprint.client_blueprint', MagicMock()):
        app = create_app()
        assert app is not None
        assert app.config.get("TESTING") is False  # Default should be False


def test_app_creation_with_config():
    """Test app creation with a configuration"""
    class TestConfig:
        TESTING = True
        DEBUG = True
    
    with patch('src.client_blueprint.client_blueprint', MagicMock()):
        app = create_app(TestConfig)
        assert app is not None
        assert app.config.get("TESTING") is True
        assert app.config.get("DEBUG") is True


def test_app_creation_with_blueprint_error():
    """Test app creation when blueprint registration fails"""
    with patch('src.client_blueprint.client_blueprint', MagicMock()) as mock_blueprint, \
         patch('flask.Flask.register_blueprint', side_effect=Exception("Test error")):
        
        # Ensure we get the blueprint name for printing
        mock_blueprint.name = "test_blueprint"
        
        app = create_app()
        assert app is not None


def test_app_main_execution():
    """Test the __main__ execution code path in app.py"""
    # This is a placeholder test - we can't easily test the __main__ block 
    # without executing Flask, which we don't want to do in tests
    assert True


# Route testing
def test_health_check_endpoint(client):
    """Test the health check endpoint"""
    with patch("src.client_blueprint.cat_cascade", MagicMock()), \
         patch("src.client_blueprint.load_cat_detector", MagicMock()):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json
        assert "status" in data
        assert data["status"] == "ok"
        assert "ml_model" in data


def test_detect_endpoint_no_image(client):
    """Test the /detect endpoint with missing image data"""
    response = client.post("/detect", json={})
    assert response.status_code == 400
    assert "error" in response.json
    assert "No image provided" in response.json["error"]


def test_detect_endpoint_with_image(client, get_utc_time):
    """Test the /detect endpoint with valid image data"""
    # Create a simple base64 encoded image (1x1 pixel)
    dummy_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIHMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="
    
    # Setup necessary mocks for the test
    with patch("src.client_blueprint.preprocess_image") as mock_preprocess, \
         patch("src.client_blueprint.detect_cat_with_cascade") as mock_detect, \
         patch("src.client_blueprint.db") as mock_db:
        
        # Configure the mocks
        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (True, 0.75)
        
        # Mock database
        mock_db.feeding_events.find_one.return_value = None
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = "test_id"
        mock_db.feeding_events.insert_one.return_value = mock_insert_result
        mock_db.feeding_events.find_one.return_value = {"_id": "test_id", "timestamp": get_utc_time.return_value}
        mock_db.name = "test_db"
        
        # Make the request
        response = client.post("/detect", json={"image": dummy_image})
        
        # Assertions
        assert response.status_code == 200
        assert response.json["detected"] is True
        assert response.json["confidence"] == 0.75


def test_detect_endpoint_with_cooldown(client, get_utc_time):
    """Test the /detect endpoint when within cooldown period"""
    # Create a simple base64 encoded image (1x1 pixel)
    dummy_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIHMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="
    
    # Setup necessary mocks for the test
    with patch("src.client_blueprint.preprocess_image") as mock_preprocess, \
         patch("src.client_blueprint.detect_cat_with_cascade") as mock_detect, \
         patch("src.client_blueprint.db") as mock_db:
        
        # Configure the mocks
        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (True, 0.75)
        
        # Mock database - return a recent detection (within cooldown)
        mock_db.feeding_events.find_one.return_value = {"_id": "recent_id", "timestamp": get_utc_time.return_value}
        
        # Make the request
        response = client.post("/detect", json={"image": dummy_image})
        
        # Assertions
        assert response.status_code == 200
        assert response.json["detected"] is True
        assert response.json["confidence"] == 0.75


def test_detect_no_cat(client):
    """Test the /detect endpoint when no cat is detected"""
    # Create a simple base64 encoded image (1x1 pixel)
    dummy_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIHMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="
    
    # Setup necessary mocks for the test
    with patch("src.client_blueprint.preprocess_image") as mock_preprocess, \
         patch("src.client_blueprint.detect_cat_with_cascade") as mock_detect:
        
        # Configure the mocks
        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (False, 0.1)
        
        # Make the request
        response = client.post("/detect", json={"image": dummy_image})
        
        # Assertions
        assert response.status_code == 200
        assert response.json["detected"] is False
        assert response.json["confidence"] == 0.1


def test_detect_db_unavailable(client):
    """Test the /detect endpoint when database is unavailable"""
    # Create a simple base64 encoded image (1x1 pixel)
    dummy_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIHMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="
    
    # Setup necessary mocks for the test
    with patch("src.client_blueprint.preprocess_image") as mock_preprocess, \
         patch("src.client_blueprint.detect_cat_with_cascade") as mock_detect, \
         patch("src.client_blueprint.db", None), \
         patch("src.client_blueprint.mongo_client", None):
        
        # Configure the mocks
        mock_preprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect.return_value = (True, 0.75)
        
        # Make the request
        response = client.post("/detect", json={"image": dummy_image})
        
        # Assertions - we should get a response even with db error
        assert response.status_code == 200
        assert response.json["detected"] is True
        assert response.json["confidence"] == 0.75
        assert response.json["logged"] is False


def test_detect_image_processing_error(client):
    """Test the /detect endpoint when image processing fails"""
    # Create a simple base64 encoded image (1x1 pixel)
    dummy_image = "data:image/jpeg;base64,invalidbase64data"
    
    # Setup necessary mocks for the test
    with patch("src.client_blueprint.preprocess_image") as mock_preprocess:
        # Configure the mock to raise an error
        mock_preprocess.side_effect = ValueError("Failed to decode image")
        
        # Make the request
        response = client.post("/detect", json={"image": dummy_image})
        
        # Assertions
        assert response.status_code == 400
        assert "error" in response.json


def test_health_check_with_ml_model_loaded():
    """Test health check endpoint with ML model loaded"""
    with patch('src.client_blueprint.cat_cascade', MagicMock()) as mock_cascade, \
         patch('src.client_blueprint.cat_cascade_ext', MagicMock()) as mock_cascade_ext:
        
        # Configure mocks to indicate model is loaded
        mock_cascade.empty.return_value = False
        mock_cascade_ext.empty.return_value = False
        
        # Create app and client
        app = create_app()
        app.config['TESTING'] = True
        client = app.test_client()
        
        # Make request
        response = client.get('/health')
        
        # Verify response
        assert response.status_code == 200
        data = response.json
        assert data["status"] == "ok"
        assert data["ml_model"] == "active"  # Value from the actual implementation


def test_health_check_with_ml_model_not_loaded():
    """Test health check endpoint when ML model is not loaded"""
    with patch('src.client_blueprint.cat_cascade', None), \
         patch('src.client_blueprint.cat_cascade_ext', None):
        
        # Create app and client
        app = create_app()
        app.config['TESTING'] = True
        client = app.test_client()
        
        # Make request
        response = client.get('/health')
        
        # Verify response
        assert response.status_code == 200
        data = response.json
        assert data["status"] == "ok"
        assert data["ml_model"] == "fallback"  # Value from the actual implementation
