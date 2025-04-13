import unittest
from unittest.mock import patch, MagicMock, Mock
import json
from datetime import datetime, timedelta
import pytz
from flask import url_for, flash, current_app
import pytest
from bson.objectid import ObjectId
import requests
from werkzeug.exceptions import InternalServerError

# Create test for routes.py
@patch('flask_login.utils._get_user')
def test_index_route(mock_get_user, client, app, mock_db):
    """Test the index route with a logged-in user"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the database response
    mock_events = [
        {
            "_id": ObjectId(),
            "timestamp": datetime.now(),
            "type": "cat_eating",
            "confidence": 0.95,
            "image": "base64_image_data"
        }
    ]
    
    # Mock the find method for the feeding_events collection
    mock_find = MagicMock()
    mock_find.sort.return_value = mock_events
    mock_db.feeding_events.find.return_value = mock_find
    
    # Mock the list_collection_names method
    mock_db.list_collection_names.return_value = ["feeding_events"]
    
    with app.app_context():
        app.db = mock_db
        # Make the request
        response = client.get('/')
        
        # Check the response
        assert response.status_code == 200
        assert b'feeding_events' in response.data or b'total_feedings' in response.data

@patch('flask_login.utils._get_user')
def test_index_route_no_events(mock_get_user, client, app, mock_db):
    """Test the index route with no feeding events collection"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the list_collection_names method to not include feeding_events
    mock_db.list_collection_names.return_value = ["users"]
    
    with app.app_context():
        app.db = mock_db
        # Make the request
        response = client.get('/')
        
        # Check the response
        assert response.status_code == 200
        assert b'total_feedings' in response.data

@patch('flask_login.utils._get_user')
def test_index_route_db_error(mock_get_user, client, app, mock_db):
    """Test the index route with a database error"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the list_collection_names method to raise an exception
    mock_db.list_collection_names.side_effect = Exception("Database error")
    
    with app.app_context():
        app.db = mock_db
        # Make the request
        response = client.get('/')
        
        # Check the response
        assert response.status_code == 200
        assert b'error' in response.data

def test_login_route_get(client):
    """Test the login route GET request"""
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Login' in response.data

def test_login_route_post_success(client, app, mock_db):
    """Test the login route POST request with valid credentials"""
    # Mock the database response
    mock_user = {
        "_id": ObjectId(),
        "username": "testuser",
        "password": "hashed_password"
    }
    
    # Mock the find_one method for the users collection
    mock_db.users.find_one.return_value = mock_user
    
    # Mock the check_password_hash function and login_user
    with app.app_context(), \
         patch('src.routes.check_password_hash', return_value=True), \
         patch('src.routes.login_user') as mock_login:
        
        app.db = mock_db
        # Make the request
        response = client.post('/login', data={
            "username": "testuser",
            "password": "password"
        }, follow_redirects=True)
        
        # Check login_user was called
        mock_login.assert_called_once()
        
        # Check the response
        assert response.status_code == 200

def test_login_route_post_invalid_credentials(client, app, mock_db):
    """Test the login route POST request with invalid credentials"""
    # Mock the database response
    mock_user = {
        "_id": ObjectId(),
        "username": "testuser",
        "password": "hashed_password"
    }
    
    # Mock the find_one method for the users collection
    mock_db.users.find_one.return_value = mock_user
    
    # Mock the check_password_hash function
    with app.app_context(), \
         patch('src.routes.check_password_hash', return_value=False), \
         patch('src.routes.flash') as mock_flash:
        
        app.db = mock_db
        # Make the request
        response = client.post('/login', data={
            "username": "testuser",
            "password": "wrong_password"
        })
        
        # Check that flash was called with the correct message
        mock_flash.assert_called_with("Invalid username or password")
        
        # Check the response
        assert response.status_code == 200

def test_login_route_post_user_not_found(client, app, mock_db):
    """Test the login route POST request with non-existent user"""
    # Mock the find_one method to return None (user not found)
    mock_db.users.find_one.return_value = None
    
    # Mock the flash function
    with app.app_context(), \
         patch('src.routes.flash') as mock_flash:
        
        app.db = mock_db
        # Make the request
        response = client.post('/login', data={
            "username": "nonexistent",
            "password": "password"
        })
        
        # Check that flash was called with the correct message
        mock_flash.assert_called_with("Invalid username or password")
        
        # Check the response
        assert response.status_code == 200

def test_login_route_post_db_error(client, app, mock_db):
    """Test the login route POST request with a database error"""
    # Mock the find_one method to raise an exception
    mock_db.users.find_one.side_effect = Exception("Database error")
    
    # Mock the flash function
    with app.app_context(), \
         patch('src.routes.flash') as mock_flash:
        
        app.db = mock_db
        # Make the request
        response = client.post('/login', data={
            "username": "testuser",
            "password": "password"
        })
        
        # Check that flash was called with the correct message
        mock_flash.assert_called_with("An error occurred during login. Please try again.")
        
        # Check the response
        assert response.status_code == 200

def test_login_route_post_missing_fields(client):
    """Test the login route POST request with missing fields"""
    # Mock the flash function
    with patch('src.routes.flash') as mock_flash:
        # Make the request
        response = client.post('/login', data={})
        
        # Check that flash was called with the correct message
        mock_flash.assert_called_with("Username and password are required")
        
        # Check the response
        assert response.status_code == 200

@patch('flask_login.utils._get_user') 
def test_logout_route(mock_get_user, client):
    """Test the logout route"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the logout_user function
    with patch('src.routes.logout_user') as mock_logout:
        # Make the request
        response = client.get('/logout', follow_redirects=True)
        
        # Check that logout_user was called
        mock_logout.assert_called_once()
        
        # Check the response
        assert response.status_code == 200

def test_register_route_get(client):
    """Test the register route GET request"""
    response = client.get('/register')
    assert response.status_code == 200
    assert b'Register' in response.data

def test_register_route_post_success(client, app, mock_db):
    """Test the register route POST request with valid data"""
    # Mock the find_one method to return None (username doesn't exist)
    mock_db.users.find_one.return_value = None
    
    # Mock the flash function
    with app.app_context(), \
         patch('src.routes.flash') as mock_flash, \
         patch('src.routes.generate_password_hash', return_value="hashed_password"), \
         patch('src.routes.redirect'):
        
        app.db = mock_db
        # Make the request
        response = client.post('/register', data={
            "username": "newuser",
            "password": "password"
        })
        
        # Check that flash was called with the correct message
        mock_flash.assert_called_with("Registration successful! Please login.")
        
        # Check that insert_one was called
        mock_db.users.insert_one.assert_called_once()

def test_register_route_post_existing_user(client, app, mock_db):
    """Test the register route POST request with existing username"""
    # Mock the find_one method to return a user (username exists)
    mock_db.users.find_one.return_value = {"username": "existinguser"}
    
    # Mock the flash function
    with app.app_context(), \
         patch('src.routes.flash') as mock_flash, \
         patch('src.routes.redirect'):
        
        app.db = mock_db
        # Make the request
        response = client.post('/register', data={
            "username": "existinguser",
            "password": "password"
        })
        
        # Check that flash was called with the correct message
        mock_flash.assert_called_with("Username already exists")

@patch('flask_login.utils._get_user')
@patch('src.routes.requests.post')
def test_detect_cat_route_success(mock_post, mock_get_user, client, app):
    """Test the detect-cat route with successful ML response"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the requests.post response
    mock_response = MagicMock()
    mock_response.json.return_value = {"cat_detected": True, "confidence": 0.95}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    # Make the request
    response = client.post('/api/detect-cat', 
                          json={"image": "base64_image_data"},
                          content_type='application/json')
    
    # Check the response
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data["cat_detected"] == True

@patch('flask_login.utils._get_user')
@patch('src.routes.requests.post')
def test_detect_cat_route_timeout(mock_post, mock_get_user, client):
    """Test the detect-cat route with ML service timeout"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the requests.post to raise Timeout
    mock_post.side_effect = requests.exceptions.Timeout
    
    # Make the request
    response = client.post('/api/detect-cat', 
                          json={"image": "base64_image_data"},
                          content_type='application/json')
    
    # Check the response
    assert response.status_code == 504
    response_data = json.loads(response.data)
    assert "error" in response_data

@patch('flask_login.utils._get_user')
@patch('src.routes.requests.post')
def test_detect_cat_route_connection_error(mock_post, mock_get_user, client):
    """Test the detect-cat route with ML service connection error"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the requests.post to raise ConnectionError
    mock_post.side_effect = requests.exceptions.ConnectionError
    
    # Make the request
    response = client.post('/api/detect-cat', 
                          json={"image": "base64_image_data"},
                          content_type='application/json')
    
    # Check the response
    assert response.status_code == 503
    response_data = json.loads(response.data)
    assert "error" in response_data

@patch('flask_login.utils._get_user')
@patch('src.routes.requests.post')
def test_detect_cat_route_other_error(mock_post, mock_get_user, client):
    """Test the detect-cat route with any other error"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the requests.post to raise generic Exception
    mock_post.side_effect = Exception("Unexpected error")
    
    # Make the request
    response = client.post('/api/detect-cat', 
                          json={"image": "base64_image_data"},
                          content_type='application/json')
    
    # Check the response
    assert response.status_code == 500
    response_data = json.loads(response.data)
    assert "error" in response_data

@patch('flask_login.utils._get_user')
def test_detect_cat_route_missing_image(mock_get_user, client):
    """Test the detect-cat route with missing image data"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Make the request
    response = client.post('/api/detect-cat', 
                          json={},  # No image data
                          content_type='application/json')
    
    # Check the response
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert "error" in response_data

@patch('flask_login.utils._get_user')
def test_feeding_events_route(mock_get_user, client, app, mock_db):
    """Test the feeding-events route"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Setup UTC timezone
    utc_timezone = pytz.UTC
    
    # Create mock events
    today = datetime.now(utc_timezone)
    mock_events = [
        {
            "_id": ObjectId(),
            "timestamp": today,
            "type": "cat_eating",
            "confidence": 0.95,
            "image": "base64_image_data"
        }
    ]
    
    # Mock the find method for the feeding_events collection
    mock_find = MagicMock()
    mock_find.sort.return_value = mock_events
    mock_db.feeding_events.find.return_value = mock_find
    
    # Mock the list_collection_names method
    mock_db.list_collection_names.return_value = ["feeding_events"]
    
    with app.app_context():
        app.db = mock_db
        # Make the request
        response = client.get('/api/feeding-events')
        
        # Check the response
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert "feeding_events" in response_data
        assert "total_feedings" in response_data
        assert "last_feeding" in response_data
        assert "timestamps" in response_data
        assert "confidences" in response_data

@patch('flask_login.utils._get_user')
def test_feeding_events_route_no_events(mock_get_user, client, app, mock_db):
    """Test the feeding-events route with no events"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the list_collection_names method to not include feeding_events
    mock_db.list_collection_names.return_value = ["users"]
    
    with app.app_context():
        app.db = mock_db
        # Make the request
        response = client.get('/api/feeding-events')
        
        # Check the response
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data["feeding_events"] == []
        assert response_data["total_feedings"] == 0
        assert response_data["last_feeding"] is None

@patch('flask_login.utils._get_user')
def test_feeding_events_route_timezone_error(mock_get_user, client, app):
    """Test the feeding-events route with a timezone error"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the pytz.timezone to raise an error
    with patch('src.routes.pytz.timezone', side_effect=pytz.exceptions.UnknownTimeZoneError('America/New_York')):
        # Make the request
        response = client.get('/api/feeding-events')
        
        # Check the response
        assert response.status_code == 500
        response_data = json.loads(response.data)
        assert "error" in response_data

@patch('flask_login.utils._get_user')
def test_feeding_events_route_db_error(mock_get_user, client, app, mock_db):
    """Test the feeding-events route with a database error"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Mock the list_collection_names to include feeding_events
    mock_db.list_collection_names.return_value = ["feeding_events"]
    
    # Mock the find method to raise an exception
    mock_db.feeding_events.find.side_effect = Exception("Database error")
    
    with app.app_context():
        app.db = mock_db
        # Make the request
        response = client.get('/api/feeding-events')
        
        # Check the response - should still return without error but with empty data
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data["feeding_events"] == []

@patch('flask_login.utils._get_user')
def test_feeding_events_route_with_invalid_timestamp(mock_get_user, client, app, mock_db):
    """Test the feeding-events route with invalid timestamp in event"""
    # Mock the login_required decorator
    mock_get_user.return_value = MagicMock(is_authenticated=True)
    
    # Create mock events with invalid timestamp
    mock_events = [
        {
            "_id": ObjectId(),
            "timestamp": "invalid-timestamp",  # Invalid timestamp
            "type": "cat_eating",
            "confidence": 0.95,
            "image": "base64_image_data"
        }
    ]
    
    # Mock the find method for the feeding_events collection
    mock_find = MagicMock()
    mock_find.sort.return_value = mock_events
    mock_db.feeding_events.find.return_value = mock_find
    
    # Mock the list_collection_names method
    mock_db.list_collection_names.return_value = ["feeding_events"]
    
    with app.app_context():
        app.db = mock_db
        # Make the request
        response = client.get('/api/feeding-events')
        
        # Check the response
        assert response.status_code == 200
        response_data = json.loads(response.data)
        # Should not crash, but the event should not be included
        assert response_data["feeding_events"] == [] 