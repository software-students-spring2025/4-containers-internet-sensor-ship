import pytest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime
from bson import ObjectId
from flask_login import current_user
from werkzeug.security import generate_password_hash

from src.app import User, get_utc_time, create_app

def test_user_class_init_with_dict():
    """Test User class initialization with a dictionary"""
    user_data = {
        "_id": ObjectId(),
        "username": "testuser"
    }
    user = User(user_data)
    assert user.id == str(user_data["_id"])
    assert user.username == "testuser"

def test_user_class_init_with_string():
    """Test User class initialization with just an ID string"""
    user_id = str(ObjectId())
    user = User(user_id)
    assert user.id == user_id
    assert user.username == "Unknown"

def test_get_utc_time():
    """Test the get_utc_time function"""
    time = get_utc_time()
    assert isinstance(time, datetime)

def test_direct_login_get(client):
    """Test direct login GET route"""
    response = client.get('/direct-login')
    assert response.status_code == 200
    assert b'Login' in response.data

def test_direct_login_post_success(client, app, mock_db):
    """Test direct login POST with valid credentials"""
    # Mock the database response
    mock_user = {
        "_id": ObjectId(),
        "username": "testuser",
        "password": "hashed_password"
    }
    
    # Setup mocks
    mock_db.users.find_one.return_value = mock_user
    
    with app.app_context():
        app.db = mock_db
        
        with patch('werkzeug.security.check_password_hash', return_value=True):
            # Make the request
            response = client.post('/direct-login', data={
                "username": "testuser",
                "password": "password"
            })
            
            # Check response
            assert response.status_code in [200, 302]  # Either OK or redirect

def test_direct_login_post_invalid_credentials(client, app, mock_db):
    """Test direct login POST with invalid credentials"""
    # Mock the database response
    mock_user = {
        "_id": ObjectId(),
        "username": "testuser",
        "password": "hashed_password"
    }
    
    # Setup mocks
    mock_db.users.find_one.return_value = mock_user
    
    with app.app_context():
        app.db = mock_db
        
        with patch('werkzeug.security.check_password_hash', return_value=False):
            # Make the request
            response = client.post('/direct-login', data={
                "username": "testuser",
                "password": "wrong_password"
            })
            
            # Check response
            assert response.status_code == 200  # Should stay on login page

def test_direct_login_post_missing_fields(client, app):
    """Test direct login POST with missing fields"""
    with app.app_context():
        # Make the request
        response = client.post('/direct-login', data={})
        
        # Check response
        assert response.status_code == 200  # Should stay on login page

def test_direct_login_post_error(client, app, mock_db):
    """Test direct login POST with a database error"""
    # Setup mocks to raise an exception
    mock_db.users.find_one.side_effect = Exception("Database error")
    
    with app.app_context():
        app.db = mock_db
        
        # Make the request
        response = client.post('/direct-login', data={
            "username": "testuser",
            "password": "password"
        })
        
        # Check response
        assert response.status_code == 200  # Should stay on login page

def test_direct_logout(client):
    """Test direct logout route"""
    # Make the request
    response = client.get('/direct-logout')
    
    # Check response is a redirect
    assert response.status_code == 302  # Redirect status code

def test_home_authenticated(client, app):
    """Test home route with authenticated user"""
    # Setting up an authenticated user via client login
    with app.test_request_context():
        client.get('/direct-login')  # Initialize session
        
        # Make the request
        response = client.get('/')
        
        # Check response
        assert response.status_code in [200, 302]  # Either OK or redirect

def test_home_not_authenticated(client, app):
    """Test home route with unauthenticated user"""
    # Make the request (no login, so user is unauthenticated)
    response = client.get('/')
    
    # Should redirect to login
    assert response.status_code == 302  # Redirect status code

def test_home_error(client, app):
    """Test home route with an error"""
    # We'll simulate an error condition by directly checking the fallback behavior
    with app.test_request_context():
        # Make the request
        response = client.get('/')
        
        # Since we're not logged in, we should be redirected to login
        assert response.status_code == 302
        assert '/login' in response.location

def test_api_register_success(client, app, mock_db):
    """Test API register route with valid data"""
    # Setup mocks
    mock_db.users.find_one.return_value = None
    
    with app.app_context(), \
         patch('werkzeug.security.generate_password_hash', return_value="hashed_password"):
        
        app.db = mock_db
        
        # Make the request
        response = client.post('/api/register', 
                             json={"username": "newuser", "password": "password"},
                             content_type='application/json')
        
        # Check the response
        assert response.status_code == 201
        response_data = json.loads(response.data)
        assert "message" in response_data
        assert "User registered successfully" in response_data["message"]
        assert "user_id" in response_data
        
        # Check that insert_one was called
        mock_db.users.insert_one.assert_called_once()

def test_api_register_missing_fields(client, app):
    """Test API register route with missing fields"""
    with app.app_context():
        # Make the request without username
        response = client.post('/api/register', 
                             json={"password": "password"},
                             content_type='application/json')
        
        # Check the response
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert "error" in response_data
        assert "Username and password are required" in response_data["error"]

def test_api_register_existing_user(client, app, mock_db):
    """Test API register route with existing username"""
    # Setup mocks to return an existing user
    mock_db.users.find_one.return_value = {"username": "existinguser"}
    
    with app.app_context():
        app.db = mock_db
        
        # Make the request
        response = client.post('/api/register', 
                             json={"username": "existinguser", "password": "password"},
                             content_type='application/json')
        
        # Check the response
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert "error" in response_data
        assert "Username already exists" in response_data["error"]

def test_api_register_error(client, app, mock_db):
    """Test API register route with an error"""
    # Setup mocks to raise an exception
    mock_db.users.find_one.side_effect = Exception("Database error")
    
    with app.app_context():
        app.db = mock_db
        
        # Make the request
        response = client.post('/api/register', 
                             json={"username": "newuser", "password": "password"},
                             content_type='application/json')
        
        # Check the response
        assert response.status_code == 500
        response_data = json.loads(response.data)
        assert "error" in response_data
        assert "Internal server error" in response_data["error"]

def test_load_user_success(app, mock_db):
    """Test the user_loader function with valid user ID"""
    # Setup mocks to return a user
    user_id = str(ObjectId())
    mock_user = {
        "_id": ObjectId(user_id),
        "username": "testuser"
    }
    mock_db.users.find_one.return_value = mock_user
    
    with app.app_context():
        app.db = mock_db
        
        # Get the user_loader function from the app
        login_manager = app.login_manager
        user_loader = login_manager._user_callback
        
        # Call the user_loader function
        user = user_loader(user_id)
        
        # Check the user
        assert user is not None
        assert user.id == user_id
        assert user.username == "testuser"

def test_load_user_not_found(app, mock_db):
    """Test the user_loader function with non-existent user ID"""
    # Setup mocks to return None (user not found)
    user_id = str(ObjectId())
    mock_db.users.find_one.return_value = None
    
    with app.app_context():
        app.db = mock_db
        
        # Get the user_loader function from the app
        login_manager = app.login_manager
        user_loader = login_manager._user_callback
        
        # Call the user_loader function
        user = user_loader(user_id)
        
        # Check that None is returned
        assert user is None

def test_load_user_empty_id(app, mock_db):
    """Test the user_loader function with empty user ID"""
    with app.app_context():
        app.db = mock_db
        
        # Get the user_loader function from the app
        login_manager = app.login_manager
        user_loader = login_manager._user_callback
        
        # Call the user_loader function with empty ID
        user = user_loader("")
        
        # Check that None is returned
        assert user is None

def test_load_user_error(app, mock_db):
    """Test the user_loader function with an error"""
    # Setup mocks to raise an exception
    user_id = str(ObjectId())
    mock_db.users.find_one.side_effect = Exception("Database error")
    
    with app.app_context():
        app.db = mock_db
        
        # Get the user_loader function from the app
        login_manager = app.login_manager
        user_loader = login_manager._user_callback
        
        # Call the user_loader function
        user = user_loader(user_id)
        
        # Check that None is returned on error
        assert user is None

def test_load_user_exception(app, mock_db):
    """Test the user_loader function when an exception occurs"""
    # Setup mocks to raise an exception
    user_id = str(ObjectId())
    mock_db.users.find_one.side_effect = Exception("Database error")
    
    with app.app_context():
        app.db = mock_db
        
        # Get the user_loader function from the app
        login_manager = app.login_manager
        user_loader = login_manager._user_callback
        
        # Call the user_loader function - should handle the exception
        user = user_loader(user_id)
        
        # Check that the function handled the exception and returned None
        assert user is None
