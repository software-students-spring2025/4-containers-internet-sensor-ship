import pytest
from app import app, db
from flask_login import FlaskLoginClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.test_client_class = FlaskLoginClient
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_mongo():
    with patch('pymongo.MongoClient') as mock:
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock.return_value.__getitem__.return_value = mock_db
        yield mock

def test_index_redirects_to_login(client):
    response = client.get('/')
    assert response.status_code == 302
    assert '/login' in response.location

def test_login_page_loads(client):
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Login' in response.data

def test_register_page_loads(client):
    response = client.get('/register')
    assert response.status_code == 200
    assert b'Register' in response.data

def test_successful_login(client, mock_mongo):
    # Mock user data
    mock_mongo.return_value.__getitem__.return_value.users.find_one.return_value = {
        '_id': '123',
        'username': 'testuser',
        'password': 'testpass'
    }
    
    response = client.post('/login', data={
        'username': 'testuser',
        'password': 'testpass'
    })
    
    assert response.status_code == 302
    assert '/' in response.location

def test_failed_login(client, mock_mongo):
    # Mock no user found
    mock_mongo.return_value.__getitem__.return_value.users.find_one.return_value = None
    
    response = client.post('/login', data={
        'username': 'wronguser',
        'password': 'wrongpass'
    })
    
    assert response.status_code == 200
    assert b'Invalid username or password' in response.data

def test_successful_registration(client, mock_mongo):
    # Mock no existing user
    mock_mongo.return_value.__getitem__.return_value.users.find_one.return_value = None
    
    response = client.post('/register', data={
        'username': 'newuser',
        'password': 'newpass'
    })
    
    assert response.status_code == 302
    assert '/login' in response.location

def test_duplicate_registration(client, mock_mongo):
    # Mock existing user
    mock_mongo.return_value.__getitem__.return_value.users.find_one.return_value = {
        '_id': '123',
        'username': 'existinguser',
        'password': 'existingpass'
    }
    
    response = client.post('/register', data={
        'username': 'existinguser',
        'password': 'newpass'
    })
    
    assert response.status_code == 200
    assert b'Username already exists' in response.data

def test_logout(client, mock_mongo):
    # First login
    mock_mongo.return_value.__getitem__.return_value.users.find_one.return_value = {
        '_id': '123',
        'username': 'testuser',
        'password': 'testpass'
    }
    
    client.post('/login', data={
        'username': 'testuser',
        'password': 'testpass'
    })
    
    # Then logout
    response = client.get('/logout')
    assert response.status_code == 302
    assert '/login' in response.location 