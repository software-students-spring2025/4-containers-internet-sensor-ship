import pytest
from unittest.mock import MagicMock, patch
from src.app import create_app


@pytest.fixture(scope='module')
def test_app():
    with patch('pymongo.MongoClient') as mock_client:
        # Create mock database and collections
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        
        # Create the app with the mocked MongoDB
        app = create_app()
        app.config.update({'TESTING': True})
        yield app

@pytest.fixture
def app():
    with patch('pymongo.MongoClient') as mock_client:
        # Create mock database and collections
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        
        # Create the app with the mocked MongoDB
        app = create_app()
        app.config.update({'TESTING': True})
        yield app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def mock_db():
    """Fixture to provide a mocked database object with collections"""
    db = MagicMock()
    
    # Set up collections
    db.users = MagicMock()
    db.users.find_one = MagicMock()
    db.users.insert_one = MagicMock()
    
    db.feeding_events = MagicMock()
    db.feeding_events.find = MagicMock()
    db.feeding_events.insert_one = MagicMock()
    
    db.list_collection_names = MagicMock()
    
    return db
