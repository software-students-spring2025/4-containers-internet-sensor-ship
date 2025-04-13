from unittest.mock import MagicMock, patch
import sys
import numpy as np

import pytest

def mock_module(name):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]

mock_module('cv2')
mock_module('pymongo')
mock_module('tensorflow')
mock_module('tensorflow.keras')
mock_module('tensorflow.keras.models')
mock_module('dotenv')
mock_module('python-dotenv')

dotenv = mock_module('dotenv')
dotenv.load_dotenv = MagicMock()

from src.app import create_app


@pytest.fixture
def mock_mongo_module():
    """Mock the MongoDB module to prevent actual connections"""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_camera_module():
    """Mock OpenCV camera functions"""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_model_fixture():
    """Mock ML model to prevent actual loading"""
    mock = MagicMock()
    return mock


@pytest.fixture
def test_app_module():
    """Create a testable app instance"""
    from src.app import create_app
    app = create_app()
    app.config.update({"TESTING": True})
    return app


@pytest.fixture
def client(test_app_module):
    """Create a test client for the application"""
    with test_app_module.test_client() as client:
        yield client


@pytest.fixture
def db_mock():
    """Mock database for direct client_blueprint testing"""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.feeding_events = mock_collection
    return mock_db
