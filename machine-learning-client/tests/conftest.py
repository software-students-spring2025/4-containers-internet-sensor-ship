import pytest
from src.app import create_app
from unittest.mock import MagicMock, patch
from datetime import datetime
from src.app import create_app


@pytest.fixture
def mock_mongo():
    with patch("pymongo.MongoClient") as mock:
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock.return_value.__getitem__.return_value = mock_db
        yield mock


@pytest.fixture
def mock_camera():
    with patch("cv2.VideoCapture") as mock:
        mock_cam = MagicMock()
        mock_cam.read.return_value = (True, None)
        mock.return_value = mock_cam
        yield mock


@pytest.fixture
def mock_model():
    with patch("tensorflow.keras.models.load_model") as mock:
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.8]]  # Simulate cat detection
        mock.return_value = mock_model
        yield mock

# A boilerplate fixture for creating a testable app
@pytest.fixture
def app():
    app = create_app()
    return app