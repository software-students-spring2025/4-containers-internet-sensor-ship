import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from main import CatDetector


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


def test_cat_detector_initialization(mock_mongo, mock_camera, mock_model):
    detector = CatDetector()
    assert detector.last_detection_time is None
    assert detector.detection_cooldown == 60


def test_preprocess_image():
    detector = CatDetector()
    mock_frame = MagicMock()
    mock_frame.shape = (480, 640, 3)
    processed = detector.preprocess_image(mock_frame)
    assert processed.shape == (1, 224, 224, 3)


def test_detect_cat(mock_model):
    detector = CatDetector()
    mock_frame = MagicMock()
    mock_frame.shape = (480, 640, 3)
    assert detector.detect_cat(mock_frame) is True


def test_save_detection(mock_mongo):
    detector = CatDetector()
    detector.save_detection()
    detector.db.feeding_events.insert_one.assert_called_once()


@patch("time.time")
def test_run_detection_loop(mock_time, mock_mongo, mock_camera, mock_model):
    mock_time.return_value = 0
    detector = CatDetector()

    # Simulate two frames with enough time between them
    mock_camera.return_value.read.side_effect = [
        (True, None),  # First frame
        (True, None),  # Second frame
        (False, None),  # Break the loop
    ]

    detector.run()
    assert detector.db.feeding_events.insert_one.call_count == 1
