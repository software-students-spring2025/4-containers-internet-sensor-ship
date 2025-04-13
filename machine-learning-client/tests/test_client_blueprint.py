import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import datetime

# Import with mocks to avoid loading actual dependencies
with patch('cv2.imread', MagicMock()), \
     patch('cv2.CascadeClassifier', MagicMock()), \
     patch('pymongo.MongoClient', MagicMock()):
    from src.client_blueprint import detect_cat_with_cascade, preprocess_image, get_utc_time


def test_preprocess_image_with_dataurl():
    """Test preprocess_image with data URL format"""
    with patch('base64.b64decode') as mock_b64decode, \
         patch('numpy.frombuffer') as mock_frombuffer, \
         patch('cv2.imdecode') as mock_imdecode:
        
        # Configure mocks
        mock_b64decode.return_value = b"test_bytes"
        mock_frombuffer.return_value = np.array([1, 2, 3])
        mock_img = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_img
        
        # Test with data URL format
        img_url = "data:image/jpeg;base64,/9j/test=="
        result = preprocess_image(img_url)
        
        # Verify result and calls
        assert result is mock_img
        mock_b64decode.assert_called_with("/9j/test==")
        mock_frombuffer.assert_called_once()
        mock_imdecode.assert_called_once()


def test_preprocess_image_with_raw_base64():
    """Test preprocess_image with raw base64 data"""
    with patch('base64.b64decode') as mock_b64decode, \
         patch('numpy.frombuffer') as mock_frombuffer, \
         patch('cv2.imdecode') as mock_imdecode:
        
        # Configure mocks
        mock_b64decode.return_value = b"test_bytes"
        mock_frombuffer.return_value = np.array([1, 2, 3])
        mock_img = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_img
        
        # Test with raw base64
        raw_data = "/9j/test=="
        result = preprocess_image(raw_data)
        
        # Verify result and calls
        assert result is mock_img
        mock_b64decode.assert_called_with("/9j/test==")


def test_preprocess_image_with_invalid_data():
    """Test preprocess_image with invalid data"""
    with patch('base64.b64decode') as mock_b64decode, \
         patch('numpy.frombuffer') as mock_frombuffer, \
         patch('cv2.imdecode') as mock_imdecode:
        
        # Configure mocks to simulate decoding failure
        mock_imdecode.return_value = None
        mock_b64decode.return_value = b"test_bytes"
        mock_frombuffer.return_value = np.array([1, 2, 3])
        
        # Test with invalid image data
        with pytest.raises(ValueError):
            preprocess_image("invalid data")


def test_detect_cat_with_cascade_no_detection():
    """Test cat detection with no cats detected"""
    with patch('cv2.cvtColor', return_value=np.zeros((10, 10), dtype=np.uint8)):
        
        # Mock the cascade detectors
        with patch('src.client_blueprint.cat_cascade') as mock_cascade, \
             patch('src.client_blueprint.cat_cascade_ext') as mock_cascade_ext:
            
            # Configure mocks for no detections
            mock_cascade.detectMultiScale.return_value = []
            mock_cascade_ext.detectMultiScale.return_value = []
            
            # Create test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Run detection
            detected, confidence = detect_cat_with_cascade(test_image)
            
            # Verify results
            assert detected is False
            assert confidence == 0.1
            mock_cascade.detectMultiScale.assert_called_once()
            mock_cascade_ext.detectMultiScale.assert_called_once()


def test_detect_cat_with_cascade_with_detection():
    """Test cat detection with cats detected"""
    with patch('cv2.cvtColor', return_value=np.zeros((10, 10), dtype=np.uint8)), \
         patch('numpy.std', return_value=30):  # Ensure enough variation
        
        # Mock the cascade detectors
        with patch('src.client_blueprint.cat_cascade') as mock_cascade, \
             patch('src.client_blueprint.cat_cascade_ext') as mock_cascade_ext:
            
            # Configure mocks for detection
            mock_cascade.detectMultiScale.return_value = [(10, 10, 30, 30)]
            mock_cascade_ext.detectMultiScale.return_value = []
            
            # Create test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Run detection
            detected, confidence = detect_cat_with_cascade(test_image)
            
            # Verify results
            assert detected is True
            assert 0.5 <= confidence <= 0.7  # Typical range for single detection
            mock_cascade.detectMultiScale.assert_called_once()
            mock_cascade_ext.detectMultiScale.assert_called_once()


def test_detect_cat_with_cascade_multiple_detections():
    """Test cat detection with multiple detected cats"""
    with patch('cv2.cvtColor', return_value=np.zeros((10, 10), dtype=np.uint8)), \
         patch('numpy.std', return_value=30):  # Ensure enough variation
        
        # Mock the cascade detectors
        with patch('src.client_blueprint.cat_cascade') as mock_cascade, \
             patch('src.client_blueprint.cat_cascade_ext') as mock_cascade_ext:
            
            # Configure mocks for multiple detections
            mock_cascade.detectMultiScale.return_value = [(10, 10, 30, 30), (50, 50, 30, 30)]
            mock_cascade_ext.detectMultiScale.return_value = []
            
            # Create test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Run detection
            detected, confidence = detect_cat_with_cascade(test_image)
            
            # Verify results - confidence should be higher with more detections
            assert detected is True
            assert 0.6 <= confidence  # Value from the actual implementation
            mock_cascade.detectMultiScale.assert_called_once()
            mock_cascade_ext.detectMultiScale.assert_called_once()


def test_detect_cat_with_cascade_null_detectors():
    """Test cat detection when detectors are None"""
    # Save original cascade objects
    with patch('src.client_blueprint.cat_cascade', None), \
         patch('src.client_blueprint.cat_cascade_ext', None):
        
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Run detection
        detected, confidence = detect_cat_with_cascade(test_image)
        
        # Verify results
        assert detected is False
        assert confidence == 0.0  # Should be 0 when no detector is available


def test_detect_cat_with_cascade_combined_detections():
    """Test cat detection with detections from both cascades"""
    with patch('cv2.cvtColor', return_value=np.zeros((10, 10), dtype=np.uint8)), \
         patch('numpy.std', return_value=30):  # Ensure enough variation
        
        # Mock the cascade detectors
        with patch('src.client_blueprint.cat_cascade') as mock_cascade, \
             patch('src.client_blueprint.cat_cascade_ext') as mock_cascade_ext:
            
            # Configure mocks for detections from both cascades
            mock_cascade.detectMultiScale.return_value = [(10, 10, 30, 30)]
            mock_cascade_ext.detectMultiScale.return_value = [(50, 50, 30, 30)]
            
            # Create test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Run detection
            detected, confidence = detect_cat_with_cascade(test_image)
            
            # Verify results
            assert detected is True
            assert confidence > 0.5  # Should have higher confidence with detections from both cascades
            mock_cascade.detectMultiScale.assert_called_once()
            mock_cascade_ext.detectMultiScale.assert_called_once()


def test_detect_cat_with_verification_failed():
    """Test cat detection when the verification step fails (low variation)"""
    with patch('cv2.cvtColor', return_value=np.zeros((10, 10), dtype=np.uint8)), \
         patch('numpy.std', return_value=10):  # Low variation
        
        # Mock the cascade detectors
        with patch('src.client_blueprint.cat_cascade') as mock_cascade, \
             patch('src.client_blueprint.cat_cascade_ext') as mock_cascade_ext:
            
            # Configure mocks for detection
            mock_cascade.detectMultiScale.return_value = [(10, 10, 30, 30)]
            mock_cascade_ext.detectMultiScale.return_value = []
            
            # Create test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Run detection
            detected, confidence = detect_cat_with_cascade(test_image)
            
            # Verify results - should fail verification
            assert detected is False
            assert confidence == 0.1  # Low confidence when verification fails
            mock_cascade.detectMultiScale.assert_called_once()
            mock_cascade_ext.detectMultiScale.assert_called_once()


def test_get_utc_time():
    """Test the get_utc_time function returns a datetime object"""
    # We will simply check that the function returns a datetime object
    # and doesn't crash
    with patch('src.client_blueprint.datetime') as mock_datetime:
        # Setup the mock to return a predictable datetime
        test_datetime = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = test_datetime
        
        # Call the function
        result = get_utc_time()
        
        # Verify the function called utcnow
        mock_datetime.utcnow.assert_called_once()
        
        # Verify the result is what we expect based on our mock
        assert result == test_datetime