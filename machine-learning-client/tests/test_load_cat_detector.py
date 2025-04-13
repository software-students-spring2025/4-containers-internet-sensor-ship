import pytest
from unittest.mock import patch, MagicMock
import cv2
import os

def test_load_cat_detector_success():
    """Test successful loading of cascade classifiers"""
    with patch('os.path.exists', return_value=True), \
         patch('cv2.CascadeClassifier') as mock_cascade_classifier:
        
        from src.client_blueprint import load_cat_detector
        
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade_classifier.return_value = mock_cascade
        
        result1, result2 = load_cat_detector()

        assert result1 is mock_cascade
        assert result2 is mock_cascade
        assert mock_cascade_classifier.call_count == 2


def test_load_cat_detector_empty_cascades():
    """Test loading cascade classifiers that return empty"""
    with patch('os.path.exists', return_value=True), \
         patch('cv2.CascadeClassifier') as mock_cascade_classifier:
        
        from src.client_blueprint import load_cat_detector
        
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = True
        mock_cascade_classifier.return_value = mock_cascade
        
        result1, result2 = load_cat_detector()
        
        assert result1 is None
        assert result2 is None
        assert mock_cascade_classifier.call_count == 2


def test_load_cat_detector_file_not_found():
    """Test loading when cascade files don't exist"""
    with patch('os.path.exists', return_value=False), \
         patch('os.getcwd', return_value="/app"), \
         patch('os.listdir', return_value=["src", "models"]), \
         patch('cv2.CascadeClassifier') as mock_cascade_classifier:
        
        from src.client_blueprint import load_cat_detector
        
        result1, result2 = load_cat_detector()
        
        assert result1 is None
        assert result2 is None
        assert mock_cascade_classifier.call_count == 0


def test_load_cat_detector_exception():
    """Test exception handling during cascade loading"""
    with patch('os.path.exists', return_value=True), \
         patch('cv2.CascadeClassifier', side_effect=cv2.error("Error loading cascade")):
        
        from src.client_blueprint import load_cat_detector
        
        result1, result2 = load_cat_detector()
        
        assert result1 is None
        assert result2 is None


def test_get_utc_time_format():
    """Test get_utc_time function returns expected format"""
    from src.client_blueprint import get_utc_time
    
    result = get_utc_time()

    assert hasattr(result, 'year')
    assert hasattr(result, 'month')
    assert hasattr(result, 'day')
    assert hasattr(result, 'hour')
    assert hasattr(result, 'minute')
    assert hasattr(result, 'second')


def test_detect_cat_with_cascade_params():
    """Test that detect_cat_with_cascade uses correct parameters"""
    with patch('cv2.cvtColor', return_value=MagicMock()), \
         patch('numpy.std', return_value=30):
        
        from src.client_blueprint import detect_cat_with_cascade
        
        with patch('src.client_blueprint.cat_cascade') as mock_cascade, \
             patch('src.client_blueprint.cat_cascade_ext') as mock_cascade_ext:
            
            import numpy as np
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            detect_cat_with_cascade(test_image)
            
            mock_cascade.detectMultiScale.assert_called_once()
            args, kwargs = mock_cascade.detectMultiScale.call_args
            
            assert kwargs['scaleFactor'] == 1.1
            assert kwargs['minNeighbors'] == 5
            assert kwargs['minSize'] == (40, 40)
            
            mock_cascade_ext.detectMultiScale.assert_called_once()
            args, kwargs = mock_cascade_ext.detectMultiScale.call_args
            
            assert kwargs['scaleFactor'] == 1.1
            assert kwargs['minNeighbors'] == 5
            assert kwargs['minSize'] == (40, 40)


def test_detect_cat_with_no_cascades_available():
    """Test detection when cascade classifiers are not available"""
    from src.client_blueprint import detect_cat_with_cascade
    with patch('src.client_blueprint.cat_cascade', None), \
         patch('src.client_blueprint.cat_cascade_ext', None):
        
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        detected, confidence = detect_cat_with_cascade(test_image)
        
        assert detected is False
        assert confidence == 0.0


def test_preprocess_image_with_data_url():
    """Test preprocess_image with data URL format"""
    with patch('base64.b64decode') as mock_b64decode, \
         patch('numpy.frombuffer') as mock_frombuffer, \
         patch('cv2.imdecode') as mock_imdecode:
        
        from src.client_blueprint import preprocess_image
        
        mock_b64decode.return_value = b"test_bytes"
        mock_frombuffer.return_value = MagicMock()
        import numpy as np
        mock_img = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_img
        
        img_url = "data:image/jpeg;base64,/9j/test=="
        result = preprocess_image(img_url)
        
        mock_b64decode.assert_called_with("/9j/test==")
        mock_frombuffer.assert_called_once()
        mock_imdecode.assert_called_once()
        
        assert result is mock_img


def test_preprocess_image_empty_result():
    """Test preprocess_image when decoding fails"""
    with patch('base64.b64decode') as mock_b64decode, \
         patch('numpy.frombuffer') as mock_frombuffer, \
         patch('cv2.imdecode') as mock_imdecode:
        
        from src.client_blueprint import preprocess_image
        
        mock_b64decode.return_value = b"test_bytes"
        mock_frombuffer.return_value = MagicMock()
        mock_imdecode.return_value = None
        
        with pytest.raises(ValueError, match="Failed to decode image"):
            preprocess_image("data:image/jpeg;base64,invalid==")


def test_preprocess_image_without_data_url():
    """Test preprocess_image with raw base64 data"""
    with patch('base64.b64decode') as mock_b64decode, \
         patch('numpy.frombuffer') as mock_frombuffer, \
         patch('cv2.imdecode') as mock_imdecode:
        
        from src.client_blueprint import preprocess_image
        
        mock_b64decode.return_value = b"test_bytes"
        mock_frombuffer.return_value = MagicMock()
        import numpy as np
        mock_img = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_img
        
        raw_data = "raw_base64_data=="
        result = preprocess_image(raw_data)
        
        mock_b64decode.assert_called_with("raw_base64_data==")
        
        assert result is mock_img 