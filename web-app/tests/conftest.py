import pytest
from src.app import create_app
import json

# A boilerplate fixture for creating a testable app
@pytest.fixture
def app():
    app = create_app(TEST_CONTEXT=True)
    return app