import pytest
from src.app import create_app

# A boilerplate fixture for creating a testable app
@pytest.fixture
def app():
    app = create_app()
    return app