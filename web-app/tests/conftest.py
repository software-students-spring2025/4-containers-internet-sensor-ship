import pytest
from src.app import create_app 


@pytest.fixture(scope='module')
def test_app():
    app = create_app()
    app.config.update({'TESTING': True})
    yield app

@pytest.fixture
def app():
    app = create_app()
    app.config.update({'TESTING': True})
    return app

@pytest.fixture
def client(app):
    return app.test_client()
