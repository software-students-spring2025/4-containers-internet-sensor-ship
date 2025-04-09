import pytest


# Placeholder test to be better defined later
def test_app_running(client):
    assert client.get("/health").status_code == 200
