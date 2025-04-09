import pytest


# Placeholder test to be better defined later
def test_get_index(client):
    assert client.get("/").status_code != 404
