import pytest
from flask import jsonify

def test_index(client):
    # Not logged in yet
    response = client.get("/")
    assert response.status_code == 302
    # Log in
    client.post("/login")
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.content_type

def test_detect_cat(client):
    # Not logged in yet
    response = client.post("/api/detect-cat")
    assert response.status_code == 302
    # Log in
    client.post("/login")
    response = client.post("/api/detect-cat")
    # Make sure an error is thrown b/c no image sent with request
    assert response.status_code // 100 == 4

    # Send arbitrary image data, will be 400 or 500 b/c no database
    client.post("/api/detect-cat", json={"image": "0000000"})
    assert 3 < response.status_code // 100 < 6


def test_get_feeding_events(client):
     # Not logged in yet
    response = client.get("/api/feeding-events")
    assert response.status_code == 302
    client.post("/login")
    response = client.get("/api/feeding-events")
    assert response.status_code == 200
    assert "application/json" in response.content_type
    # Will be empty because this is just a unit test, not an integration test :)


def test_register(client):
    response = client.get("/register")
    # Make sure valid response received
    assert response.status_code == 200

    response = client.post("/register")
    assert response.status_code == 400
    


def test_logout(client):
    response = client.get("/logout")
    assert response.status_code == 302
