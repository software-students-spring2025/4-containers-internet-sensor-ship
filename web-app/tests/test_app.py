
def test_app_creation(test_app):
    assert test_app is not None
    assert test_app.config['TESTING'] is True

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok", "message": "App is running"}
