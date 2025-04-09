import pytest
import src.app as flapp 
from datetime import datetime

def test_app_running(client):
    """
    Basic test to make sure the app is up and running
    """
    assert client.get("/health").status_code == 200

def test_user_class():
    """
    Tests the class user to make sure all fields can be assigned
    and id field is made into a string
    """
    test_user = flapp.User({"_id": 150, "username": "fake_user"})
    assert test_user.id == "150", test_user.username == "fake_user"

def test_get_utc_time():
    """
    Makes sure the timestamp function worms
    """
    start = datetime.utcnow()
    alpha = flapp.get_utc_time()
    end = datetime.utcnow()
    assert start <= alpha <= end

def test_jinja2_filter_datetime(client):
    """
    Test the _jinja2_filter_datetime function
    """
    result = flapp.filter_datetime(datetime(2024, 4, 15)) 
    expected = "2024-04-15 00:00:00"
    assert result == expected

"""
Slow tests below this line
"""

def test_no_error_on_bad_mongodb_connection():
    """
    Makes sure a bad mongodb connection does not crash the app
    """
    try:
        test_app = flapp.create_app(test_config={
            "SECRET_KEY":"BAD", 
            "MONGODB_URI":"BAD",
            "MONGODB_DBNAME":"BAD"})
    except Exception:
        pytest.fail("Error raised on bad mongodb connection")

def test_load_user_on_bad_input(client):
    """
    Make sure load_user throws an error on bad input
    """
    with pytest.raises(Exception) as e:
        client.load_user(None)