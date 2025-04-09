from flask import Flask
from . import client_blueprint

def create_app(config=None):
    app = Flask(__name__)

    try:
        print(client_blueprint.client_blueprint.name)
        app.register_blueprint(client_blueprint.client_blueprint, port=5000)
    except Exception as e:
        print(client_blueprint.client_blueprint.name)
        print(f"Error registering routes blueprint: {e}")

    return app

if __name__ == "__main__":
    print("Starting Cat Detection API server...")
    create_app()
