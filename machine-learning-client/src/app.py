from flask import Flask

def create_app(config=None):
    app = Flask(__name__)

    try:
        from src.client_blueprint import ml_client
        app.register_blueprint(ml_client, port=5000)
    except Exception as e:
        print(f"Error ")
        print(f"Error registering routes blueprint: {e}")

    return app

if __name__ == "__main__":
    print("Starting Cat Detection API server...")
    create_app()
