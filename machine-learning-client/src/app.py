from flask import Flask
import src.client_blueprint as client_blueprint

def create_app(config=None):
    app = Flask(__name__)

    try:
        print(client_blueprint.client_blueprint.name)
        app.register_blueprint(client_blueprint.client_blueprint)
    except Exception as e:
        print(f"Error registering routes blueprint: {e}")

    return app

if __name__ == "__main__":
    print("Starting Cat Detection API server...")
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
