"""
Main application module for the cat detection API.
This module creates and configures the Flask application.
"""
from flask import Flask
from src.client_blueprint import client_blueprint


def create_app(config_object=None):
    """
    Create and configure the Flask application.
    
    Args:
        config_object: Configuration object (optional)
        
    Returns:
        Flask: The configured Flask application
    """
    application = Flask(__name__)
    
    # Apply optional configuration
    if config_object:
        application.config.from_object(config_object)

    try:
        print(client_blueprint.name)
        application.register_blueprint(client_blueprint)
    except Exception as e:  # pylint: disable=broad-except
        # We use broad exception to catch any registration issues
        print(f"Error registering routes blueprint: {e}")

    return application


if __name__ == "__main__":
    print("Starting Cat Detection API server...")
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
