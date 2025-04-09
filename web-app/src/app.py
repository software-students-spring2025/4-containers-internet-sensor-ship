from flask import Flask


def create_app():
    app = Flask(__name__)

    # All routing is stored in ./routes.py to be imported
    from src.routes import routes

    app.register_blueprint(routes)

    return app


if __name__ == "__main__":
    create_app().run()
