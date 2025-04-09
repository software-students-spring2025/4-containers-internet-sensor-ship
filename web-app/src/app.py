'''
The main flask app
'''
from flask import Flask

# All routing is stored in ./routes.py to be imported
from src.routes import routes


def create_app():
    ''' 
    The app factory for creating instances of the app
    '''
    app = Flask(__name__)

    app.register_blueprint(routes)

    return app


if __name__ == "__main__":
    create_app().run()


'''Just changing something silly here to see what happens'''