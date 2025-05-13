# (your existing file)

from flask import Flask
try:
    from flask_cors import CORS  # Import CORS
    cors_available = True
except ModuleNotFoundError:
    cors_available = False
from app.search_datasets import search_datasets_bp

def create_app():
    app = Flask(__name__)

    # Enable CORS for all routes
    if cors_available:
        CORS(app)
    else:
        @app.after_request
        def add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
            return response

    # Alternatively, you can restrict CORS to specific domains:
    # CORS(app, origins=["http://localhost:3000"])

    # Register Blueprints
    app.register_blueprint(search_datasets_bp)

    return app
