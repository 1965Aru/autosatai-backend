# run.py (application entry-point)
import os
import ee
import tempfile
import json
from dotenv import load_dotenv
from google.oauth2 import service_account

# Load local .env for development; on Render, env-vars are provided in settings
load_dotenv()

def initialize_ee_service_account():
    key_json = None

    # 1) Prefer file path if available (for local development)
    key_path_env = os.getenv("GEE_SERVICE_ACCOUNT_PATH", "secrets/gee-key.json")
    if key_path_env and os.path.exists(key_path_env):
        with open(key_path_env, "r") as f:
            key_json = f.read()
    else:
        # 2) Fallback to environment blob (for Render)
        blob = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
        if blob:
            key_json = blob.strip().strip('"').strip("'")

    if key_json:
        # 3) Write JSON to temp file
        temp_dir = tempfile.gettempdir()
        creds_path = os.path.join(temp_dir, "ee_service_account.json")
        with open(creds_path, "w") as f:
            f.write(key_json)

        # 4) Load credentials via google-auth
        info = json.loads(key_json)
        project = info.get("project_id") or os.getenv("EE_PROJECT")
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )

        # 5) Initialize Earth Engine with explicit credentials
        ee.Initialize(credentials=creds, project=project)
        print(f"✅ Earth Engine initialized via service account: {info.get('client_email')} for project {project}")
        return

    # 6) Fallback: local user credentials
    try:
        ee.Initialize()
        print("✅ Earth Engine initialized via local user credentials")
    except Exception as e:
        raise RuntimeError(
            "Earth Engine authentication failed: no service account JSON/path found and no local credentials."  
        ) from e

# Initialize Earth Engine before importing any EE-using modules
initialize_ee_service_account()

from flask import Flask, jsonify
try:
    from flask_cors import CORS
    cors_available = True
except ModuleNotFoundError:
    cors_available = False

# ── Blueprints ────────────────────────────────────────────────────────
from app.search_datasets       import search_datasets_bp
from app.analyze               import analyse_bp
from app.analyse_all_nights    import analyse_all_bp
from app.analyse_agri          import analyse_agri_bp
from app.analyse_all_agri      import analyse_all_agri_bp
try:
    from app.natural_resources import natural_resources_bp
except ModuleNotFoundError:
    natural_resources_bp = None
from app.report_agri            import report_agri_bp
from app.report_night_lights    import report_nl_bp


def create_app() -> Flask:
    app = Flask(__name__)

    # ── CORS Configuration ──────────────────────────────────────────────
    if cors_available:
        CORS(
            app,
            resources={r"/*": {"origins": "*"}},
            supports_credentials=True,
            methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
        )
    else:
        @app.after_request
        def add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
            return response

    # ── Register Blueprints ─────────────────────────────────────────────
    app.register_blueprint(search_datasets_bp)
    app.register_blueprint(analyse_bp)
    app.register_blueprint(analyse_all_bp)
    app.register_blueprint(analyse_agri_bp)
    app.register_blueprint(analyse_all_agri_bp)
    if natural_resources_bp:
        app.register_blueprint(natural_resources_bp)
    app.register_blueprint(report_agri_bp)
    app.register_blueprint(report_nl_bp)

    # ── Health-check Endpoint ──────────────────────────────────────────
    @app.route("/health")
    def health_check():
        return jsonify(status="ok")

    return app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    create_app().run(host="0.0.0.0", port=port, debug=True)
