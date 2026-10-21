"""
CrimeVisionAI Flask Application Package.
"""

import os
from pathlib import Path
from flask import Flask, jsonify


def create_app(test_config: dict | None = None) -> Flask:
    """Application factory for CrimeVisionAI."""
    app = Flask(__name__)

    # Default configuration
    project_root = Path(app.root_path).parent
    default_upload = project_root / "evidence" / "uploads"
    default_db = project_root / "data" / "crimevision.db"

    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "crimevision-dev-key"),
        UPLOAD_FOLDER=str(default_upload),
        DATABASE_PATH=os.getenv("DATABASE_PATH", str(default_db)),
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100 MB max upload
    )

    if test_config:
        app.config.update(test_config)

    # Ensure upload and database directories exist
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["DATABASE_PATH"]).parent.mkdir(parents=True, exist_ok=True)

    # Register Blueprints
    from app.routes import api_bp, main_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(main_bp)

    # Common error handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "error": "Payload too large",
            "message": "Uploaded audio file exceeds maximum allowed size (100 MB).",
        }), 413

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not found",
            "message": "Requested endpoint does not exist.",
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred while processing your request.",
        }), 500

    return app
