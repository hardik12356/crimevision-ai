"""
Tests for CrimeVisionAI Dashboard MVP routes and static file availability.
"""

import pytest
from app import create_app


@pytest.fixture
def client():
    """Create a test client for the CrimeVisionAI Flask application."""
    app = create_app({"TESTING": True})
    with app.test_client() as test_client:
        yield test_client


def test_get_dashboard_index(client):
    """Verify that GET / returns HTTP 200 and loads the dashboard HTML structure."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # Core branding and panels
    assert "CrimeVision" in html
    assert "Evidence Ingestion" in html
    assert "Evidence Facts &amp; Transcript" in html or "Evidence Facts & Transcript" in html
    assert "Unified Forensic Timeline" in html
    assert "Investigator AI Q&amp;A Assistant" in html or "Investigator AI Q&A Assistant" in html
    assert "Spatial &amp; Video Analysis" in html or "Spatial & Video Analysis" in html

    # Static asset links
    assert "style.css" in html
    assert "api.js" in html
    assert "app.js" in html


def test_static_assets_served(client):
    """Verify that CSS and JavaScript static files are served successfully."""
    css_res = client.get("/static/css/style.css")
    assert css_res.status_code == 200
    assert "CrimeVisionAI" in css_res.get_data(as_text=True)

    api_js_res = client.get("/static/js/api.js")
    assert api_js_res.status_code == 200
    assert "CrimeVisionAPI" in api_js_res.get_data(as_text=True)

    app_js_res = client.get("/static/js/app.js")
    assert app_js_res.status_code == 200
    assert "AppState" in app_js_res.get_data(as_text=True)


def test_api_audio_endpoint_preserved(client):
    """Verify that existing POST /api/audio/analyze endpoint remains functional."""
    response = client.post("/api/audio/analyze")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Missing audio file"
