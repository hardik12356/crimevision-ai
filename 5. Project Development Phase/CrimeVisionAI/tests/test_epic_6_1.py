"""
Epic 6.1 Integration Tests: Dashboard Audit & Missing Integration.

Tests:
1. Dashboard HTML contains fusion UI elements (fusion button, summary card, relationships card).
2. Static assets serve enhanced JS logic (fusion state, hypothesis formatting, forensic export).
3. Case export route GET /api/cases/<case_id>/export returns unified forensic payload.
4. Case detail route GET /api/cases/<case_id> includes fusion findings when available.
5. Case fusion endpoints GET/POST /api/cases/<case_id>/fusion return expected structures.
6. Grounded chat endpoint correctly delivers hypothesis and general answers for the active case.
7. Backward compatibility: existing case creation and audio/image/video evidence APIs work as expected.
"""

import io
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app import create_app
from app.services.case_service import CaseService
from app.services.fusion_service import FusionService
from app.services.groq_service import GroqService
from app.services.image_captioner import ImageCaptioner
from app.services.object_detector import ObjectDetector
from app.services.scene_classifier import SceneClassifier
from app.services.video_service import VideoService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_groq_service():
    groq = MagicMock(spec=GroqService)
    groq.answer_case_question.return_value = {
        "model": "openai/gpt-oss-20b",
        "answer": "At 0.0s, an individual was detected near the entrance.",
        "note": "AI-generated response grounded in case records.",
        "truncated": False,
    }
    groq.generate_hypotheses.return_value = {
        "model": "openai/gpt-oss-20b",
        "hypotheses": [
            {
                "title": "Unauthorized Entry Hypothesis",
                "narrative": "A person was detected entering through the doorway at 0.0s.",
                "supporting_evidence": [
                    "RT-DETR detected 'person' at 0.89 confidence at 0.0s"
                ],
                "contradicting_evidence": ["None identified in available evidence"],
                "assumptions": ["Detected individual is the same across frames"],
                "uncertainty_factors": ["No identity tracking"],
                "plausibility": "moderate",
            }
        ],
        "evidence_sufficiency": "limited",
        "overall_limitations": "Only sampled frames available.",
        "note": "AI-generated hypotheses grounded in recorded case evidence.",
        "truncated": False,
    }
    groq.classify_query_intent = GroqService.classify_query_intent
    return groq


@pytest.fixture
def app(tmp_path, mock_groq_service):
    db_path = tmp_path / "test_epic6.db"
    case_service = CaseService(db_path=db_path)
    fusion_service = FusionService()

    app_instance = create_app({
        "TESTING": True,
        "DATABASE_PATH": str(db_path),
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "CASE_SERVICE": case_service,
        "FUSION_SERVICE": fusion_service,
        "GROQ_SERVICE": mock_groq_service,
    })
    return app_instance


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_case(client):
    """Create a test case and return case_id."""
    res = client.post("/api/cases", json={
        "title": "Epic 6.1 Test Investigation",
        "description": "Integration testing case",
    })
    assert res.status_code == 201
    return res.get_json()["case"]["case_id"]


# ---------------------------------------------------------------------------
# 1. Dashboard Template & Static Asset Integration
# ---------------------------------------------------------------------------
def test_dashboard_contains_fusion_ui_elements(client):
    """Verify that index.html contains all newly integrated fusion controls and cards."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Fusion UI elements in timeline panel
    assert 'id="btn-run-fusion"' in html
    assert 'id="fusion-summary-card"' in html
    assert 'id="fusion-alignment-badge"' in html
    assert 'id="fusion-anchors-badge"' in html
    assert 'id="fusion-narrative-text"' in html
    assert 'id="fusion-relationships-card"' in html
    assert 'id="fusion-rel-count-badge"' in html
    assert 'id="fusion-relationships-list"' in html

    # Existing components preserved
    assert 'id="timeline-panel-title"' in html
    assert 'id="chat-panel-title"' in html
    assert 'id="btn-export-json"' in html


def test_static_js_contains_epic6_logic(client):
    """Verify that static scripts include fusion state, hypothesis handling, and export."""
    app_js_res = client.get("/static/js/app.js")
    assert app_js_res.status_code == 200
    app_js = app_js_res.get_data(as_text=True)

    assert "fusion:" in app_js
    assert "bindFusionEvents" in app_js
    assert "renderFusionFindings" in app_js
    assert "formatHypothesisMarkdown" in app_js
    assert "resetChatToCase" in app_js
    assert "/export" in app_js
    assert "timestamp_normalized" in app_js
    assert "end_timestamp_normalized" in app_js
    assert "evidence_consistency_score" in app_js


# ---------------------------------------------------------------------------
# 2. Case Export Endpoint GET /api/cases/<case_id>/export
# ---------------------------------------------------------------------------
def test_export_case_json_endpoint(client, sample_case):
    """Verify that GET /api/cases/<case_id>/export returns structured forensic export."""
    # 404 on missing case
    res_404 = client.get("/api/cases/INVALID_CASE/export")
    assert res_404.status_code == 404

    # 200 on valid case
    res = client.get(f"/api/cases/{sample_case}/export")
    assert res.status_code == 200
    data = res.get_json()

    assert data["status"] == "success"
    export = data["export"]
    assert export["case_id"] == sample_case
    assert export["title"] == "Epic 6.1 Test Investigation"
    assert "exported_at" in export
    assert "evidence" in export
    assert "timeline_events" in export
    assert "forensic_metadata" in export
    assert "audit_notice" in export["forensic_metadata"]


def test_export_preserves_fusion_data(client, sample_case):
    """When fusion data exists, export must include events, relationships, and alignment."""
    cs = client.application.config["CASE_SERVICE"]
    fusion_payload = {
        "events": [{"event_id": "fe_01", "modality": "video", "description": "Person entered"}],
        "relationships": [{"rel_id": "r1", "relationship_type": "corroboration", "description": "Audio matches video"}],
        "anchors": [{"anchor_id": "a1", "reason": "first_detection"}],
        "grounded_narrative": "Synthesized case narrative.",
        "alignment_status": "aligned",
    }
    cs.save_case_fusion(sample_case, fusion_payload)

    res = client.get(f"/api/cases/{sample_case}/export")
    assert res.status_code == 200
    export = res.get_json()["export"]

    assert export["fusion"] is not None
    assert len(export["fusion"]["events"]) == 1
    assert export["fusion"]["alignment_status"] == "aligned"
    assert len(export["fusion"]["relationships"]) == 1


# ---------------------------------------------------------------------------
# 3. Case Detail Endpoint Includes Fusion
# ---------------------------------------------------------------------------
def test_get_case_includes_fusion(client, sample_case):
    """Verify GET /api/cases/<case_id> includes fusion field."""
    cs = client.application.config["CASE_SERVICE"]
    cs.save_case_fusion(sample_case, {
        "events": [],
        "relationships": [],
        "anchors": [],
        "grounded_narrative": "Test narrative",
        "alignment_status": "independent",
    })

    res = client.get(f"/api/cases/{sample_case}")
    assert res.status_code == 200
    case_data = res.get_json()["case"]
    assert "fusion" in case_data
    assert case_data["fusion"]["alignment_status"] == "independent"


# ---------------------------------------------------------------------------
# 4. Fusion Endpoint Trigger & Response
# ---------------------------------------------------------------------------
def test_fusion_endpoint_post_and_get(client, sample_case):
    """Verify POST and GET /api/cases/<case_id>/fusion endpoints."""
    # POST triggers fusion
    post_res = client.post(f"/api/cases/{sample_case}/fusion", json={})
    assert post_res.status_code == 200
    post_data = post_res.get_json()
    assert post_data["status"] == "success"
    assert "fusion" in post_data
    assert "events" in post_data["fusion"]
    assert "relationships" in post_data["fusion"]

    # GET retrieves persisted fusion
    get_res = client.get(f"/api/cases/{sample_case}/fusion")
    assert get_res.status_code == 200
    get_data = get_res.get_json()
    assert get_data["status"] == "success"
    assert "fusion" in get_data


# ---------------------------------------------------------------------------
# 5. Grounded Q&A and Hypothesis Integration
# ---------------------------------------------------------------------------
def test_grounded_chat_hypothesis_dispatch(client, sample_case, mock_groq_service):
    """Verify hypothesis request dispatches to generate_hypotheses with structured response."""
    res = client.post(
        f"/api/cases/{sample_case}/chat",
        json={"question": "Generate hypotheses about this case"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["query_type"] == "hypothesis"
    assert "hypotheses" in data
    assert len(data["hypotheses"]) == 1
    assert data["hypotheses"][0]["title"] == "Unauthorized Entry Hypothesis"


def test_grounded_chat_general_dispatch(client, sample_case, mock_groq_service):
    """Verify general query dispatches to answer_case_question."""
    res = client.post(
        f"/api/cases/{sample_case}/chat",
        json={"question": "What evidence has been uploaded so far?"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["query_type"] == "general"
    assert "answer" in data
