"""
Focused Unit and Integration Tests for Activity 6.3:
CrimeVisionAI Forensic PDF Investigative Report Generation.

Tests:
1. Missing case returns 404 JSON error response.
2. Case with minimal/empty evidence generates valid, non-empty A4 PDF with %PDF- header.
3. Multimodal case with image, audio, video, and fusion data compiles all required sections.
4. Text extraction with PyPDF2 verifies all sections A through I and key forensic disclaimers.
5. Missing modalities (e.g. audio-only, image-only) generate gracefully with appropriate fallback notices.
6. Dashboard template contains the PDF report download button with id='btn-export-pdf'.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import PyPDF2

from app import create_app
from app.services.case_service import CaseService
from app.services.fusion_service import FusionService
from app.services.pdf_service import PdfReportService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "test_pdf.db"
    case_service = CaseService(db_path=db_path)
    fusion_service = FusionService()
    pdf_service = PdfReportService()

    app_instance = create_app({
        "TESTING": True,
        "DATABASE_PATH": str(db_path),
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "CASE_SERVICE": case_service,
        "FUSION_SERVICE": fusion_service,
        "PDF_SERVICE": pdf_service,
    })
    return app_instance


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def case_service(app):
    return app.config["CASE_SERVICE"]


# ---------------------------------------------------------------------------
# 1. Error Handling & Validation
# ---------------------------------------------------------------------------
def test_export_pdf_missing_case_returns_404(client):
    """GET /api/cases/<case_id>/export/pdf returns 404 for non-existent case."""
    res = client.get("/api/cases/NON_EXISTENT_CASE_999/export/pdf")
    assert res.status_code == 404
    data = res.get_json()
    assert data["error"] == "Not found"
    assert "does not exist" in data["message"]


# ---------------------------------------------------------------------------
# 2. Minimal / Empty Case PDF Generation
# ---------------------------------------------------------------------------
def test_export_pdf_minimal_case(client, case_service):
    """A newly created case without evidence generates a valid, well-formed PDF."""
    created = case_service.create_case(
        title="Empty Case Investigation",
        description="Investigation with no evidence yet.",
        case_id="CASE-PDF-EMPTY",
    )
    case_id = created["case_id"]

    res = client.get(f"/api/cases/{case_id}/export/pdf")
    assert res.status_code == 200
    assert res.mimetype == "application/pdf"
    assert "attachment" in res.headers.get("Content-Disposition", "")
    assert "crimevision_CASE-PDF-EMPTY_investigative_report.pdf" in res.headers.get("Content-Disposition", "")

    pdf_bytes = res.data
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000

    # Inspect PDF structure with PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1

    extracted_text = " ".join(page.extract_text() for page in reader.pages)
    assert "CrimeVisionAI — Investigative Evidence Report" in extracted_text
    assert "Empty Case Investigation" in extracted_text
    assert "CASE-PDF-EMPTY" in extracted_text
    assert "A. Case Overview" in extracted_text
    assert "B. Evidence Inventory" in extracted_text
    assert "I. Limitations" in extracted_text
    assert "NOTICE OF AI-ASSISTED FORENSIC DECISION SUPPORT" in extracted_text


# ---------------------------------------------------------------------------
# 3. Full Multimodal Case PDF Generation & Section Verification
# ---------------------------------------------------------------------------
def test_export_pdf_full_multimodal_case(client, case_service):
    """Case with image, audio, video, timeline, and fusion data renders all sections A through I."""
    created = case_service.create_case(
        title="High Street Incident Investigation",
        description="Comprehensive multimodal store burglary case.",
        case_id="CASE-PDF-FULL",
    )
    case_id = created["case_id"]

    # 1. Attach Image Evidence with Analysis
    ev_img = case_service.add_evidence(
        case_id=case_id,
        original_filename="entrance_camera.jpg",
        media_type="image",
        file_path="mock/entrance_camera.jpg",
        file_size=245000,
    )
    case_service.save_analysis_result(
        case_id=case_id,
        evidence_id=ev_img["evidence_id"],
        analysis_type="image_analysis",
        result_data={
            "object_detection": {
                "detections": [
                    {"label": "person", "confidence": 0.92, "box": [50, 100, 200, 400]},
                    {"label": "knife", "confidence": 0.84, "box": [120, 210, 180, 260]},
                ],
                "detection_count": 2,
            },
            "scene_classification": {
                "top_scene": "store entrance",
                "confidence": 0.88,
            },
            "image_caption": {
                "caption": "a person holding an object near a glass door",
            },
            "llm_analysis": {
                "analysis": "Subject observed at doorway holding a potential bladed implement.",
            },
        },
    )

    # 2. Attach Audio Evidence with Analysis
    ev_aud = case_service.add_evidence(
        case_id=case_id,
        original_filename="witness_call.wav",
        media_type="audio",
        file_path="mock/witness_call.wav",
        file_size=1048576,
    )
    case_service.save_analysis_result(
        case_id=case_id,
        evidence_id=ev_aud["evidence_id"],
        analysis_type="audio_analysis",
        result_data={
            "recording_duration_seconds": 12.5,
            "transcription": {
                "transcript": "Someone just smashed the side window and ran inside!",
                "detected_language": "English (en)",
                "segments": [
                    {"start": 0.0, "end": 4.5, "text": "Someone just smashed the side window"},
                    {"start": 4.6, "end": 9.2, "text": "and ran inside!"},
                ],
            },
            "timeline": {
                "events": [
                    {
                        "event_id": "aud_01",
                        "start_raw": 1.2,
                        "end_raw": 4.5,
                        "event_type": "property_damage",
                        "description": "Witness reports glass breakage sound",
                        "confidence": 0.90,
                    }
                ]
            },
        },
    )

    # 3. Attach Video Evidence with Analysis
    ev_vid = case_service.add_evidence(
        case_id=case_id,
        original_filename="surveillance_cctv.mp4",
        media_type="video",
        file_path="mock/surveillance_cctv.mp4",
        file_size=5242880,
    )
    case_service.save_analysis_result(
        case_id=case_id,
        evidence_id=ev_vid["evidence_id"],
        analysis_type="video_analysis",
        result_data={
            "duration_seconds": 10.0,
            "sampled_fps": 1.0,
            "incident_hypothesis": {
                "incident_type_hypothesis": "possible_theft_related_activity",
                "incident_hypothesis_reasoning": "Individual visible inside after hours near register.",
            },
            "frame_observations": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "detections": [{"label": "person", "confidence": 0.89}],
                    "caption": "person walking through doorway",
                    "scene": "retail interior",
                },
                {
                    "frame_index": 1,
                    "timestamp_seconds": 1.0,
                    "detections": [{"label": "person", "confidence": 0.91}],
                    "caption": "person approaching counter",
                    "scene": "retail counter",
                },
            ],
            "timeline": {
                "events": [
                    {
                        "event_id": "vid_01",
                        "start_raw": 0.0,
                        "end_raw": 2.0,
                        "event_type": "person_entered_frame",
                        "description": "Person entered through front doorway",
                        "confidence": 0.88,
                    }
                ]
            },
        },
    )

    # 4. Attach Multimodal Fusion Data
    case_service.save_case_fusion(
        case_id=case_id,
        fusion_data={
            "alignment_status": "aligned",
            "grounded_narrative": "Multimodal analysis corroborates audio breakage report with CCTV visual entry.",
            "anchors": [{"anchor_id": "a_01", "reason": "first_visual_detection", "time": 0.0}],
            "events": [
                {
                    "event_id": "fe_001",
                    "timestamp_normalized": 0.0,
                    "end_timestamp_normalized": 2.0,
                    "original_time_basis": "synchronized_cctv",
                    "media_type": "video",
                    "source_filename": "surveillance_cctv.mp4",
                    "source_evidence_id": ev_vid["evidence_id"],
                    "event_type": "person_entered_frame",
                    "description": "Person entered store",
                    "certainty_score": 0.88,
                    "is_anchor": True,
                },
                {
                    "event_id": "fe_002",
                    "timestamp_normalized": 1.2,
                    "end_timestamp_normalized": 4.5,
                    "original_time_basis": "synchronized_audio",
                    "media_type": "audio",
                    "source_filename": "witness_call.wav",
                    "source_evidence_id": ev_aud["evidence_id"],
                    "event_type": "property_damage",
                    "description": "Witness reports glass breakage",
                    "certainty_score": 0.85,
                    "is_anchor": False,
                },
            ],
            "relationships": [
                {
                    "rel_id": "rel_001",
                    "relationship_type": "corroboration",
                    "evidence_sources": ["entrance_camera.jpg", "surveillance_cctv.mp4"],
                    "source_event_ids": ["fe_001"],
                    "description": "Image weapon detection potentially consistent with surveillance posture.",
                    "caveats": ["Identity unverified across cameras"],
                }
            ],
            "hypotheses": [
                {
                    "title": "Forcible Commercial Entry",
                    "plausibility": "high",
                    "narrative": "Perpetrator broke glass and entered retail premises.",
                    "supporting_evidence": ["Audio witness statement at 1.2s", "CCTV person detection at 0.0s"],
                    "contradicting_evidence": ["None identified in available evidence"],
                    "assumptions": ["Glass sound relates to front entrance"],
                    "uncertainty_factors": ["Single witness caller; uncalibrated camera clock"],
                }
            ],
        },
    )

    # 5. Fetch PDF Report
    res = client.get(f"/api/cases/{case_id}/export/pdf")
    assert res.status_code == 200
    assert res.mimetype == "application/pdf"

    reader = PyPDF2.PdfReader(io.BytesIO(res.data))
    assert len(reader.pages) >= 2  # Multi-page report verified

    full_text = " ".join(page.extract_text() for page in reader.pages)

    # Check all sections present
    assert "A. Case Overview & Executive Summary" in full_text
    assert "B. Evidence Inventory" in full_text
    assert "C. Image Analysis Findings" in full_text
    assert "D. Audio Analysis Findings" in full_text
    assert "E. Video Surveillance Findings" in full_text
    assert "F. Unified Forensic Timeline" in full_text
    assert "G. Cross-Modal Evidence Fusion Findings" in full_text
    assert "H. Investigative Hypotheses & Uncertainty" in full_text
    assert "I. Limitations & Investigator-Review Disclaimer" in full_text

    # Check specific evidence values
    assert "entrance_camera.jpg" in full_text
    assert "witness_call.wav" in full_text
    assert "surveillance_cctv.mp4" in full_text
    assert "Someone just smashed the side window" in full_text
    assert "knife" in full_text
    assert "Forcible Commercial Entry" in full_text
    assert "Corroboration" in full_text or "CORROBORATION" in full_text
    # Check neutral header/footer wording
    assert "Human Review Required" in full_text
    assert "CONFIDENTIAL — FORENSIC" not in full_text
    assert "End of Official Case Report" not in full_text


# ---------------------------------------------------------------------------
# 4. Partial / Missing Modality Graceful Handling
# ---------------------------------------------------------------------------
def test_export_pdf_missing_modalities_graceful(client, case_service):
    """Audio-only case explicitly states image and video are not provided."""
    case_res = case_service.create_case(
        title="Audio Only Investigation",
        description="Case with only 911 audio call.",
        case_id="CASE-PDF-AUDIO-ONLY",
    )
    case_id = case_res["case_id"]

    case_service.add_evidence(
        case_id=case_id,
        original_filename="emergency_911.wav",
        media_type="audio",
        file_path="mock/emergency_911.wav",
        file_size=500000,
    )

    res = client.get(f"/api/cases/{case_id}/export/pdf")
    assert res.status_code == 200
    reader = PyPDF2.PdfReader(io.BytesIO(res.data))
    text = " ".join(page.extract_text() for page in reader.pages)

    assert "No image evidence provided" in text
    assert "No video surveillance evidence" in text
    assert "emergency_911.wav" in text


# ---------------------------------------------------------------------------
# 5. Dashboard HTML & Static JS Integration
# ---------------------------------------------------------------------------
def test_dashboard_contains_pdf_export_button(client):
    """Verify that index.html contains the active #btn-export-pdf button."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    assert 'id="btn-export-pdf"' in html
    assert "PDF Report" in html

    # Verify app.js wires btnExportPdf
    js_res = client.get("/static/js/app.js")
    assert js_res.status_code == 200
    js = js_res.get_data(as_text=True)
    assert "btnExportPdf" in js
    assert "/export/pdf" in js
