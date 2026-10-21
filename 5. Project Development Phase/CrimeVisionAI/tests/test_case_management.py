"""
Unit and integration tests for CrimeVisionAI Case & Evidence Management.
Tests SQLite persistence across restarts, API endpoints, evidence association,
multimodal analysis linking, and timeline aggregation without live models.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app import create_app
from app.services.audio_service import AudioService
from app.services.case_service import CaseService
from app.services.vision_service import VisionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_audio_service():
    service = MagicMock(spec=AudioService)
    service.process_audio.return_value = {
        "status": "success",
        "audio_file": "dispatch_call.wav",
        "recording_duration_seconds": 12.4,
        "transcription": {
            "transcript": "Suspect fleeing north towards 4th Street.",
            "segment_count": 1,
            "segments": [
                {"id": 0, "start": 0.0, "end": 12.4, "text": "Suspect fleeing north towards 4th Street."}
            ],
            "timestamp_quality": "complete",
        },
        "timeline": {
            "timeline_summary": {"total_events": 1, "event_types_found": ["spatial_movement"]},
            "events": [
                {
                    "event_id": "evt_001",
                    "event_type": "spatial_movement",
                    "description": "Suspect fleeing north",
                    "start_raw": 1.5,
                    "end_raw": 8.0,
                    "supporting_quotes": ["Suspect fleeing north"],
                    "uncertainties": ["Speaker identity unverified"],
                }
            ],
        },
    }
    return service


@pytest.fixture
def mock_vision_service():
    service = MagicMock(spec=VisionService)
    service.analyze_image.return_value = {
        "image": "alleyway.jpg",
        "object_detection": {
            "model": "RT-DETR",
            "detection_count": 1,
            "person_detection_count": 1,
            "detections": [
                {"label": "person", "class_id": 0, "confidence": 0.94, "bbox_xyxy": [10, 20, 100, 200]}
            ],
        },
        "scene_classification": {
            "top_scene": "a photograph of an alleyway",
            "candidates": [{"scene": "a photograph of an alleyway", "similarity_score": 0.95}],
        },
        "image_caption": {
            "caption": "a person in a dark jacket walking down an alleyway",
        },
        "llm_analysis": {
            "analysis": "Image indicates a person walking away from camera.",
            "note": "AI-generated interpretation.",
        },
    }
    return service


@pytest.fixture
def test_db_path(tmp_path: Path):
    return tmp_path / "test_crimevision.db"


@pytest.fixture
def test_uploads_path(tmp_path: Path):
    return tmp_path / "uploads"


@pytest.fixture
def client(test_db_path, test_uploads_path, mock_audio_service, mock_vision_service):
    app = create_app({
        "TESTING": True,
        "DATABASE_PATH": str(test_db_path),
        "UPLOAD_FOLDER": str(test_uploads_path),
        "AUDIO_SERVICE": mock_audio_service,
        "VISION_SERVICE": mock_vision_service,
    })
    with app.test_client() as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# 1. Direct CaseService & SQLite Persistence Tests
# ---------------------------------------------------------------------------
def test_case_service_sqlite_persistence_across_restarts(tmp_path: Path):
    """Verify that cases, evidence, and analysis results survive database re-initialization."""
    db_file = tmp_path / "persistence.db"

    # Step A: Initialize first service instance and write records
    service1 = CaseService(db_path=db_file)
    cases_initial = service1.list_cases()
    assert len(cases_initial) == 1
    assert cases_initial[0]["case_id"] == CaseService.DEFAULT_CASE_ID

    created = service1.create_case(
        title="Armed Robbery — Downtown Branch",
        description="Two suspects entered at 14:02.",
        case_id="CASE-2026-TEST-001",
    )
    assert created["case_id"] == "CASE-2026-TEST-001"

    dummy_file = tmp_path / "call.wav"
    dummy_file.write_bytes(b"dummy wav data")

    evidence = service1.add_evidence(
        case_id="CASE-2026-TEST-001",
        original_filename="911_call.wav",
        media_type="audio",
        file_path=str(dummy_file),
        file_size=len(dummy_file.read_bytes()),
    )
    assert evidence["status"] == "pending"

    # Save mock analysis
    service1.save_analysis_result(
        case_id="CASE-2026-TEST-001",
        evidence_id=evidence["evidence_id"],
        analysis_type="audio",
        result_data={"summary": "Audio processed"},
    )

    # Step B: Instantiate a fresh CaseService against the SAME db file (simulating app restart)
    service2 = CaseService(db_path=db_file)
    case_reloaded = service2.get_case("CASE-2026-TEST-001")

    assert case_reloaded is not None
    assert case_reloaded["title"] == "Armed Robbery — Downtown Branch"
    assert len(case_reloaded["evidence"]) == 1
    reloaded_ev = case_reloaded["evidence"][0]
    assert reloaded_ev["original_filename"] == "911_call.wav"
    assert reloaded_ev["status"] == "completed"
    assert reloaded_ev["analysis"] == {"summary": "Audio processed"}


def test_case_service_validations(tmp_path: Path):
    """Verify input validation rules in CaseService."""
    service = CaseService(db_path=tmp_path / "validation.db")

    # Empty title
    with pytest.raises(ValueError, match="required"):
        service.create_case(title="")

    # Invalid characters in case_id
    with pytest.raises(ValueError, match="Invalid case ID"):
        service.create_case(title="Test", case_id="CASE/../../traversal")

    # Duplicate case_id
    service.create_case(title="Original", case_id="CASE-DUP-01")
    with pytest.raises(ValueError, match="already exists"):
        service.create_case(title="Duplicate", case_id="CASE-DUP-01")

    # Non-existent case for evidence
    with pytest.raises(FileNotFoundError):
        service.add_evidence(
            case_id="NON-EXISTENT",
            original_filename="file.wav",
            media_type="audio",
            file_path="dummy",
        )


# ---------------------------------------------------------------------------
# 2. Flask API Endpoint Tests
# ---------------------------------------------------------------------------
def test_api_list_cases(client):
    """Verify GET /api/cases returns default case and status 200."""
    response = client.get("/api/cases")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["cases"]) >= 1
    assert data["cases"][0]["case_id"] == "CASE-2026-MVP-001"


def test_api_create_case_success_and_validation(client):
    """Verify POST /api/cases creation and validation error handling."""
    # Valid creation
    res = client.post(
        "/api/cases",
        json={"title": "Store Break-in", "description": "Rear entrance door pried open."},
    )
    assert res.status_code == 201
    data = res.get_json()
    assert data["status"] == "success"
    assert data["case"]["title"] == "Store Break-in"
    created_id = data["case"]["case_id"]

    # Missing title
    res_bad = client.post("/api/cases", json={"title": "   "})
    assert res_bad.status_code == 400
    assert "Case title is required" in res_bad.get_json()["message"]

    # Duplicate ID
    res_dup = client.post("/api/cases", json={"title": "Again", "case_id": created_id})
    assert res_dup.status_code == 400
    assert "already exists" in res_dup.get_json()["message"]


def test_api_get_case_detail(client):
    """Verify GET /api/cases/<case_id> returns case details and 404 for missing."""
    res_ok = client.get("/api/cases/CASE-2026-MVP-001")
    assert res_ok.status_code == 200
    data = res_ok.get_json()
    assert data["case"]["case_id"] == "CASE-2026-MVP-001"
    assert "evidence" in data["case"]
    assert "timeline_events" in data["case"]

    res_not_found = client.get("/api/cases/NON-EXISTENT-CASE")
    assert res_not_found.status_code == 404


def test_api_upload_audio_evidence_to_case_with_analysis(client, mock_audio_service):
    """Verify uploading audio evidence to a case executes analysis and saves findings."""
    payload = {
        "file": (io.BytesIO(b"RIFF....WAVEfmt ....data...."), "emergency_911.wav"),
        "analyze": "true",
        "language": "en",
    }
    res = client.post(
        "/api/cases/CASE-2026-MVP-001/evidence",
        data=payload,
        content_type="multipart/form-data",
    )
    assert res.status_code == 201
    data = res.get_json()
    assert data["status"] == "success"
    assert data["evidence"]["media_type"] == "audio"
    assert data["evidence"]["original_filename"] == "emergency_911.wav"
    assert data["evidence"]["status"] == "completed"
    assert data["analysis"] is not None
    assert "transcription" in data["analysis"]

    # Verify service was invoked
    mock_audio_service.process_audio.assert_called_once()

    # Verify case detail now contains evidence and chronological timeline event
    case_res = client.get("/api/cases/CASE-2026-MVP-001")
    case_data = case_res.get_json()["case"]
    assert len(case_data["evidence"]) == 1
    assert len(case_data["timeline_events"]) == 1
    assert case_data["timeline_events"][0]["source_filename"] == "emergency_911.wav"


def test_api_upload_image_evidence_to_case_with_analysis(client, mock_vision_service):
    """Verify uploading image evidence to a case executes VisionService and links results."""
    payload = {
        "file": (io.BytesIO(b"\xff\xd8\xff\xe0dummyjpeg"), "suspect_photo.jpg"),
        "analyze": "true",
        "confidence_threshold": "0.35",
    }
    res = client.post(
        "/api/cases/CASE-2026-MVP-001/evidence",
        data=payload,
        content_type="multipart/form-data",
    )
    assert res.status_code == 201
    data = res.get_json()
    assert data["status"] == "success"
    assert data["evidence"]["media_type"] == "image"
    assert data["evidence"]["status"] == "completed"
    assert data["analysis"] is not None
    assert "object_detection" in data["analysis"]

    mock_vision_service.analyze_image.assert_called_once()


def test_api_upload_case_evidence_validations(client):
    """Verify validation errors during case evidence upload."""
    # 1. Non-existent case
    res1 = client.post(
        "/api/cases/UNKNOWN-CASE/evidence",
        data={"file": (io.BytesIO(b"data"), "call.wav")},
        content_type="multipart/form-data",
    )
    assert res1.status_code == 404

    # 2. Missing file
    res2 = client.post(
        "/api/cases/CASE-2026-MVP-001/evidence",
        data={},
        content_type="multipart/form-data",
    )
    assert res2.status_code == 400
    assert res2.get_json()["error"] == "Missing file"

    # 3. Unsupported format
    res3 = client.post(
        "/api/cases/CASE-2026-MVP-001/evidence",
        data={"file": (io.BytesIO(b"shellscript"), "script.sh")},
        content_type="multipart/form-data",
    )
    assert res3.status_code == 400
    assert res3.get_json()["error"] == "Unsupported file format"

    # 4. Empty file (0 bytes)
    res4 = client.post(
        "/api/cases/CASE-2026-MVP-001/evidence",
        data={"file": (io.BytesIO(b""), "empty.wav")},
        content_type="multipart/form-data",
    )
    assert res4.status_code == 400
    assert res4.get_json()["error"] == "Empty file"


def test_api_analyze_existing_pending_evidence(client, mock_audio_service):
    """Verify separate POST /api/cases/<case_id>/evidence/<evidence_id>/analyze endpoint."""
    # Step 1: Upload without immediate analysis
    payload = {
        "file": (io.BytesIO(b"RIFF....WAVEfmt ...."), "interview.wav"),
        "analyze": "false",
    }
    upload_res = client.post(
        "/api/cases/CASE-2026-MVP-001/evidence",
        data=payload,
        content_type="multipart/form-data",
    )
    assert upload_res.status_code == 201
    evidence_id = upload_res.get_json()["evidence"]["evidence_id"]
    assert upload_res.get_json()["evidence"]["status"] == "pending"

    # Step 2: Trigger analysis explicitly
    analyze_res = client.post(
        f"/api/cases/CASE-2026-MVP-001/evidence/{evidence_id}/analyze",
        data={"language": "en"},
        content_type="multipart/form-data",
    )
    assert analyze_res.status_code == 200
    assert analyze_res.get_json()["status"] == "success"
    assert "timeline" in analyze_res.get_json()["analysis"]

    # Step 3: Verify evidence status in case is updated to completed
    case_res = client.get("/api/cases/CASE-2026-MVP-001")
    evidence_list = case_res.get_json()["case"]["evidence"]
    assert any(e["evidence_id"] == evidence_id and e["status"] == "completed" for e in evidence_list)
