"""
Unit tests for Flask API endpoint POST /api/audio/analyze using mocks.
Verifies file upload validation, safe disk storage, parameter handling,
and error responses without executing live models or external APIs.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app import create_app
from app.services.audio_service import AudioService


@pytest.fixture
def mock_audio_service():
    """Mock AudioService instance."""
    service = MagicMock(spec=AudioService)
    service.process_audio.return_value = {
        "status": "success",
        "audio_file": "sample_call.wav",
        "recording_duration_seconds": 8.5,
        "transcription": {
            "transcript": "Robbery in progress at 5th and Main.",
            "segment_count": 1,
            "segments": [
                {"id": 0, "start": 0.0, "end": 8.5, "text": "Robbery in progress at 5th and Main."}
            ],
            "timestamp_quality": "complete",
        },
        "timeline": {
            "timeline_summary": {
                "total_events": 1,
                "event_types_found": ["robbery_report"],
            },
            "events": [
                {
                    "event_id": "evt_001",
                    "event_type": "robbery_report",
                    "description": "Robbery reported at 5th and Main",
                    "start_raw": 0.0,
                    "end_raw": 8.5,
                }
            ],
            "limitations": {
                "timestamp_accuracy": "Approximate acoustic boundaries.",
            },
        },
    }
    return service


@pytest.fixture
def client(tmp_path: Path, mock_audio_service):
    """Flask test client with isolated upload directory and mocked AudioService."""
    upload_dir = tmp_path / "uploads"
    app = create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": str(upload_dir),
        "AUDIO_SERVICE": mock_audio_service,
    })
    with app.test_client() as test_client:
        yield test_client


def test_analyze_audio_success(client, mock_audio_service):
    """Verify successful upload, service execution, and HTTP 200 response."""
    data = {
        "file": (io.BytesIO(b"RIFF....WAVEfmt ....data...."), "emergency.wav"),
        "language": "en",
    }
    response = client.post(
        "/api/audio/analyze",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["audio_file"] == "sample_call.wav"
    assert "transcription" in json_data
    assert "timeline" in json_data
    assert json_data["timeline"]["events"][0]["event_id"] == "evt_001"

    # Verify service was called with language parameter
    mock_audio_service.process_audio.assert_called_once()
    _, kwargs = mock_audio_service.process_audio.call_args
    assert kwargs["language"] == "en"


def test_analyze_audio_missing_file_field(client):
    """Verify HTTP 400 when no file is present in multipart form."""
    response = client.post(
        "/api/audio/analyze",
        data={"language": "en"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Missing audio file"


def test_analyze_audio_empty_filename(client):
    """Verify HTTP 400 when uploaded file has an empty filename."""
    data = {
        "file": (io.BytesIO(b"empty"), ""),
    }
    response = client.post(
        "/api/audio/analyze",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Missing audio file"


def test_analyze_audio_unsupported_format(client):
    """Verify HTTP 400 when file extension is not a supported audio format."""
    data = {
        "file": (io.BytesIO(b"not an audio file"), "malicious_script.exe"),
    }
    response = client.post(
        "/api/audio/analyze",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Unsupported audio format"
    assert ".exe" in json_data["message"]


def test_analyze_audio_processing_failure_hides_stack_trace(client, mock_audio_service):
    """Verify HTTP 500 when processing fails, without leaking stack traces or secrets."""
    mock_audio_service.process_audio.side_effect = RuntimeError("Whisper decoding engine crashed")

    data = {
        "file": (io.BytesIO(b"corrupted bytes"), "sample.wav"),
    }
    response = client.post(
        "/api/audio/analyze",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 500
    json_data = response.get_json()
    assert json_data["error"] == "Processing error"
    assert "Audio processing failed" in json_data["message"]
    # Ensure no Python traceback is included
    assert "Traceback" not in response.get_data(as_text=True)
