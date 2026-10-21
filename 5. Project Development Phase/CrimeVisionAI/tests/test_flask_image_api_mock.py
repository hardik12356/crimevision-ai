"""
Unit tests for Flask API endpoint POST /api/image/analyze using mocks.
Verifies file upload validation, safe disk storage, parameter handling,
and error responses without executing live models or external APIs.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app import create_app
from app.services.vision_service import VisionService


# ---------------------------------------------------------------------------
# Shared mock VisionService response matching the real VisionService schema
# ---------------------------------------------------------------------------
MOCK_VISION_RESULT = {
    "image": "scene_photo.jpg",
    "object_detection": {
        "image": "scene_photo.jpg",
        "model": "RT-DETR",
        "detection_count": 2,
        "person_detection_count": 1,
        "detections": [
            {
                "label": "person",
                "class_id": 0,
                "confidence": 0.9272,
                "bbox_xyxy": [120.0, 80.0, 340.0, 510.0],
            },
            {
                "label": "car",
                "class_id": 3,
                "confidence": 0.7841,
                "bbox_xyxy": [400.0, 200.0, 750.0, 480.0],
            },
        ],
        "count_note": "Person count may overcount due to overlapping bounding boxes.",
    },
    "scene_classification": {
        "model": "openai/clip-vit-base-patch32",
        "device": "cpu",
        "top_scene": "a photograph of a street",
        "candidates": [
            {"scene": "a photograph of a street", "similarity_score": 0.9312},
            {"scene": "a photograph of a parking lot", "similarity_score": 0.8140},
        ],
        "score_note": "Scores are relative across candidates; not calibrated probabilities.",
    },
    "image_caption": {
        "model": "Salesforce/blip-image-captioning-base",
        "caption": "a person standing next to a car on a city street",
        "note": "Automatically generated description; verify against original evidence.",
    },
    "scenario_hypotheses": {
        "status": "pending",
        "items": [],
        "note": "Scenario hypotheses require additional context.",
    },
    "llm_analysis": {
        "model": "openai/gpt-oss-20b",
        "analysis": "The image shows a person near a vehicle on an urban street.",
        "note": "AI-generated interpretation. Requires human review.",
        "truncated": False,
    },
}


@pytest.fixture
def mock_vision_service():
    """Mock VisionService that returns MOCK_VISION_RESULT without loading models."""
    service = MagicMock(spec=VisionService)
    service.analyze_image.return_value = MOCK_VISION_RESULT
    return service


@pytest.fixture
def client(tmp_path: Path, mock_vision_service):
    """Flask test client with isolated upload directory and mocked VisionService."""
    upload_dir = tmp_path / "uploads"
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(upload_dir),
            "VISION_SERVICE": mock_vision_service,
        }
    )
    with app.test_client() as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

def test_analyze_image_success(client, mock_vision_service):
    """Verify successful upload returns HTTP 200 with full vision schema."""
    data = {
        "file": (io.BytesIO(b"\xff\xd8\xff\xe0fake-jpeg-bytes"), "scene_photo.jpg"),
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    json_data = response.get_json()

    # Top-level structure
    assert json_data["image"] == "scene_photo.jpg"
    assert "object_detection" in json_data
    assert "scene_classification" in json_data
    assert "image_caption" in json_data
    assert "llm_analysis" in json_data

    # Detection content
    od = json_data["object_detection"]
    assert od["detection_count"] == 2
    assert od["person_detection_count"] == 1
    assert od["detections"][0]["label"] == "person"
    assert od["detections"][0]["confidence"] == pytest.approx(0.9272)

    # Scene
    assert json_data["scene_classification"]["top_scene"] == "a photograph of a street"

    # Caption
    assert "person" in json_data["image_caption"]["caption"]

    # Service was invoked with the saved path and default threshold
    mock_vision_service.analyze_image.assert_called_once()
    call_kwargs = mock_vision_service.analyze_image.call_args[1]
    assert call_kwargs["confidence_threshold"] == pytest.approx(0.25)


def test_analyze_image_using_image_field(client, mock_vision_service):
    """Verify 'image' multipart field is accepted as an alternative to 'file'."""
    data = {
        "image": (io.BytesIO(b"\xff\xd8\xff\xe0more-bytes"), "evidence.jpg"),
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    mock_vision_service.analyze_image.assert_called_once()


def test_analyze_image_custom_confidence_threshold(client, mock_vision_service):
    """Verify custom confidence_threshold is forwarded to VisionService."""
    data = {
        "file": (io.BytesIO(b"\x89PNG\r\n\x1a\n"), "photo.png"),
        "confidence_threshold": "0.5",
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    call_kwargs = mock_vision_service.analyze_image.call_args[1]
    assert call_kwargs["confidence_threshold"] == pytest.approx(0.5)


def test_analyze_image_png_extension_accepted(client):
    """Verify PNG is an accepted image format."""
    data = {
        "file": (io.BytesIO(b"\x89PNG\r\n\x1a\nfake"), "evidence.png"),
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Validation failure cases
# ---------------------------------------------------------------------------

def test_analyze_image_missing_file_field(client):
    """Verify HTTP 400 when no file field is present."""
    response = client.post(
        "/api/image/analyze",
        data={"confidence_threshold": "0.3"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Missing image file"


def test_analyze_image_empty_filename(client):
    """Verify HTTP 400 when uploaded file has an empty filename."""
    data = {
        "file": (io.BytesIO(b"some bytes"), ""),
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Missing image file"


def test_analyze_image_unsupported_audio_format(client):
    """Verify HTTP 400 when an audio file is uploaded to the image endpoint."""
    data = {
        "file": (io.BytesIO(b"RIFF....WAVEfmt ...."), "recording.mp3"),
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Unsupported image format"
    assert ".mp3" in json_data["message"]


def test_analyze_image_unsupported_exe_format(client):
    """Verify HTTP 400 when a dangerous file extension is uploaded."""
    data = {
        "file": (io.BytesIO(b"MZ\x90\x00malicious"), "malware.exe"),
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Unsupported image format"


def test_analyze_image_empty_file_bytes(client, tmp_path):
    """Verify HTTP 400 when file content is completely empty (0 bytes)."""
    data = {
        "file": (io.BytesIO(b""), "empty_photo.jpg"),
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Empty file"


def test_analyze_image_invalid_confidence_threshold_string(client):
    """Verify HTTP 400 when confidence_threshold is non-numeric."""
    data = {
        "file": (io.BytesIO(b"\xff\xd8\xff\xe0"), "test.jpg"),
        "confidence_threshold": "abc",
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Invalid parameter"


def test_analyze_image_confidence_threshold_out_of_range(client):
    """Verify HTTP 400 when confidence_threshold > 1.0."""
    data = {
        "file": (io.BytesIO(b"\xff\xd8\xff\xe0"), "test.jpg"),
        "confidence_threshold": "1.5",
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Invalid parameter"
    assert "0.0" in json_data["message"] and "1.0" in json_data["message"]


# ---------------------------------------------------------------------------
# Processing failure
# ---------------------------------------------------------------------------

def test_analyze_image_processing_failure_hides_stack_trace(client, mock_vision_service):
    """Verify HTTP 500 on processing failure without exposing stack traces."""
    mock_vision_service.analyze_image.side_effect = RuntimeError("RT-DETR inference crash")

    data = {
        "file": (io.BytesIO(b"\xff\xd8\xff\xe0corrupted"), "photo.jpg"),
    }
    response = client.post(
        "/api/image/analyze",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 500
    json_data = response.get_json()
    assert json_data["error"] == "Processing error"
    assert "Image processing failed" in json_data["message"]
    # No Python traceback in the body
    assert "Traceback" not in response.get_data(as_text=True)
