"""
Mock-based unit and integration tests for CrimeVisionAI Epic 4:
Video Analysis and Timeline Generation (Activities 4.1 & 4.2).

Tests:
1. Video file validation (missing, unsupported extension, zero-byte file).
2. Video frame sampling, timestamp calculation, and limit enforcement (max duration, max frames -> partial status).
3. VideoService sequential analysis pipeline with mocked ObjectDetector, ImageCaptioner, and GroqService.
4. Heuristic visual event extraction fallback when LLM output is unavailable.
5. Standalone video endpoint POST /api/video/analyze (valid and validation failure cases).
6. Case-associated video upload, analysis persistence, and unified timeline merging.
7. Safe media endpoint GET /api/evidence/<evidence_id>/media.
8. Investigator grounded Q&A endpoint POST /api/cases/<case_id>/chat.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app import create_app
from app.routes import _extract_query_target
from app.services.case_service import CaseService
from app.services.groq_service import GroqService
from app.services.image_captioner import ImageCaptioner
from app.services.object_detector import ObjectDetector
from app.services.scene_classifier import SceneClassifier
from app.services.video_service import VideoService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_object_detector():
    detector = MagicMock(spec=ObjectDetector)
    detector.detect.return_value = {
        "model": "RT-DETR",
        "detections": [
            {
                "label": "person",
                "confidence": 0.89,
                "box": [100, 150, 300, 450],
            }
        ],
        "detection_count": 1,
    }
    return detector


@pytest.fixture
def mock_image_captioner():
    captioner = MagicMock(spec=ImageCaptioner)
    captioner.caption.return_value = {
        "model": "BLIP",
        "caption": "a person walking through a doorway",
    }
    return captioner


@pytest.fixture
def mock_scene_classifier():
    classifier = MagicMock(spec=SceneClassifier)
    classifier.classify.return_value = {
        "model": "CLIP",
        "top_scene": "indoor entrance",
        "confidence": 0.82,
        "scene_predictions": [{"scene": "indoor entrance", "confidence": 0.82}],
    }
    return classifier


@pytest.fixture
def mock_groq_service():
    groq = MagicMock(spec=GroqService)
    inner_json = {
        "incident_type_hypothesis": "unauthorized_entry",
        "incident_hypothesis_reasoning": "Person observed entering through doorway at 0.0s.",
        "alternative_hypotheses": ["routine_entry"],
        "events": [
            {
                "event_id": "evt_v001",
                "timestamp_seconds": 0.0,
                "end_timestamp_seconds": 2.0,
                "event_type": "person_entered_frame",
                "description": "Individual entered the frame near the doorway",
                "observed_evidence": "Frame detection at 0.0s showed a person",
                "source_frame_indices": [0],
                "source_timestamps": [0.0],
                "confidence": 0.90,
                "reasoning_limitations": ["Identity unknown from visual data alone"],
            }
        ],
    }
    groq.analyze_video_events.return_value = {
        "model": "llama-3.3-70b-versatile",
        "raw_content": json.dumps(inner_json),
        "truncated": False,
    }
    groq.answer_case_question.return_value = {
        "status": "success",
        "question": "What time did the person enter?",
        "answer": "According to video evidence, an individual entered the frame at 0.0s - 2.0s near the doorway.",
        "note": "AI-generated response grounded in case records.",
    }
    return groq


@pytest.fixture
def video_service(mock_object_detector, mock_image_captioner, mock_scene_classifier, mock_groq_service):
    return VideoService(
        object_detector=mock_object_detector,
        image_captioner=mock_image_captioner,
        scene_classifier=mock_scene_classifier,
        groq_service=mock_groq_service,
        max_duration_seconds=10.0,
        max_frames=40,
        target_fps=4.0,
    )


@pytest.fixture
def mock_video_file(tmp_path):
    """Creates a dummy valid video file for validation checks."""
    v_path = tmp_path / "test_surveillance.mp4"
    v_path.write_bytes(b"dummy video binary stream data")
    return v_path


@pytest.fixture
def app(tmp_path, video_service, mock_groq_service):
    db_path = tmp_path / "test_video.db"
    case_service = CaseService(db_path=db_path)

    app_instance = create_app({
        "TESTING": True,
        "DATABASE_PATH": str(db_path),
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "CASE_SERVICE": case_service,
        "VIDEO_SERVICE": video_service,
        "GROQ_SERVICE": mock_groq_service,
    })
    return app_instance


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Unit Tests: Video Validation & Frame Extraction
# ---------------------------------------------------------------------------
def test_validate_video_file_errors(video_service, tmp_path):
    # Non-existent file
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        video_service._validate_video_file(tmp_path / "missing.mp4")

    # Unsupported extension
    unsupported = tmp_path / "doc.txt"
    unsupported.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported video format"):
        video_service._validate_video_file(unsupported)

    # Empty file
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="Video file is empty"):
        video_service._validate_video_file(empty)


def test_extract_sampled_frames_with_mock_cv2(video_service, mock_video_file):
    """Verifies 4 FPS sampling, accurate timestamps, and OpenCV resource cleanup."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        3: 1280.0,  # CAP_PROP_FRAME_WIDTH
        4: 720.0,   # CAP_PROP_FRAME_HEIGHT
        5: 25.0,    # CAP_PROP_FPS (25 fps)
        7: 75.0,    # CAP_PROP_FRAME_COUNT (3 seconds = 75 frames)
    }.get(prop, 0.0)

    # Return 75 dummy frames then EOF
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frames_sequence = [(True, dummy_frame)] * 75 + [(False, None)]
    mock_cap.read.side_effect = frames_sequence

    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, meta = video_service.extract_sampled_frames(mock_video_file)

    # At 25 FPS with target 4 FPS, step is round(25/4) = 6 frames.
    # Indices sampled: 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72 -> 13 frames
    assert len(frames) == 13
    assert frames[0]["frame_index"] == 0
    assert frames[0]["timestamp_seconds"] == 0.0
    assert frames[1]["frame_index"] == 6
    assert frames[1]["timestamp_seconds"] == 0.24  # round(6 / 25, 2)
    assert frames[2]["frame_index"] == 12
    assert frames[2]["timestamp_seconds"] == 0.48  # round(12 / 25, 2)
    assert frames[4]["frame_index"] == 24
    assert frames[4]["timestamp_seconds"] == 0.96

    # Verify aspect ratio preserved and scaled to max 640
    # Original 1280x720 -> 640x360
    assert frames[0]["frame_data"].shape[1] == 640
    assert frames[0]["frame_data"].shape[0] == 360

    assert meta["status"] == "complete"
    assert meta["extracted_count"] == 13
    assert meta["sampled_fps"] == 4.0
    # OpenCV release called
    mock_cap.release.assert_called_once()


def test_extract_sampled_frames_limit_enforcement(mock_video_file):
    """Verifies that max_frames / max_duration triggers 'partial' status."""
    service = VideoService(max_duration_seconds=1.5, max_frames=2, target_fps=1.0)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {5: 10.0, 7: 100.0}.get(prop, 0.0)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, dummy_frame)] * 50 + [(False, None)]

    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, meta = service.extract_sampled_frames(mock_video_file)

    assert meta["status"] == "partial"
    assert "Exceeded maximum" in meta["partial_reason"]
    assert len(frames) <= 2


# ---------------------------------------------------------------------------
# Unit Tests: Video Pipeline & Heuristics
# ---------------------------------------------------------------------------
def test_analyze_video_pipeline(video_service, mock_video_file):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {5: 4.0, 7: 8.0}.get(prop, 0.0)
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # At 4 FPS video with 4 FPS target, step is 1 -> 8 frames for 8 total frames
    mock_cap.read.side_effect = [(True, dummy_frame)] * 8 + [(False, None)]

    with patch("cv2.VideoCapture", return_value=mock_cap):
        result = video_service.analyze_video(mock_video_file)

    assert result["status"] in ("complete", "partial")
    assert "duration_seconds" in result
    assert "frame_observations" in result
    assert len(result["frame_observations"]) == 8
    assert result["sampled_fps"] == 4.0
    assert "timeline" in result
    assert len(result["timeline"]["events"]) >= 1
    assert "incident_hypothesis" in result
    assert result["incident_hypothesis"]["type"] == "unauthorized_entry"


def test_heuristic_events_fallback():
    service = VideoService()
    observations = [
        {"frame_index": 0, "timestamp_seconds": 0.0, "person_count": 0, "detections": []},
        {"frame_index": 1, "timestamp_seconds": 1.0, "person_count": 1, "detections": [{"label": "person"}]},
        {"frame_index": 2, "timestamp_seconds": 2.0, "person_count": 1, "detections": [{"label": "person"}, {"label": "knife"}]},
        {"frame_index": 3, "timestamp_seconds": 3.0, "person_count": 0, "detections": []},
    ]

    events = service._generate_heuristic_events(observations, max_time=3.0)
    event_types = [e["event_type"] for e in events]

    assert "person_observation" in event_types
    assert "object_detected_in_frame" in event_types
    # Verify events are strictly observations without unfounded tracking/action assertions
    assert len(events) >= 2


# ---------------------------------------------------------------------------
# Integration Tests: Standalone POST /api/video/analyze
# ---------------------------------------------------------------------------
def test_api_video_analyze_validation(client):
    # No file field
    res = client.post("/api/video/analyze", data={})
    assert res.status_code == 400
    assert "Missing video file" in res.get_json()["error"]

    # Unsupported extension
    res = client.post(
        "/api/video/analyze",
        data={"file": (io.BytesIO(b"data"), "suspicious.pdf")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "Unsupported video format" in res.get_json()["error"]

    # Empty file
    res = client.post(
        "/api/video/analyze",
        data={"file": (io.BytesIO(b""), "empty.mp4")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "Empty file" in res.get_json()["error"]


def test_api_video_analyze_success(client):
    data = {"file": (io.BytesIO(b"fake mp4 video bytes"), "hallway.mp4")}
    with patch("cv2.VideoCapture") as mock_cv:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.get.side_effect = lambda prop: {5: 1.0, 7: 1.0}.get(prop, 0.0)
        instance.read.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8)), (False, None)]
        mock_cv.return_value = instance

        res = client.post("/api/video/analyze", data=data, content_type="multipart/form-data")

    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["status"] in ("complete", "partial")
    assert "duration_seconds" in json_data
    assert "frame_observations" in json_data


# ---------------------------------------------------------------------------
# Integration Tests: Case Evidence Association & Timeline Aggregation
# ---------------------------------------------------------------------------
def test_case_video_evidence_and_unified_timeline(client):
    # 1. Create a case
    create_res = client.post(
        "/api/cases",
        json={"title": "Store Break-in", "description": "Investigation into midnight robbery"},
    )
    assert create_res.status_code == 201
    case_id = create_res.get_json()["case"]["case_id"]

    # 2. Upload video evidence with immediate analysis
    video_upload_data = {
        "file": (io.BytesIO(b"fake store footage"), "cctv_entrance.mp4"),
        "media_type": "video",
        "analyze": "true",
    }
    with patch("cv2.VideoCapture") as mock_cv:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.get.side_effect = lambda prop: {5: 1.0, 7: 2.0}.get(prop, 0.0)
        instance.read.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8)), (False, None)]
        mock_cv.return_value = instance

        ev_res = client.post(
            f"/api/cases/{case_id}/evidence",
            data=video_upload_data,
            content_type="multipart/form-data",
        )

    assert ev_res.status_code == 201
    ev_json = ev_res.get_json()
    assert ev_json["status"] == "success"
    assert ev_json["evidence"]["media_type"] == "video"
    evidence_id = ev_json["evidence"]["evidence_id"]

    # 3. Check case detail for unified timeline and evidence list
    case_res = client.get(f"/api/cases/{case_id}")
    assert case_res.status_code == 200
    case_data = case_res.get_json()["case"]

    assert len(case_data["evidence"]) == 1
    assert case_data["evidence"][0]["media_type"] == "video"

    # Unified timeline contains the video events
    timeline = case_data["timeline_events"]
    assert len(timeline) >= 1
    assert timeline[0]["media_type"] == "video"
    assert timeline[0]["source_evidence_id"] == evidence_id
    assert timeline[0]["source_filename"] == "cctv_entrance.mp4"


# ---------------------------------------------------------------------------
# Integration Tests: Safe Media Streaming GET /api/evidence/<id>/media
# ---------------------------------------------------------------------------
def test_evidence_media_streaming(client, tmp_path):
    # Create case and evidence record
    case_res = client.post("/api/cases", json={"title": "Media Test"})
    case_id = case_res.get_json()["case"]["case_id"]

    # Test missing evidence ID
    res_404 = client.get("/api/evidence/non_existent_id/media")
    assert res_404.status_code == 404

    # Upload video without immediate analysis
    res_upload = client.post(
        f"/api/cases/{case_id}/evidence",
        data={
            "file": (io.BytesIO(b"fake media stream bytes"), "recording.mp4"),
            "media_type": "video",
            "analyze": "false",
        },
        content_type="multipart/form-data",
    )
    ev_id = res_upload.get_json()["evidence"]["evidence_id"]

    # Stream the media
    media_res = client.get(f"/api/evidence/{ev_id}/media")
    assert media_res.status_code == 200
    assert media_res.content_type == "video/mp4"
    assert media_res.data == b"fake media stream bytes"


# ---------------------------------------------------------------------------
# Integration Tests: Grounded Investigator Q&A POST /api/cases/<id>/chat
# ---------------------------------------------------------------------------
def test_case_chat_endpoint(client):
    # 1. Create a case
    case_res = client.post("/api/cases", json={"title": "Q&A Test Case"})
    case_id = case_res.get_json()["case"]["case_id"]

    # 2. Test 404 on missing case
    res_missing = client.post("/api/cases/invalid_case/chat", json={"question": "What happened?"})
    assert res_missing.status_code == 404

    # 3. Test 400 on empty question
    res_empty = client.post(f"/api/cases/{case_id}/chat", json={"question": "   "})
    assert res_empty.status_code == 400

    # 4. Valid question
    chat_res = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "What time did the suspect enter at second 1.0?"},
    )
    assert chat_res.status_code == 200
    chat_json = chat_res.get_json()
    assert chat_json["status"] == "success"
    assert "According to video evidence" in chat_json["answer"]


def test_extract_query_target_helper():
    # Explicit seconds
    assert _extract_query_target("What happened at second 1.0?") == {"target_timestamp": 1.0, "type": "timestamp"}
    assert _extract_query_target("Who was visible at 2.5s?") == {"target_timestamp": 2.5, "type": "timestamp"}
    assert _extract_query_target("Check timestamp 3.0 in footage") == {"target_timestamp": 3.0, "type": "timestamp"}
    assert _extract_query_target("What occurred at 0:02?") == {"target_timestamp": 2.0, "type": "timestamp"}
    assert _extract_query_target("Check 01:15 in the recording") == {"target_timestamp": 75.0, "type": "timestamp"}

    # Frame index
    assert _extract_query_target("What is in frame 12?") == {"target_frame_index": 12, "type": "frame"}

    # General query
    assert _extract_query_target("Who is the primary suspect?") == {}


def test_case_chat_retrieves_frame_observations_and_limitations(client, mock_groq_service):
    # 1. Create case
    case_res = client.post("/api/cases", json={"title": "Store Burglary"})
    case_id = case_res.get_json()["case"]["case_id"]

    # 2. Upload video with 2 sampled frames (0.0s, 1.0s)
    video_upload_data = {
        "file": (io.BytesIO(b"dummy store video"), "store_cam.mp4"),
        "media_type": "video",
        "analyze": "true",
    }
    with patch("cv2.VideoCapture") as mock_cv:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.get.side_effect = lambda prop: {5: 1.0, 7: 2.0}.get(prop, 0.0)
        instance.read.side_effect = [
            (True, np.zeros((100, 100, 3), dtype=np.uint8)),
            (True, np.zeros((100, 100, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_cv.return_value = instance

        ev_res = client.post(
            f"/api/cases/{case_id}/evidence",
            data=video_upload_data,
            content_type="multipart/form-data",
        )
    assert ev_res.status_code == 201
    ev_id = ev_res.get_json()["evidence"]["evidence_id"]

    # 3. Query at second 1.0
    chat_res = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "What was detected at second 1.0?"},
    )
    assert chat_res.status_code == 200
    call_args = mock_groq_service.answer_case_question.call_args[1]
    context = call_args["case_context"]

    # Primary source of truth contains the sampled frame
    assert len(context["primary_frame_observations"]) == 1
    frame_obs = context["primary_frame_observations"][0]
    assert frame_obs["evidence_id"] == ev_id
    assert frame_obs["video_filename"] == "store_cam.mp4"
    assert frame_obs["timestamp_seconds"] == 1.0
    assert frame_obs["status"] == "available"
    assert len(frame_obs["model_detections"]) >= 1
    assert frame_obs["model_detections"][0]["label"] == "person"
    assert frame_obs["model_detections"][0]["confidence"] == 0.89

    # 4. Query at second 50.0 (unavailable)
    chat_unavail = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "What is visible at second 50.0?"},
    )
    assert chat_unavail.status_code == 200
    call_args2 = mock_groq_service.answer_case_question.call_args[1]
    context2 = call_args2["case_context"]
    assert len(context2["retrieval_limitations"]) >= 1
    assert "unavailable" in context2["retrieval_limitations"][0].lower()
    assert "50.0s" in context2["retrieval_limitations"][0]


def test_groq_service_answer_case_question_grounding_prompt():
    from types import SimpleNamespace
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="At 1.0s in store_cam.mp4 (Evidence ID: evi_1, Frame 1), the object detector predicted a person with 0.89 confidence."
                ),
                finish_reason="stop",
            )
        ]
    )

    with patch("app.services.groq_service.Groq") as mock_groq_cls:
        mock_client = mock_groq_cls.return_value
        mock_client.chat.completions.create.return_value = fake_response

        service = GroqService()
        result = service.answer_case_question(
            case_context={"case_id": "TEST", "primary_frame_observations": [{"timestamp_seconds": 1.0}]},
            question="What happened at second 1.0?",
        )

        assert "note" in result
        assert "Model predictions are unverified" in result["note"]

        # Inspect prompt passed to Groq API
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]

        # System message instructs caution and separation of predictions from facts
        assert "unverified model predictions" in system_content.lower()
        assert "never as verified facts" in system_content.lower()
        assert "never infer movement, entry, or exit without tracking" in system_content.lower()

        # User prompt contains grounding rules
        assert "PRIMARY SOURCE OF TRUTH" in user_content
        assert "MODEL DETECTIONS VS. VERIFIED FACTS" in user_content
        assert "AVOID UNSUPPORTED MOVEMENT / ENTRY / EXIT CLAIMS" in user_content
        assert "UNAVAILABLE FRAMES OR EVIDENCE" in user_content
