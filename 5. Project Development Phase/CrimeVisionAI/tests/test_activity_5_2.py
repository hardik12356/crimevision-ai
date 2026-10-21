"""
Activity 5.2 Tests: Grounded Natural-Language Querying and Hypothesis Generation.

Tests:
1. Intent classification: hypothesis, timestamp, general queries.
2. Fusion context is included in case_context when fusion data exists.
3. Hypothesis generation endpoint returns structured response.
4. General Q&A returns query_type and preserves existing behavior.
5. Timestamp Q&A returns query_type 'timestamp' and still works.
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
            {"label": "person", "confidence": 0.89, "box": [100, 150, 300, 450]}
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
        "model": "openai/gpt-oss-20b",
        "raw_content": json.dumps(inner_json),
        "truncated": False,
    }
    groq.answer_case_question.return_value = {
        "model": "openai/gpt-oss-20b",
        "answer": "According to video evidence, an individual was detected at 0.0s.",
        "note": "AI-generated response grounded in case records. Model predictions are unverified by human investigators.",
        "truncated": False,
    }
    groq.generate_hypotheses.return_value = {
        "model": "openai/gpt-oss-20b",
        "hypotheses": [
            {
                "title": "Unauthorized Entry Hypothesis",
                "narrative": "A person was detected entering through the doorway at 0.0s.",
                "supporting_evidence": [
                    "RT-DETR detected 'person' at 0.89 confidence at 0.0s (store_cam.mp4, evi_001)"
                ],
                "contradicting_evidence": ["None identified in available evidence"],
                "assumptions": ["Detected individual is the same across frames"],
                "uncertainty_factors": ["No identity tracking, single-frame detections only"],
                "plausibility": "moderate",
            }
        ],
        "evidence_sufficiency": "limited",
        "overall_limitations": "Only sampled frames available; no continuous tracking.",
        "note": "AI-generated hypotheses grounded in recorded case evidence.",
        "truncated": False,
    }
    # Let classify_query_intent work normally (it's a static method)
    groq.classify_query_intent = GroqService.classify_query_intent
    return groq


@pytest.fixture
def video_service(mock_object_detector, mock_image_captioner, mock_scene_classifier, mock_groq_service):
    return VideoService(
        object_detector=mock_object_detector,
        image_captioner=mock_image_captioner,
        scene_classifier=mock_scene_classifier,
        groq_service=mock_groq_service,
        max_duration_seconds=10.0,
        max_frames=10,
        target_fps=1.0,
    )


@pytest.fixture
def app(tmp_path, video_service, mock_groq_service):
    db_path = tmp_path / "test_5_2.db"
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


@pytest.fixture
def case_with_video(client, mock_groq_service):
    """Create a case with analyzed video evidence, returning (case_id, evidence_id)."""
    case_res = client.post("/api/cases", json={"title": "Activity 5.2 Test Case"})
    case_id = case_res.get_json()["case"]["case_id"]

    video_data = {
        "file": (io.BytesIO(b"dummy video bytes"), "cam_lobby.mp4"),
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
            data=video_data,
            content_type="multipart/form-data",
        )
    assert ev_res.status_code == 201
    ev_id = ev_res.get_json()["evidence"]["evidence_id"]
    return case_id, ev_id


# ---------------------------------------------------------------------------
# 1. Intent Classification
# ---------------------------------------------------------------------------
class TestIntentClassification:
    def test_hypothesis_keywords(self):
        assert GroqService.classify_query_intent("Generate hypotheses about this crime") == "hypothesis"
        assert GroqService.classify_query_intent("What are possible explanations?") == "hypothesis"
        assert GroqService.classify_query_intent("What could have happened here?") == "hypothesis"
        assert GroqService.classify_query_intent("Give me a theory about the incident") == "hypothesis"
        assert GroqService.classify_query_intent("Suggest possible scenarios for the break-in") == "hypothesis"
        assert GroqService.classify_query_intent("Can you hypothesize about what occurred?") == "hypothesis"
        assert GroqService.classify_query_intent("Reconstruct the events based on evidence") == "hypothesis"

    def test_timestamp_queries(self):
        assert GroqService.classify_query_intent("What happened at second 1.0?") == "timestamp"
        assert GroqService.classify_query_intent("Who was visible at 2.5s?") == "timestamp"
        assert GroqService.classify_query_intent("What occurred at 0:02?") == "timestamp"
        assert GroqService.classify_query_intent("Show me frame 12") == "timestamp"

    def test_general_queries(self):
        assert GroqService.classify_query_intent("Who is the primary suspect?") == "general"
        assert GroqService.classify_query_intent("What evidence was collected?") == "general"
        assert GroqService.classify_query_intent("Summarize the case findings") == "general"


# ---------------------------------------------------------------------------
# 2. Fusion Context Inclusion
# ---------------------------------------------------------------------------
def test_fusion_context_included_when_available(client, mock_groq_service, case_with_video):
    """When fusion data exists for a case, it should appear in case_context."""
    case_id, ev_id = case_with_video

    # Manually insert fusion data into the case
    from app.services.case_service import CaseService
    cs = client.application.config["CASE_SERVICE"]
    fusion_payload = {
        "events": [{"event_id": "fe_001", "modality": "video", "description": "Person detected"}],
        "relationships": [{"rel_id": "r1", "relationship_type": "qualified_corroboration"}],
        "anchors": [{"anchor_id": "a1", "reason": "first_detection"}],
        "grounded_narrative": "A person was detected entering the lobby area.",
        "alignment_status": "independent",
        "hash_index": {"fe_001": "abc123"},
    }
    cs.save_case_fusion(case_id, fusion_payload)

    # Send a general question
    chat_res = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "What evidence was collected?"},
    )
    assert chat_res.status_code == 200

    # Verify fusion context was passed to answer_case_question
    call_args = mock_groq_service.answer_case_question.call_args[1]
    context = call_args["case_context"]
    assert "fusion_context" in context
    fc = context["fusion_context"]
    assert len(fc["events"]) == 1
    assert fc["events"][0]["event_id"] == "fe_001"
    assert fc["alignment_status"] == "independent"
    assert "person was detected" in fc["grounded_narrative"].lower()


def test_no_fusion_context_when_absent(client, mock_groq_service, case_with_video):
    """When no fusion data exists, fusion_context should not be in case_context."""
    case_id, ev_id = case_with_video

    chat_res = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "What evidence was collected?"},
    )
    assert chat_res.status_code == 200

    call_args = mock_groq_service.answer_case_question.call_args[1]
    context = call_args["case_context"]
    assert "fusion_context" not in context


# ---------------------------------------------------------------------------
# 3. Hypothesis Generation Endpoint
# ---------------------------------------------------------------------------
def test_hypothesis_query_routes_to_generate_hypotheses(client, mock_groq_service, case_with_video):
    """Hypothesis queries should call generate_hypotheses and return structured response."""
    case_id, ev_id = case_with_video

    chat_res = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "Generate hypotheses about what happened in this case"},
    )
    assert chat_res.status_code == 200
    data = chat_res.get_json()

    # Should route to hypothesis path
    assert data["query_type"] == "hypothesis"
    assert data["status"] == "success"
    assert "hypotheses" in data
    assert len(data["hypotheses"]) == 1
    assert data["hypotheses"][0]["title"] == "Unauthorized Entry Hypothesis"
    assert data["evidence_sufficiency"] == "limited"
    assert "overall_limitations" in data
    assert "note" in data

    # Should NOT have 'answer' key (hypothesis responses have 'hypotheses')
    assert "answer" not in data

    # Verify generate_hypotheses was called, not answer_case_question
    mock_groq_service.generate_hypotheses.assert_called_once()
    mock_groq_service.answer_case_question.assert_not_called()


# ---------------------------------------------------------------------------
# 4. General Q&A Returns query_type
# ---------------------------------------------------------------------------
def test_general_query_returns_query_type(client, mock_groq_service, case_with_video):
    """General queries should include query_type='general' in response."""
    case_id, ev_id = case_with_video

    chat_res = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "What evidence was collected?"},
    )
    assert chat_res.status_code == 200
    data = chat_res.get_json()

    assert data["query_type"] == "general"
    assert data["status"] == "success"
    assert "answer" in data
    assert "hypotheses" not in data

    mock_groq_service.answer_case_question.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Timestamp Q&A Returns query_type
# ---------------------------------------------------------------------------
def test_timestamp_query_returns_query_type(client, mock_groq_service, case_with_video):
    """Timestamp queries should include query_type='timestamp' and still retrieve frame observations."""
    case_id, ev_id = case_with_video

    chat_res = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "What was detected at second 1.0?"},
    )
    assert chat_res.status_code == 200
    data = chat_res.get_json()

    assert data["query_type"] == "timestamp"
    assert data["status"] == "success"
    assert "answer" in data

    # Verify frame observations were retrieved
    call_args = mock_groq_service.answer_case_question.call_args[1]
    context = call_args["case_context"]
    assert len(context["primary_frame_observations"]) >= 1


# ---------------------------------------------------------------------------
# 6. Hypothesis generation with fusion context
# ---------------------------------------------------------------------------
def test_hypothesis_with_fusion_context(client, mock_groq_service, case_with_video):
    """Hypothesis queries should include fusion context if available."""
    case_id, ev_id = case_with_video

    # Insert fusion data
    cs = client.application.config["CASE_SERVICE"]
    cs.save_case_fusion(case_id, {
        "events": [{"event_id": "fe_001", "modality": "video"}],
        "relationships": [],
        "anchors": [],
        "grounded_narrative": "Fusion narrative.",
    })

    chat_res = client.post(
        f"/api/cases/{case_id}/chat",
        json={"question": "What are possible explanations for the incident?"},
    )
    assert chat_res.status_code == 200
    data = chat_res.get_json()
    assert data["query_type"] == "hypothesis"

    # Verify fusion context was passed to generate_hypotheses
    call_args = mock_groq_service.generate_hypotheses.call_args[1]
    context = call_args["case_context"]
    assert "fusion_context" in context


# ---------------------------------------------------------------------------
# 7. Existing tests backward compatibility
# ---------------------------------------------------------------------------
def test_backward_compat_empty_question(client, case_with_video):
    """Empty questions should still return 400."""
    case_id, _ = case_with_video
    res = client.post(f"/api/cases/{case_id}/chat", json={"question": "   "})
    assert res.status_code == 400


def test_backward_compat_missing_case(client):
    """Missing case should still return 404."""
    res = client.post("/api/cases/nonexistent_case/chat", json={"question": "Hello?"})
    assert res.status_code == 404
