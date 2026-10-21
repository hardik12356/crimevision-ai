"""
Focused Tests for Groq Token Budget Management, Question-Specific Context,
and Rate-Limit Fallbacks.

Verifies:
1. Timestamp-specific Q&A returns grounded observations and citations.
2. Person-detection questions work with compact chronological evidence.
3. Timeline summarization works without transmitting raw frame payloads.
4. Conflict analysis compares only relevant modalities and relationships.
5. Missing modalities and empty evidence are handled honestly.
6. Token budgeting and sliding-window rate tracking prevent exceeding 8,000 TPM.
7. Graceful fallback to deterministic evidence findings when rate-limited.
8. Bounded backoff on retryable provider errors.
"""

import collections
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from app.services.case_service import CaseService
from app.services.context_builder import ContextBuilder, extract_query_target
from app.services.groq_service import GroqService
from app.services.token_budget import (
    GroqRateLimitExceeded,
    TokenBudgetManager,
    default_token_budget,
)


# ---------------------------------------------------------------------------
# Fixtures & Sample Forensic Data
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_case_data():
    """Create a realistic case context with 1 video (many frames), 2 images, and fusion."""
    # Generate 40 sequential video frames (simulating a 40-second video)
    # Most frames have 0 people, but frames 5, 6, 7, 15, 16 have a person.
    frame_observations = []
    for i in range(40):
        t = float(i)
        is_person = i in (5, 6, 7, 15, 16)
        detections = []
        if is_person:
            detections.append({"label": "person", "confidence": 0.88, "box": [10, 20, 100, 200]})
        if i % 3 == 0:
            detections.append({"label": "chair", "confidence": 0.75, "box": [50, 60, 80, 90]})

        frame_observations.append({
            "frame_index": i * 25,
            "timestamp_seconds": t,
            "person_count": 1 if is_person else 0,
            "detection_count": len(detections),
            "detections": detections,
            "caption": "a person walking past a doorway" if is_person else "an empty office hallway",
            "scene": "indoor hallway",
        })

    return {
        "case_id": "CASE-2026-TEST-TPM",
        "title": "Token Budget Forensic Test Case",
        "description": "Validation case for token-efficient Investigator AI queries",
        "evidence": [
            {
                "evidence_id": "evi_video_01",
                "original_filename": "surveillance_hallway_25fps.mp4",
                "media_type": "video",
                "status": "completed",
                "analysis": {
                    "duration_seconds": 40.0,
                    "sampled_fps": 1.0,
                    "frame_observations": frame_observations,
                    "incident_hypothesis": {
                        "incident_type_hypothesis": "possible_trespassing",
                        "incident_hypothesis_reasoning": "Person observed in restricted hallway at 5.0s.",
                    },
                },
            },
            {
                "evidence_id": "evi_image_01",
                "original_filename": "lobby_entrance.jpg",
                "media_type": "image",
                "status": "completed",
                "analysis": {
                    "object_detection": {
                        "detections": [{"label": "person", "confidence": 0.94}],
                    },
                    "image_caption": {"caption": "a person entering lobby"},
                    "scene_classification": {"top_scene": "lobby entrance"},
                },
            },
        ],
        "timeline_events": [
            {
                "event_id": "evt_001",
                "start_raw": 5.0,
                "end_raw": 7.0,
                "event_type": "person_entered_frame",
                "description": "Individual detected in hallway near doorway",
                "source_filename": "surveillance_hallway_25fps.mp4",
                "source_evidence_id": "evi_video_01",
                "media_type": "video",
                "certainty_score": 0.88,
            },
            {
                "event_id": "evt_002",
                "start_raw": 15.0,
                "end_raw": 16.0,
                "event_type": "movement",
                "description": "Individual moved toward exit corridor",
                "source_filename": "surveillance_hallway_25fps.mp4",
                "source_evidence_id": "evi_video_01",
                "media_type": "video",
                "certainty_score": 0.85,
            },
        ],
        "fusion": {
            "alignment_status": "independent",
            "grounded_narrative": "Video surveillance indicates person presence in hallway at 5.0s. Static image confirms person in lobby.",
            "relationships": [
                {
                    "rel_id": "rel_001",
                    "relationship_type": "qualified_corroboration",
                    "source_a_evidence_id": "evi_video_01",
                    "source_b_evidence_id": "evi_image_01",
                    "rationale": "Both files contain automated person detections.",
                    "caveats": "Independent time bases; cannot establish simultaneous physical presence or shared identity.",
                },
                {
                    "rel_id": "rel_002",
                    "relationship_type": "conflict",
                    "source_a_evidence_id": "evi_video_01",
                    "source_b_evidence_id": "evi_image_01",
                    "rationale": "Lighting conditions differ between hallway and lobby captures.",
                    "caveats": "Different camera exposure settings.",
                },
            ],
            "anchors": [
                {
                    "anchor_id": "anc_001",
                    "timestamp_normalized": "5.0s",
                    "event_type": "first_person_detection",
                    "reason": "Earliest appearance of person in video stream",
                    "significance": "Initial observation milestone",
                }
            ],
            "events": [
                {
                    "event_id": "fe_001",
                    "modality": "video",
                    "timestamp_normalized": "5.0s",
                    "description": "Individual observed in hallway",
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# 1. Query Focus Detection Tests
# ---------------------------------------------------------------------------
def test_detect_query_focus():
    """Verify that user queries are correctly classified into targeted forensic focuses."""
    # Timestamp queries
    assert ContextBuilder.detect_query_focus("What is at 5?") == "timestamp"
    assert ContextBuilder.detect_query_focus("What was detected at 0:05?") == "timestamp"
    assert ContextBuilder.detect_query_focus("Show me second 15.0") == "timestamp"
    assert ContextBuilder.detect_query_focus("What happened in frame 25?") == "timestamp"

    # Person detection queries
    assert ContextBuilder.detect_query_focus("Who was observed and at what timestamps?") == "person"
    assert ContextBuilder.detect_query_focus("Were any individuals spotted?") == "person"
    assert ContextBuilder.detect_query_focus("List all person appearances") == "person"

    # Timeline summary queries
    assert ContextBuilder.detect_query_focus("Summarize key timeline events and anchors.") == "timeline"
    assert ContextBuilder.detect_query_focus("What is the chronology of events?") == "timeline"

    # Conflict / cross-modal queries
    assert ContextBuilder.detect_query_focus("Are there any cross-modal conflicts or caveats?") == "conflict"
    assert ContextBuilder.detect_query_focus("Check for discrepancies between video and image") == "conflict"

    # Hypothesis queries
    assert ContextBuilder.detect_query_focus("Generate hypotheses about what happened") == "hypothesis"
    assert ContextBuilder.detect_query_focus("What could have happened here?") == "hypothesis"


# ---------------------------------------------------------------------------
# 2. Token-Efficient Context Pruning Tests
# ---------------------------------------------------------------------------
def test_person_query_context_efficiency(mock_case_data):
    """
    Verify person queries filter out empty frames and produce a compact payload (< 1,500 tokens)
    instead of transmitting the entire 40-frame raw dataset.
    """
    question = "Who was observed and at what timestamps?"
    context = ContextBuilder.build_compact_context(mock_case_data, question, max_input_tokens=4000)

    # Must contain person frames (5, 6, 7, 15, 16) and omit frames with 0 persons
    frames = context.get("primary_frame_observations", [])
    assert len(frames) > 0
    assert len(frames) <= 15
    for f in frames:
        assert f["detected_person_count"] > 0
        assert any(d["label"] == "person" for d in f["model_detections"])

    # Verify overall token estimate is well within safe bounds
    serialized = json.dumps(context)
    token_count = TokenBudgetManager.count_tokens(serialized)
    assert token_count < 1500, f"Person query context was unexpectedly large: {token_count} tokens"


def test_timeline_query_omits_raw_frame_dumps(mock_case_data):
    """
    Verify timeline summarization queries include events and anchors but completely omit
    raw per-frame object detections, bounding boxes, and per-second BLIP captions.
    """
    question = "Summarize key timeline events and anchors."
    context = ContextBuilder.build_compact_context(mock_case_data, question, max_input_tokens=4000)

    # Raw frame observations must NOT be included in timeline summary context
    assert "primary_frame_observations" not in context
    # Timeline events and fusion anchors must be included
    assert len(context.get("derived_timeline_events", [])) >= 2
    assert "anchors" in context.get("fusion_context", {})

    token_count = TokenBudgetManager.count_tokens(json.dumps(context))
    assert token_count < 1200, f"Timeline context was unexpectedly large: {token_count} tokens"


def test_conflict_query_includes_only_relationships_and_modalities(mock_case_data):
    """
    Verify cross-modal conflict queries include fusion relationships, alignment status,
    and representative frame summaries for cross-modal comparison.
    """
    question = "Are there any cross-modal conflicts or caveats?"
    context = ContextBuilder.build_compact_context(mock_case_data, question, max_input_tokens=4000)

    # Conflict context should now include representative frame observations
    frames = context.get("primary_frame_observations", [])
    assert len(frames) <= 4, f"Conflict context should sample at most 4 frames, got {len(frames)}"
    fc = context.get("fusion_context", {})
    assert "relationships" in fc
    assert any(r["type"] == "conflict" for r in fc["relationships"])
    assert fc.get("alignment_status") == "independent"

    token_count = TokenBudgetManager.count_tokens(json.dumps(context))
    assert token_count < 2000


def test_timestamp_query_preserves_nearby_interval(mock_case_data):
    """
    Verify timestamp queries (e.g., 'What is at 5?') isolate the exact or closest frame
    at 5.0s and matching timeline events.
    """
    question = "What is at 5?"
    context = ContextBuilder.build_compact_context(mock_case_data, question, max_input_tokens=4000)

    frames = context.get("primary_frame_observations", [])
    assert len(frames) == 1
    assert frames[0]["timestamp_seconds"] == 5.0
    assert frames[0]["is_exact_timestamp_match"] is True

    # Timeline event spanning 5.0s must be included
    direct_events = context.get("direct_timestamp_timeline_events", [])
    assert len(direct_events) == 1
    assert direct_events[0]["event_id"] == "evt_001"


# ---------------------------------------------------------------------------
# 3. Token Budget & Rate Limiting Enforcement Tests
# ---------------------------------------------------------------------------
def test_sliding_window_rate_limiter():
    """Verify TokenBudgetManager tracks tokens over 60s and detects capacity exhaustion."""
    tb = TokenBudgetManager(tpm_limit=8000)
    tb.reset()

    # Request 1: 3,000 tokens -> should proceed
    can_proceed, wait_time, reason = tb.can_proceed(3000)
    assert can_proceed is True
    assert wait_time == 0.0
    tb.record_usage(3000)

    # Request 2: 3,000 tokens -> should proceed (total 6,000 <= 8,000)
    can_proceed, wait_time, reason = tb.can_proceed(3000)
    assert can_proceed is True
    tb.record_usage(3000)

    # Request 3: 3,000 tokens -> would exceed 8,000 TPM limit (6,000 + 3,000 = 9,000)
    can_proceed, wait_time, reason = tb.can_proceed(3000)
    assert can_proceed is False
    assert "Rate limit capacity reached" in reason or "exceeds" in reason

    # An oversized single request (> 8,000) is immediately rejected
    can_proceed, wait_time, reason = tb.can_proceed(9500)
    assert can_proceed is False
    assert "exceeds total provider limit" in reason


def test_groq_service_raises_rate_limit_exceeded_on_budget_exhaustion(mock_case_data):
    """Verify GroqService raises GroqRateLimitExceeded when token budget is exhausted."""
    tb = TokenBudgetManager(tpm_limit=1000)  # low limit to test enforcement
    tb.record_usage(900)  # 900 / 1000 tokens used

    with patch("app.services.groq_service.Groq"):
        service = GroqService(token_budget=tb)

        # Prompt will require ~600 tokens + 800 output tokens > remaining 100 TPM
        with pytest.raises(GroqRateLimitExceeded) as exc_info:
            service.answer_case_question(mock_case_data, "Who was observed?")

        assert "limit" in str(exc_info.value).lower() or "capacity" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 4. Deterministic Fallback Tests
# ---------------------------------------------------------------------------
def test_deterministic_fallback_person_query(mock_case_data):
    """
    Verify build_deterministic_answer returns exact grounded facts, citations,
    and forensic uncertainty notes without calling any LLM.
    """
    fallback = ContextBuilder.build_deterministic_answer(
        mock_case_data,
        "Who was observed and at what timestamps?",
        focus="person",
        advisory_note="Cloud LLM rate limit reached.",
    )
    answer = fallback["answer"]
    assert "Rate Budget Protection Active" in answer
    assert "surveillance_hallway_25fps.mp4" in answer
    assert "5.0s" in answer
    assert "lobby_entrance.jpg" in answer
    assert "Forensic Observation Limitation" in answer
    assert fallback["is_deterministic_fallback"] is True


def test_deterministic_fallback_conflict_query(mock_case_data):
    """Verify deterministic fallback for cross-modal conflict queries."""
    fallback = ContextBuilder.build_deterministic_answer(
        mock_case_data,
        "Are there any cross-modal conflicts or caveats?",
        focus="conflict",
        advisory_note="High query volume detected.",
    )
    answer = fallback["answer"]
    assert "Cross-Modal Consistency & Caveats" in answer
    assert "independent time bases" in answer.lower()
    assert "Conflict" in answer
    assert "Qualified Corroboration" in answer


# ---------------------------------------------------------------------------
# 5. Flask Route Integration & Graceful Recovery Tests
# ---------------------------------------------------------------------------
@pytest.fixture
def app_with_rate_limit(tmp_path, mock_case_data):
    """Create a Flask test application with mock case service."""
    db_path = tmp_path / "test_chat.db"
    cs = CaseService(db_path=db_path)
    cs.create_case("Rate Test Case", "Test description", case_id="CASE-2026-TEST-TPM")

    # Mock GroqService that raises GroqRateLimitExceeded to test route recovery
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.classify_query_intent = GroqService.classify_query_intent
    mock_groq.answer_case_question.side_effect = GroqRateLimitExceeded(
        "Rate limit reached (8000 TPM limit). Next slot in 45.0s."
    )

    app_instance = create_app({
        "TESTING": True,
        "DATABASE_PATH": str(db_path),
        "CASE_SERVICE": cs,
        "GROQ_SERVICE": mock_groq,
    })

    # Pre-populate case with mock evidence and analysis
    ev = cs.add_evidence(
        case_id="CASE-2026-TEST-TPM",
        original_filename="surveillance_hallway_25fps.mp4",
        media_type="video",
        file_path="dummy.mp4",
        file_size=1000,
    )
    cs.save_analysis_result(
        case_id="CASE-2026-TEST-TPM",
        evidence_id=ev["evidence_id"],
        analysis_type="video",
        result_data=mock_case_data["evidence"][0]["analysis"],
    )
    cs.save_case_fusion("CASE-2026-TEST-TPM", mock_case_data["fusion"])
    return app_instance


def test_route_gracefully_recovers_from_rate_limit(app_with_rate_limit):
    """
    Verify that when Groq raises GroqRateLimitExceeded, the route returns HTTP 200
    with a deterministic grounded answer rather than crashing with HTTP 500 or 413.
    """
    client = app_with_rate_limit.test_client()

    res = client.post(
        "/api/cases/CASE-2026-TEST-TPM/chat",
        json={"question": "Who was observed and at what timestamps?"},
    )
    assert res.status_code == 200
    data = res.get_json()

    assert data["status"] == "success"
    assert data["rate_limited"] is True
    assert "Rate Budget Protection Active" in data["answer"]
    assert "surveillance_hallway_25fps.mp4" in data["answer"]
    assert "Deterministic findings compiled directly from stored case evidence" in data["note"]

# ---------------------------------------------------------------------------
# 6. Audio Fusion, Cross-Modal Corroboration & Conflict Context Tests
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_case_data_with_audio():
    """Case with video (person frames), image (person detection), and audio (witness transcript)."""
    frame_observations = []
    for i in range(8):
        t = float(i)
        detections = [{"label": "person", "confidence": 0.85, "box": [10, 20, 100, 200]}]
        if i % 2 == 0:
            detections.append({"label": "laptop", "confidence": 0.70, "box": [50, 60, 80, 90]})
        frame_observations.append({
            "frame_index": i * 25,
            "timestamp_seconds": t,
            "person_count": 1,
            "detection_count": len(detections),
            "detections": detections,
            "caption": "a person at a desk",
            "scene": "office",
        })

    return {
        "case_id": "CASE-2026-AUDIO-TEST",
        "title": "Audio Fusion Integration Test Case",
        "evidence": [
            {
                "evidence_id": "evi_vid_aud",
                "original_filename": "shop_camera.mp4",
                "media_type": "video",
                "status": "completed",
                "analysis": {
                    "duration_seconds": 7.0,
                    "sampled_fps": 1.0,
                    "frame_observations": frame_observations,
                    "timeline": {
                        "events": [
                            {
                                "event_id": "vt_001",
                                "start_raw": 0.0,
                                "end_raw": 7.0,
                                "event_type": "scene_transition",
                                "description": "Continuous scene with person at desk",
                                "confidence": 0.80,
                            }
                        ]
                    },
                },
            },
            {
                "evidence_id": "evi_img_aud",
                "original_filename": "shop_scene.png",
                "media_type": "image",
                "status": "completed",
                "analysis": {
                    "object_detection": {
                        "detections": [
                            {"label": "person", "confidence": 0.90},
                            {"label": "laptop", "confidence": 0.75},
                        ],
                    },
                    "image_caption": {"caption": "a person at a counter"},
                    "scene_classification": {"top_scene": "shop"},
                },
            },
            {
                "evidence_id": "evi_aud_witness",
                "original_filename": "witness_statement.wav",
                "media_type": "audio",
                "status": "completed",
                "analysis": {
                    "duration_seconds": 20.0,
                    "transcription": {
                        "transcript": "I saw a customer at the counter. He had a green jacket.",
                        "segments": [
                            {"id": 0, "start": 0.0, "end": 4.5, "text": " I saw a customer at the counter."},
                            {"id": 1, "start": 4.5, "end": 8.0, "text": " He had a green jacket."},
                            {"id": 2, "start": 8.0, "end": 12.0, "text": " Then he walked toward the doorway."},
                        ],
                    },
                    "timeline": {"events": []},  # Empty — triggers fallback ingestion
                },
            },
        ],
    }


def test_audio_transcript_segments_ingested_as_fusion_events(mock_case_data_with_audio):
    """
    Verify that when audio timeline events are empty, transcript segments are
    ingested as speech_statement ForensicEvent records in the fusion output.
    """
    from app.services.fusion_service import FusionService

    fs = FusionService()
    result = fs.fuse_case(mock_case_data_with_audio)

    events = result["events"]
    audio_events = [e for e in events if e["modality"] == "audio"]

    # All 3 transcript segments should produce speech_statement events
    assert len(audio_events) >= 3, f"Expected >= 3 audio events, got {len(audio_events)}"

    speech_events = [e for e in audio_events if e["event_type"] == "speech_statement"]
    assert len(speech_events) >= 3, f"Expected >= 3 speech_statement events, got {len(speech_events)}"

    # Verify entity extraction from segment text
    seg0 = next((e for e in speech_events if "customer" in e.get("description", "").lower()), None)
    assert seg0 is not None, "Expected a speech event mentioning 'customer'"
    assert "customer" in seg0.get("extracted_entities", []) or "customer" in seg0.get("observed_labels", [])

    # Verify timestamps are preserved
    for se in speech_events:
        assert se.get("timestamp_normalized") is not None, "Speech event should have normalized timestamp"


def test_cross_modal_person_corroboration_generated(mock_case_data_with_audio):
    """
    Verify that fusion generates qualified_corroboration relationships between
    audio witness person mentions and visual person detections.
    """
    from app.services.fusion_service import FusionService

    fs = FusionService()
    result = fs.fuse_case(mock_case_data_with_audio)

    relationships = result["relationships"]
    person_corroborations = [
        r for r in relationships
        if r["relationship_type"] in ("thematic_match", "qualified_corroboration")
        and r["matching_rule"] == "spoken_person_visual_person_cross_modal"
    ]
    assert len(person_corroborations) > 0, (
        f"Expected cross-modal person corroborations, got 0. "
        f"Total relationships: {len(relationships)}, types: {[r['relationship_type'] for r in relationships]}"
    )

    # Verify caveats about independent time bases
    for rel in person_corroborations:
        assert any("independent" in u.lower() or "biometric" in u.lower() for u in rel.get("uncertainties", []))


def test_witness_statement_anchors_generated(mock_case_data_with_audio):
    """
    Verify that speech_statement events with entity mentions produce
    witness_statement anchors.
    """
    from app.services.fusion_service import FusionService

    fs = FusionService()
    result = fs.fuse_case(mock_case_data_with_audio)

    anchors = result["anchors"]
    witness_anchors = [a for a in anchors if a.get("category") == "witness_statement"]
    assert len(witness_anchors) > 0, (
        f"Expected witness_statement anchors, got 0. "
        f"All anchors: {[a['category'] for a in anchors]}"
    )


def test_fusion_event_count_includes_audio(mock_case_data_with_audio):
    """
    Verify total fusion event count includes audio-derived speech_statement events,
    and that relationships and anchors are non-zero.
    """
    from app.services.fusion_service import FusionService

    fs = FusionService()
    result = fs.fuse_case(mock_case_data_with_audio)

    summary = result["summary"]
    assert summary["total_events"] >= 6, f"Expected >= 6 total events (video + image + audio), got {summary['total_events']}"
    assert any(e["modality"] == "audio" for e in result["events"]), "Audio events missing from fusion output"
    # Relationships should now be non-zero due to person corroborations
    assert len(result["relationships"]) > 0, "Expected non-zero relationships with audio person mentions"
    assert summary["anchor_events_count"] > 0, "Expected non-zero anchors"


def test_conflict_context_includes_frame_observations(mock_case_data):
    """
    Verify that conflict-focus context now includes representative frame
    observations and does not omit them.
    """
    question = "Are there any cross-modal conflicts or caveats?"
    context = ContextBuilder.build_compact_context(mock_case_data, question, max_input_tokens=4000)

    # Conflict context should now include representative frames
    frames = context.get("primary_frame_observations", [])
    assert len(frames) > 0, "Conflict context should include representative frame observations"

    # Verify video summary includes analyzed_frame_count
    vs = context.get("video_summaries", [])
    assert len(vs) > 0
    video_entry = next((v for v in vs if v.get("media_type") == "video"), None)
    if video_entry:
        assert "analyzed_frame_count" in video_entry


def test_person_query_context_filters_detections(mock_case_data):
    """
    Verify that person-focused context only includes person-label detections
    in frame observations, not chairs/books/etc.
    """
    question = "Who was observed and at what timestamps?"
    context = ContextBuilder.build_compact_context(mock_case_data, question, max_input_tokens=4000)

    frames = context.get("primary_frame_observations", [])
    for f in frames:
        for det in f.get("model_detections", []):
            assert det["label"] in ("person", "man", "woman", "individual"), (
                f"Non-person detection '{det['label']}' leaked into person-focus context"
            )
