"""
Unit tests for TimelineService using mocks.
Verifies Activity 3.2 event extraction, schema compliance, chronological ordering,
deduplication, and error handling without making real network or Groq API calls.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from app.services.timeline_service import TimelineService
from app.services.groq_service import GroqService


@pytest.fixture
def sample_transcription() -> dict:
    """Fixture providing a standard 3-segment transcription with timestamps."""
    return {
        "audio_file": "emergency_call.wav",
        "audio_path": "evidence/emergency_call.wav",
        "model": "openai/whisper-base",
        "device": "cpu",
        "language_hint": "en",
        "detected_language": "en",
        "duration_seconds": 12.5,
        "transcript": (
            "Someone broke into the store on 5th Avenue. "
            "He has a weapon and is threatening the clerk. "
            "The suspect just fled in a black sedan."
        ),
        "segment_count": 3,
        "segments": [
            {
                "id": 0,
                "start": 0.5,
                "end": 3.8,
                "text": "Someone broke into the store on 5th Avenue.",
            },
            {
                "id": 1,
                "start": 4.0,
                "end": 8.2,
                "text": "He has a weapon and is threatening the clerk.",
            },
            {
                "id": 2,
                "start": 8.5,
                "end": 12.0,
                "text": "The suspect just fled in a black sedan.",
            },
        ],
        "timestamp_quality": "complete",
        "has_overlapping_segments": False,
        "limitations": {},
    }


def _make_mock_groq_service(json_content: str) -> GroqService:
    """Helper creating a mocked GroqService returning the given JSON string."""
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json_content),
                finish_reason="stop",
            )
        ]
    )
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = fake_response

    service = MagicMock(spec=GroqService)
    service.extract_audio_events.return_value = {
        "model": "openai/gpt-oss-20b",
        "raw_content": json_content,
        "truncated": False,
    }
    return service


def test_empty_transcript_handling():
    """Verify empty transcript returns 0 events gracefully without calling Groq."""
    mock_groq_service = MagicMock(spec=GroqService)
    timeline_service = TimelineService(groq_service=mock_groq_service)

    empty_payload = {
        "audio_file": "empty.wav",
        "duration_seconds": 0.0,
        "transcript": "",
        "segments": [],
        "timestamp_quality": "none",
        "has_overlapping_segments": False,
    }

    result = timeline_service.generate_timeline(empty_payload)

    assert result["audio_file"] == "empty.wav"
    assert result["timeline_summary"]["total_events"] == 0
    assert result["timeline_summary"]["status"] == "empty_transcript"
    assert result["events"] == []
    # Verify Groq was NOT called
    mock_groq_service.extract_audio_events.assert_not_called()


def test_standard_event_extraction_and_schema(sample_transcription: dict):
    """Verify event extraction, schema fields, start_raw/end_raw derivation, and limitations."""
    groq_output = json.dumps({
        "events": [
            {
                "event_type": "break_in",
                "description": "Store break-in on 5th Avenue",
                "source_segment_ids": [0],
                "supporting_quotes": ["Someone broke into the store on 5th Avenue."],
                "confidence": 0.92,
                "stated_location": "store on 5th Avenue",
                "entities_mentioned": ["store", "someone"],
                "is_allegation": True,
                "uncertainties": ["Caller statement, unconfirmed on scene"],
            },
            {
                "event_type": "threat",
                "description": "Suspect threatening clerk with weapon",
                "source_segment_ids": [1],
                "supporting_quotes": ["He has a weapon and is threatening the clerk."],
                "confidence": 0.88,
                "stated_location": None,
                "entities_mentioned": ["clerk", "weapon", "he"],
                "is_allegation": True,
                "uncertainties": ["Weapon type unverified"],
            },
            {
                "event_type": "suspect_flight",
                "description": "Suspect fled in a black sedan",
                "source_segment_ids": [2],
                "supporting_quotes": ["The suspect just fled in a black sedan."],
                "confidence": 0.90,
                "stated_location": None,
                "entities_mentioned": ["black sedan", "suspect"],
                "is_allegation": True,
                "uncertainties": ["Vehicle license plate unknown"],
            },
        ]
    })

    mock_service = _make_mock_groq_service(groq_output)
    timeline_service = TimelineService(groq_service=mock_service)

    result = timeline_service.generate_timeline(sample_transcription)

    assert result["audio_file"] == "emergency_call.wav"
    assert result["timeline_summary"]["total_events"] == 3
    assert len(result["events"]) == 3

    # Check each event has all required schema fields
    for idx, ev in enumerate(result["events"], start=1):
        assert ev["event_id"] == f"evt_{idx:03d}"
        assert "event_type" in ev
        assert "description" in ev
        assert "start_raw" in ev
        assert "end_raw" in ev
        assert "source_segment_ids" in ev
        assert "supporting_quotes" in ev
        assert "confidence" in ev
        assert "uncertainties" in ev

    # Check start_raw and end_raw mapping to source segment bounds
    ev1 = result["events"][0]
    assert ev1["start_raw"] == 0.5
    assert ev1["end_raw"] == 3.8
    assert ev1["stated_location"] == "store on 5th Avenue"
    assert ev1["is_allegation"] is True

    ev2 = result["events"][1]
    assert ev2["start_raw"] == 4.0
    assert ev2["end_raw"] == 8.2

    ev3 = result["events"][2]
    assert ev3["start_raw"] == 8.5
    assert ev3["end_raw"] == 12.0

    # Forensic limitations present
    assert "timestamp_accuracy" in result["limitations"]
    assert "speaker_attribution" in result["limitations"]
    assert "allegation_vs_fact" in result["limitations"]


def test_chronological_ordering(sample_transcription: dict):
    """Verify events are sorted chronologically by start_raw, with unsequenced events at the end."""
    # Model returns events out of order
    groq_output = json.dumps({
        "events": [
            {
                "event_type": "suspect_flight",
                "description": "Suspect fled",
                "source_segment_ids": [2],
                "supporting_quotes": ["fled in sedan"],
            },
            {
                "event_type": "unsequenced_statement",
                "description": "General claim with no segment",
                "source_segment_ids": [999],  # non-existent segment id -> start_raw will be None
                "supporting_quotes": [],
            },
            {
                "event_type": "break_in",
                "description": "Break-in started",
                "source_segment_ids": [0],
                "supporting_quotes": ["broke into the store"],
            },
        ]
    })

    mock_service = _make_mock_groq_service(groq_output)
    timeline_service = TimelineService(groq_service=mock_service)

    result = timeline_service.generate_timeline(sample_transcription)
    events = result["events"]

    assert len(events) == 3
    # First: break_in at 0.5
    assert events[0]["event_id"] == "evt_001"
    assert events[0]["event_type"] == "break_in"
    assert events[0]["start_raw"] == 0.5

    # Second: suspect_flight at 8.5
    assert events[1]["event_id"] == "evt_002"
    assert events[1]["event_type"] == "suspect_flight"
    assert events[1]["start_raw"] == 8.5

    # Third: unsequenced event with None timestamp placed at the end
    assert events[2]["event_id"] == "evt_003"
    assert events[2]["event_type"] == "unsequenced_statement"
    assert events[2]["start_raw"] is None
    assert events[2]["end_raw"] is None


def test_missing_timestamps_preserves_null(sample_transcription: dict):
    """Verify missing source segment timestamps result in start_raw=None and end_raw=None."""
    # Modify transcription so segments have None timestamps
    for seg in sample_transcription["segments"]:
        seg["start"] = None
        seg["end"] = None
    sample_transcription["timestamp_quality"] = "unavailable"

    groq_output = json.dumps({
        "events": [
            {
                "event_type": "threat",
                "description": "Threat occurred",
                "source_segment_ids": [1],
                "supporting_quotes": ["threatening the clerk"],
            }
        ]
    })

    mock_service = _make_mock_groq_service(groq_output)
    timeline_service = TimelineService(groq_service=mock_service)

    result = timeline_service.generate_timeline(sample_transcription)
    ev = result["events"][0]

    # Must preserve None honestly without manufacturing timestamps
    assert ev["start_raw"] is None
    assert ev["end_raw"] is None
    assert any("Temporal boundary unsupported" in u for u in ev["uncertainties"])


def test_duplicate_events_deduplication(sample_transcription: dict):
    """Verify duplicate events with identical types and quotes are merged into one record."""
    groq_output = json.dumps({
        "events": [
            {
                "event_type": "threat",
                "description": "Threatening clerk with weapon",
                "source_segment_ids": [1],
                "supporting_quotes": ["threatening the clerk"],
            },
            {
                "event_type": "threat",
                "description": "Threatening clerk with weapon",
                "source_segment_ids": [1],
                "supporting_quotes": ["threatening the clerk"],
            },
        ]
    })

    mock_service = _make_mock_groq_service(groq_output)
    timeline_service = TimelineService(groq_service=mock_service)

    result = timeline_service.generate_timeline(sample_transcription)

    # Merged from 2 duplicates into 1
    assert len(result["events"]) == 1
    assert result["events"][0]["event_id"] == "evt_001"
    assert len(result["deduplication_notes"]) >= 1
    assert "Merged duplicate event" in result["deduplication_notes"][0]


def test_overlapping_segments_transparent_note(sample_transcription: dict):
    """Verify overlapping segment flag adds transparency note to timeline."""
    sample_transcription["has_overlapping_segments"] = True

    groq_output = json.dumps({
        "events": [
            {
                "event_type": "arrival",
                "description": "Police arrival",
                "source_segment_ids": [0],
                "supporting_quotes": ["arrived"],
            }
        ]
    })

    mock_service = _make_mock_groq_service(groq_output)
    timeline_service = TimelineService(groq_service=mock_service)

    result = timeline_service.generate_timeline(sample_transcription)

    assert result["timeline_summary"]["has_overlapping_segments"] is True
    assert any("overlapping segments" in note for note in result["deduplication_notes"])


def test_malformed_groq_json_response(sample_transcription: dict):
    """Verify service handles non-JSON / malformed model responses gracefully without crashing."""
    malformed_output = "I could not generate valid JSON. The speaker sounded panicked."

    mock_service = _make_mock_groq_service(malformed_output)
    timeline_service = TimelineService(groq_service=mock_service)

    result = timeline_service.generate_timeline(sample_transcription)

    # Gracefully returns empty event list with status note
    assert result["timeline_summary"]["total_events"] == 0
    assert result["timeline_summary"]["status"] == "no_events_found"
    assert result["events"] == []


def test_markdown_wrapped_json_response(sample_transcription: dict):
    """Verify service correctly strips markdown fences (```json ... ```) from Groq response."""
    wrapped_output = """```json
{
  "events": [
    {
      "event_type": "distress_call",
      "description": "Caller calling 911 for help",
      "source_segment_ids": [0],
      "supporting_quotes": ["broke into the store"],
      "confidence": 0.95
    }
  ]
}
```"""

    mock_service = _make_mock_groq_service(wrapped_output)
    timeline_service = TimelineService(groq_service=mock_service)

    result = timeline_service.generate_timeline(sample_transcription)

    assert len(result["events"]) == 1
    assert result["events"][0]["event_type"] == "distress_call"
    assert result["events"][0]["start_raw"] == 0.5


def test_groq_api_error_handling(sample_transcription: dict):
    """Verify RuntimeError is raised when the Groq client encounters an API or connection error."""
    mock_service = MagicMock(spec=GroqService)
    mock_service.extract_audio_events.side_effect = RuntimeError("Groq API Rate Limit Exceeded")

    timeline_service = TimelineService(groq_service=mock_service)

    with pytest.raises(RuntimeError, match="Event extraction failed"):
        timeline_service.generate_timeline(sample_transcription)
