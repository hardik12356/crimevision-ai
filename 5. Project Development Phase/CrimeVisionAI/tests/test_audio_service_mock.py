"""
Integration tests for AudioService using mocks.
Verifies the end-to-end Epic 3 workflow:
audio file -> transcription (Activity 3.1) -> structured timeline (Activity 3.2) -> saved JSON output.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import wave
import pytest

from app.services.audio_service import AudioService
from app.services.audio_transcriber import AudioTranscriber
from app.services.timeline_service import TimelineService
from app.services.groq_service import GroqService


@pytest.fixture
def dummy_wav_file(tmp_path: Path) -> Path:
    """Create a minimal valid WAV audio file for integration testing."""
    file_path = tmp_path / "emergency_911.wav"
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 32000)  # 1.0 second
    return file_path


def test_process_audio_end_to_end(dummy_wav_file: Path, tmp_path: Path):
    """
    Verify complete integration workflow:
    audio file -> transcription -> timeline -> saved JSON.
    """
    mock_transcription_result = {
        "audio_file": "emergency_911.wav",
        "audio_path": str(dummy_wav_file),
        "model": "openai/whisper-base",
        "device": "cpu",
        "language_hint": "en",
        "detected_language": "en",
        "duration_seconds": 15.0,
        "transcript": (
            "Caller reports a robbery in progress at the convenience store. "
            "Suspect is armed with a handgun. "
            "Officers dispatched to the location."
        ),
        "segment_count": 3,
        "segments": [
            {
                "id": 0,
                "start": 0.5,
                "end": 4.2,
                "text": "Caller reports a robbery in progress at the convenience store.",
            },
            {
                "id": 1,
                "start": 4.5,
                "end": 8.0,
                "text": "Suspect is armed with a handgun.",
            },
            {
                "id": 2,
                "start": 8.2,
                "end": 14.5,
                "text": "Officers dispatched to the location.",
            },
        ],
        "timestamp_quality": "complete",
        "has_overlapping_segments": False,
        "limitations": {
            "forensic_caution": "Unverified ASR output.",
        },
    }

    mock_groq_events_json = json.dumps({
        "events": [
            {
                "event_type": "robbery_report",
                "description": "Report of robbery in progress at convenience store",
                "source_segment_ids": [0],
                "supporting_quotes": ["robbery in progress at the convenience store"],
                "confidence": 0.95,
                "stated_location": "convenience store",
                "entities_mentioned": ["convenience store", "caller"],
                "is_allegation": True,
                "uncertainties": ["Caller statement, unconfirmed on scene"],
            },
            {
                "event_type": "threat_armed_suspect",
                "description": "Suspect reported armed with a handgun",
                "source_segment_ids": [1],
                "supporting_quotes": ["Suspect is armed with a handgun."],
                "confidence": 0.90,
                "stated_location": None,
                "entities_mentioned": ["suspect", "handgun"],
                "is_allegation": True,
                "uncertainties": ["Weapon sighting unconfirmed"],
            },
            {
                "event_type": "police_dispatch",
                "description": "Officers dispatched to scene",
                "source_segment_ids": [2],
                "supporting_quotes": ["Officers dispatched to the location."],
                "confidence": 0.92,
                "stated_location": "the location",
                "entities_mentioned": ["officers"],
                "is_allegation": False,
                "uncertainties": ["Response arrival time not stated"],
            },
        ]
    })

    # Mock Transcriber and Groq Service
    mock_transcriber = MagicMock(spec=AudioTranscriber)
    mock_transcriber.transcribe.return_value = mock_transcription_result

    mock_groq_service = MagicMock(spec=GroqService)
    mock_groq_service.extract_audio_events.return_value = {
        "model": "openai/gpt-oss-20b",
        "raw_content": mock_groq_events_json,
        "truncated": False,
    }

    timeline_service = TimelineService(groq_service=mock_groq_service)
    audio_service = AudioService(
        transcriber=mock_transcriber,
        timeline_service=timeline_service,
    )

    output_json_path = tmp_path / "outputs" / "audio_analysis_test.json"

    # Execute integration
    result = audio_service.process_audio(
        audio_path=dummy_wav_file,
        language="en",
        output_path=output_json_path,
    )

    # 1. Verify top-level structure
    assert result["status"] == "success"
    assert result["audio_file"] == "emergency_911.wav"
    assert result["recording_duration_seconds"] == 15.0
    assert "transcription" in result
    assert "timeline" in result

    # 2. Verify transcription preservation
    assert result["transcription"]["segment_count"] == 3
    assert result["transcription"]["timestamp_quality"] == "complete"
    assert len(result["transcription"]["segments"]) == 3

    # 3. Verify timeline preservation and event details
    timeline = result["timeline"]
    assert timeline["timeline_summary"]["total_events"] == 3
    assert len(timeline["events"]) == 3

    ev1 = timeline["events"][0]
    assert ev1["event_id"] == "evt_001"
    assert ev1["event_type"] == "robbery_report"
    assert ev1["start_raw"] == 0.5
    assert ev1["end_raw"] == 4.2
    assert ev1["source_segment_ids"] == [0]
    assert "robbery in progress" in ev1["supporting_quotes"][0]
    assert ev1["is_allegation"] is True
    assert any("approximate acoustic boundaries" in u for u in ev1["uncertainties"])

    ev2 = timeline["events"][1]
    assert ev2["event_id"] == "evt_002"
    assert ev2["start_raw"] == 4.5
    assert ev2["end_raw"] == 8.0

    ev3 = timeline["events"][2]
    assert ev3["event_id"] == "evt_003"
    assert ev3["start_raw"] == 8.2
    assert ev3["end_raw"] == 14.5

    # 4. Verify saved JSON output
    assert output_json_path.is_file()
    saved_data = json.loads(output_json_path.read_text(encoding="utf-8"))
    assert saved_data["status"] == "success"
    assert saved_data["audio_file"] == "emergency_911.wav"
    assert len(saved_data["timeline"]["events"]) == 3


def test_process_audio_missing_file():
    """Verify FileNotFoundError is raised when target audio file does not exist."""
    audio_service = AudioService()
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        audio_service.process_audio("non_existent_audio.wav")


def test_process_audio_propagates_transcription_error(dummy_wav_file: Path):
    """Verify transcription errors propagate up without being masked."""
    mock_transcriber = MagicMock(spec=AudioTranscriber)
    mock_transcriber.transcribe.side_effect = RuntimeError("Audio decoding corrupted")

    audio_service = AudioService(transcriber=mock_transcriber)
    with pytest.raises(RuntimeError, match="Audio decoding corrupted"):
        audio_service.process_audio(dummy_wav_file)
