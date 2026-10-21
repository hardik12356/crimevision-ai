"""
Unit tests for AudioTranscriber service using mocks.
Verifies offline functionality without downloading weights, accessing network, or calling Groq.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import wave
import pytest

from app.services.audio_transcriber import AudioTranscriber


@pytest.fixture
def dummy_wav_file(tmp_path: Path) -> Path:
    """Create a valid 1.0-second WAV file with standard headers for testing."""
    file_path = tmp_path / "emergency_call.wav"
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # Write 16,000 samples of 16-bit silence (1 second duration)
        wf.writeframes(b"\x00" * 32000)
    return file_path


@pytest.fixture
def dummy_mp3_file(tmp_path: Path) -> Path:
    """Create a dummy non-empty MP3 file for testing."""
    file_path = tmp_path / "statement.mp3"
    file_path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00dummy_audio_bytes")
    return file_path


def test_lazy_loading():
    """Verify that model pipeline is NOT loaded upon instantiation."""
    with patch("transformers.pipeline") as mock_pipeline:
        transcriber = AudioTranscriber()
        assert transcriber._pipeline is None
        mock_pipeline.assert_not_called()


def test_missing_file_raises_error():
    """Verify FileNotFoundError is raised for non-existent files."""
    transcriber = AudioTranscriber()
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcriber.transcribe("non_existent_audio.wav")


def test_unsupported_format_raises_error(tmp_path: Path):
    """Verify ValueError is raised for unsupported file extensions."""
    invalid_file = tmp_path / "evidence.txt"
    invalid_file.write_text("not an audio file", encoding="utf-8")

    transcriber = AudioTranscriber()
    with pytest.raises(ValueError, match="Unsupported audio format"):
        transcriber.transcribe(str(invalid_file))


def test_empty_file_raises_error(tmp_path: Path):
    """Verify ValueError is raised for empty 0-byte audio files."""
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()

    transcriber = AudioTranscriber()
    with pytest.raises(ValueError, match="Audio file is empty"):
        transcriber.transcribe(str(empty_file))


def test_model_loading_failure(dummy_wav_file: Path):
    """Verify RuntimeError is raised when pipeline fails to load."""
    with patch("transformers.pipeline", side_effect=Exception("Weights missing")):
        transcriber = AudioTranscriber()
        with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
            transcriber.transcribe(str(dummy_wav_file))


def test_inference_failure(dummy_wav_file: Path):
    """Verify RuntimeError is raised when pipeline inference throws an error."""
    mock_pipe = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
    with patch("transformers.pipeline", return_value=mock_pipe):
        transcriber = AudioTranscriber()
        with pytest.raises(RuntimeError, match="Transcription failed for"):
            transcriber.transcribe(str(dummy_wav_file))


def test_successful_transcription_and_schema(dummy_wav_file: Path):
    """Verify successful transcription, segment parsing, duration, and complete schema."""
    fake_output = {
        "text": "Help, there is an incident here.",
        "chunks": [
            {"timestamp": (0.0, 0.45), "text": "Help,"},
            {"timestamp": (0.45, 0.95), "text": "there is an incident here."},
        ],
        "language": "en",
    }

    mock_pipe = MagicMock(return_value=fake_output)
    with patch("transformers.pipeline", return_value=mock_pipe):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file), language="en")

        # Schema structure assertions
        assert result["audio_file"] == "emergency_call.wav"
        assert result["model"] == "openai/whisper-base"
        assert result["device"] == "cpu"
        assert result["language_hint"] == "en"
        assert result["detected_language"] == "en"
        assert result["transcript"] == "Help, there is an incident here."
        assert result["segment_count"] == 2
        assert result["timestamp_quality"] == "complete"
        assert result["has_overlapping_segments"] is False
        assert result["duration_seconds"] == 1.0  # From WAV header metadata

        # Segment contents
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 0.45
        assert result["segments"][0]["text"] == "Help,"

        assert result["segments"][1]["start"] == 0.45
        assert result["segments"][1]["end"] == 0.95
        assert result["segments"][1]["text"] == "there is an incident here."

        # Forensic caveats present
        assert "forensic_caution" in result["limitations"]
        assert "timestamp_accuracy" in result["limitations"]
        assert "audio_quality_impact" in result["limitations"]

        # Assert correct generate_kwargs passed
        mock_pipe.assert_called_once()
        _, kwargs = mock_pipe.call_args
        assert kwargs["generate_kwargs"]["language"] == "en"
        assert kwargs["generate_kwargs"]["task"] == "transcribe"


def test_multilingual_support_telugu_hindi_french(dummy_mp3_file: Path):
    """Verify multilingual transcription options for Telugu, Hindi, and French."""
    test_cases = [
        ("te", "సహాయం చేయండి", "te"),
        ("hi", "मदद करो, यहाँ आपातकाल है", "hi"),
        ("fr", "Au secours, il y a une urgence", "fr"),
    ]

    for lang_code, text, expected_lang in test_cases:
        fake_output = {
            "text": text,
            "chunks": [{"timestamp": (0.0, 3.2), "text": text}],
            "language": expected_lang,
        }
        mock_pipe = MagicMock(return_value=fake_output)
        with patch("transformers.pipeline", return_value=mock_pipe):
            transcriber = AudioTranscriber(device="cpu")
            result = transcriber.transcribe(str(dummy_mp3_file), language=lang_code)

            assert result["language_hint"] == lang_code
            assert result["detected_language"] == expected_lang
            assert result["transcript"] == text
            assert result["segments"][0]["text"] == text


def test_language_provenance_distinction(dummy_wav_file: Path):
    """Verify caller language_hint is never conflated with model-detected_language."""
    # Scenario A: Caller provides hint, model returns no detected language
    fake_output_no_model_lang = {
        "text": "Bonjour",
        "chunks": [{"timestamp": (0.0, 1.0), "text": "Bonjour"}],
    }
    mock_pipe_a = MagicMock(return_value=fake_output_no_model_lang)
    with patch("transformers.pipeline", return_value=mock_pipe_a):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file), language="fr")
        assert result["language_hint"] == "fr"
        assert result["detected_language"] is None

    # Scenario B: Caller provides no hint, model auto-detects Telugu
    fake_output_with_detected_lang = {
        "text": "నమస్కారం",
        "chunks": [{"timestamp": (0.0, 1.0), "text": "నమస్కారం"}],
        "language": "te",
    }
    mock_pipe_b = MagicMock(return_value=fake_output_with_detected_lang)
    with patch("transformers.pipeline", return_value=mock_pipe_b):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file), language=None)
        assert result["language_hint"] is None
        assert result["detected_language"] == "te"


def test_negative_and_reversed_timestamps_rejected(dummy_wav_file: Path):
    """Verify that negative timestamps and reversed (start > end) timestamps are rejected."""
    fake_output_invalid = {
        "text": "Valid chunk then negative chunk then reversed chunk.",
        "chunks": [
            {"timestamp": (0.0, 2.0), "text": "Valid chunk"},
            {"timestamp": (-1.5, 2.0), "text": "then negative chunk"},
            {"timestamp": (5.0, 3.0), "text": "then reversed chunk."},
        ],
    }

    mock_pipe = MagicMock(return_value=fake_output_invalid)
    with patch("transformers.pipeline", return_value=mock_pipe), \
         patch.object(AudioTranscriber, "_get_audio_duration", return_value=10.0):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file))

        # First chunk remains valid
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.0

        # Negative timestamp chunk must be invalidated
        assert result["segments"][1]["start"] is None
        assert result["segments"][1]["end"] is None

        # Reversed timestamp chunk must be invalidated
        assert result["segments"][2]["start"] is None
        assert result["segments"][2]["end"] is None

        # Quality cannot be 'complete'
        assert result["timestamp_quality"] == "partial"


def test_overlapping_segments_detection(dummy_wav_file: Path):
    """Verify overlapping segment detection derives True without modifying timestamps."""
    fake_output_overlapping = {
        "text": "Speaker A talks. Speaker B overlaps.",
        "chunks": [
            {"timestamp": (0.0, 3.5), "text": "Speaker A talks."},
            {"timestamp": (2.8, 5.0), "text": "Speaker B overlaps."},
        ],
    }

    mock_pipe = MagicMock(return_value=fake_output_overlapping)
    with patch("transformers.pipeline", return_value=mock_pipe), \
         patch.object(AudioTranscriber, "_get_audio_duration", return_value=10.0):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file))

        assert result["has_overlapping_segments"] is True
        # Timestamps preserved without shifting
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 3.5
        assert result["segments"][1]["start"] == 2.8
        assert result["segments"][1]["end"] == 5.0


def test_strictly_sequential_segments_have_no_overlap(dummy_wav_file: Path):
    """Verify strictly sequential segments return has_overlapping_segments=False."""
    fake_output_sequential = {
        "text": "First. Second.",
        "chunks": [
            {"timestamp": (0.0, 2.0), "text": "First."},
            {"timestamp": (2.0, 4.0), "text": "Second."},
        ],
    }

    mock_pipe = MagicMock(return_value=fake_output_sequential)
    with patch("transformers.pipeline", return_value=mock_pipe), \
         patch.object(AudioTranscriber, "_get_audio_duration", return_value=10.0):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file))

        assert result["has_overlapping_segments"] is False


def test_unavailable_duration_returns_none_without_speech_fallback(dummy_mp3_file: Path):
    """Verify duration_seconds is None when metadata probing fails; never falls back to speech end."""
    fake_output = {
        "text": "Speaking for 10 seconds in a long recording.",
        "chunks": [{"timestamp": (0.0, 10.0), "text": "Speaking for 10 seconds."}],
    }

    mock_pipe = MagicMock(return_value=fake_output)
    with patch("transformers.pipeline", return_value=mock_pipe):
        with patch.object(AudioTranscriber, "_get_audio_duration", return_value=None):
            transcriber = AudioTranscriber(device="cpu")
            result = transcriber.transcribe(str(dummy_mp3_file))

            # Must NOT substitute 10.0 for duration_seconds
            assert result["duration_seconds"] is None
            assert result["segments"][0]["end"] == 10.0


def test_malformed_and_missing_timestamps_honesty(dummy_wav_file: Path):
    """Verify that malformed or missing timestamps are not fabricated."""
    fake_output_malformed = {
        "text": "First chunk then second chunk with broken time.",
        "chunks": [
            {"timestamp": (0.5, 2.0), "text": "First chunk"},
            {"timestamp": (None, "invalid"), "text": "then second chunk with broken time."},
        ],
    }

    mock_pipe = MagicMock(return_value=fake_output_malformed)
    with patch("transformers.pipeline", return_value=mock_pipe), \
         patch.object(AudioTranscriber, "_get_audio_duration", return_value=10.0):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file))

        # Check first chunk
        assert result["segments"][0]["start"] == 0.5
        assert result["segments"][0]["end"] == 2.0

        # Check second chunk: must NOT fabricate timestamps
        assert result["segments"][1]["start"] is None
        assert result["segments"][1]["end"] is None

        # Timestamp quality should honestly indicate partial timestamps
        assert result["timestamp_quality"] == "partial"


def test_no_chunks_fallback(dummy_wav_file: Path):
    """Verify behavior when model returns text but no chunks."""
    fake_output_no_chunks = {
        "text": "Audio text with no chunk data provided.",
        "chunks": [],
    }

    mock_pipe = MagicMock(return_value=fake_output_no_chunks)
    with patch("transformers.pipeline", return_value=mock_pipe):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file))

        assert result["transcript"] == "Audio text with no chunk data provided."
        assert result["segment_count"] == 1
        assert result["segments"][0]["start"] is None
        assert result["segments"][0]["end"] is None
        assert result["timestamp_quality"] == "unavailable"


def test_segment_end_clamped_to_audio_duration(dummy_wav_file: Path):
    """
    Verify segment end timestamps are strictly clamped to actual audio duration.
    E.g., if Whisper chunk timestamp reports 19.8s on a 19.7s recording,
    the normalized segment must not exceed 19.7s.
    """
    fake_output_exceeding_duration = {
        "text": "Final acoustic speech segment extending past duration.",
        "chunks": [
            {"timestamp": (0.0, 5.0), "text": "Beginning of recording,"},
            {"timestamp": (5.0, 19.8), "text": "Final acoustic speech segment extending past duration."},
        ],
        "language": "en",
    }

    mock_pipe = MagicMock(return_value=fake_output_exceeding_duration)
    with patch("transformers.pipeline", return_value=mock_pipe):
        with patch.object(AudioTranscriber, "_get_audio_duration", return_value=19.7):
            transcriber = AudioTranscriber(device="cpu")
            result = transcriber.transcribe(str(dummy_wav_file))

            assert result["duration_seconds"] == 19.7
            assert len(result["segments"]) == 2
            assert result["segments"][0]["start"] == 0.0
            assert result["segments"][0]["end"] == 5.0

            # Second segment must be clamped to duration 19.7
            assert result["segments"][1]["start"] == 5.0
            assert result["segments"][1]["end"] == 19.7
            assert result["segments"][1]["end"] <= result["duration_seconds"]


def test_genuinely_empty_transcription_handling(dummy_wav_file: Path):
    """Verify genuinely empty transcription produces empty transcript and zero segments."""
    fake_output_empty = {
        "text": "",
        "chunks": [],
        "language": "en",
    }

    mock_pipe = MagicMock(return_value=fake_output_empty)
    with patch("transformers.pipeline", return_value=mock_pipe):
        transcriber = AudioTranscriber(device="cpu")
        result = transcriber.transcribe(str(dummy_wav_file))

        assert result["transcript"] == ""
        assert result["segment_count"] == 0
        assert result["segments"] == []
        assert result["timestamp_quality"] == "none"
