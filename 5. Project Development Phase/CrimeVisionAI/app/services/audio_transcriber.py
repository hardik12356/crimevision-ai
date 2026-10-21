"""
Audio Transcriber Service for CrimeVisionAI.
Activity 3.1: Automated Speech Recognition (ASR) using Whisper.
"""

from pathlib import Path
from typing import Any
import torch


class AudioTranscriber:
    """
    Local multilingual audio transcription service using Hugging Face Transformers
    and openai/whisper-base.
    """

    DEFAULT_MODEL = "openai/whisper-base"
    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ):
        self.model_name = model_name
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy loading: pipeline is initialized on first use
        self._pipeline = None

    def _load_pipeline(self) -> None:
        """Lazily load the Hugging Face ASR pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    device=self.device,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Whisper model '{self.model_name}': {exc}"
                ) from exc

    def _validate_audio_file(self, audio_path: str | Path) -> Path:
        """Validate existence, format, and non-empty status of audio file."""
        path = Path(audio_path)

        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported audio format '{ext}'. Supported formats: {supported}"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {path}")

        return path

    def _get_audio_duration(self, path: Path) -> float | None:
        """
        Attempt to extract actual recording duration from audio metadata or probe safely.
        Never guesses or derives duration from transcribed speech endpoints.
        """
        try:
            import soundfile as sf

            info = sf.info(str(path))
            return round(float(info.duration), 2)
        except Exception:
            pass

        try:
            import librosa

            dur = librosa.get_duration(path=str(path))
            if dur is not None and dur > 0:
                return round(float(dur), 2)
        except Exception:
            pass

        if path.suffix.lower() == ".wav":
            try:
                import wave

                with wave.open(str(path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return round(frames / float(rate), 2)
            except Exception:
                pass

        return None

    def _detect_overlapping_segments(
        self, segments: list[dict[str, Any]]
    ) -> bool:
        """
        Determine if any valid timestamped segments overlap in time.
        Derived strictly from valid numeric start and end intervals.
        """
        valid_intervals = [
            (s["start"], s["end"])
            for s in segments
            if isinstance(s.get("start"), (int, float))
            and isinstance(s.get("end"), (int, float))
        ]

        if len(valid_intervals) < 2:
            return False

        # Sort intervals chronologically by start time, then end time
        sorted_intervals = sorted(valid_intervals, key=lambda x: (x[0], x[1]))
        max_prev_end = -1.0

        for start_t, end_t in sorted_intervals:
            if start_t < max_prev_end:
                return True
            if end_t > max_prev_end:
                max_prev_end = end_t

        return False

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        chunk_length_s: int = 30,
    ) -> dict[str, Any]:
        """
        Transcribe an audio file into timestamped text segments.

        Parameters
        ----------
        audio_path : str | Path
            Path to the target audio file (.wav, .mp3, .m4a, etc.).
        language : str | None
            Optional language specification (e.g., 'te'/'telugu', 'hi'/'hindi', 'fr'/'french').
            If None, Whisper will attempt automatic language detection.
        chunk_length_s : int
            Audio chunk window length in seconds for Whisper pipeline processing.

        Returns
        -------
        dict[str, Any]
            Structured findings containing transcript, segments, metadata, and forensic caveats.
        """
        path = self._validate_audio_file(audio_path)
        self._load_pipeline()

        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        try:
            raw_output = self._pipeline(
                str(path),
                return_timestamps=True,
                chunk_length_s=chunk_length_s,
                generate_kwargs=generate_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Transcription failed for '{path.name}': {exc}"
            ) from exc

        # Extract transcript and model-sourced language
        if isinstance(raw_output, dict):
            full_transcript = raw_output.get("text", "").strip()
            chunks = raw_output.get("chunks", [])
            detected_lang = raw_output.get("language")
        elif isinstance(raw_output, list) and raw_output:
            first = raw_output[0]
            full_transcript = first.get("text", "").strip()
            chunks = first.get("chunks", [])
            detected_lang = first.get("language")
        else:
            full_transcript = ""
            chunks = []
            detected_lang = None

        # Actual audio duration: sourced ONLY from reliable audio metadata or probe
        duration = self._get_audio_duration(path)

        # Parse segments and timestamps honestly without fabrication
        segments: list[dict[str, Any]] = []
        has_valid_timestamps = True
        any_valid_timestamps = False

        if chunks:
            for idx, chunk in enumerate(chunks):
                ts = chunk.get("timestamp")
                start = None
                end = None
                is_chunk_valid = False

                if isinstance(ts, (tuple, list)) and len(ts) >= 2:
                    s_val, e_val = ts[0], ts[1]
                    # Validate: both numeric, non-negative, and start <= end
                    if (
                        isinstance(s_val, (int, float))
                        and isinstance(e_val, (int, float))
                        and s_val >= 0
                        and e_val >= s_val
                    ):
                        start = round(float(s_val), 2)
                        end = round(float(e_val), 2)
                        if duration is not None and duration > 0:
                            if end > duration:
                                end = duration
                            if start > duration:
                                start = duration
                        is_chunk_valid = True
                        any_valid_timestamps = True

                if not is_chunk_valid:
                    has_valid_timestamps = False

                segments.append({
                    "id": idx,
                    "start": start,
                    "end": end,
                    "text": chunk.get("text", "").strip(),
                })
        elif full_transcript:
            # Fallback when model produces text but no chunk timestamps
            segments.append({
                "id": 0,
                "start": None,
                "end": None,
                "text": full_transcript,
            })
            has_valid_timestamps = False

        # Classify timestamp quality: cannot be 'complete' if any segment has malformed timestamps
        if not segments:
            timestamp_quality = "none"
        elif has_valid_timestamps and any_valid_timestamps:
            timestamp_quality = "complete"
        elif any_valid_timestamps:
            timestamp_quality = "partial"
        else:
            timestamp_quality = "unavailable"

        # Overlapping segment detection derived strictly from valid timestamps
        has_overlapping_segments = self._detect_overlapping_segments(segments)

        return {
            "audio_file": path.name,
            "audio_path": str(path),
            "model": self.model_name,
            "device": str(self.device),
            "language_hint": language,
            "detected_language": detected_lang,
            "duration_seconds": duration,
            "transcript": full_transcript,
            "segment_count": len(segments),
            "segments": segments,
            "timestamp_quality": timestamp_quality,
            "has_overlapping_segments": has_overlapping_segments,
            "limitations": {
                "forensic_caution": (
                    "Automated speech recognition (ASR) is probabilistic and unverified. "
                    "Transcriptions, detected languages, and segment boundaries "
                    "must be independently reviewed against the original recording."
                ),
                "timestamp_accuracy": (
                    "Timestamps represent approximate acoustic chunk boundaries and should not "
                    "be interpreted as millisecond-exact forensic markers."
                ),
                "audio_quality_impact": (
                    "Background noise, recording distortion, regional accents, overlapping speech, "
                    "or non-standard dialects significantly increase error rates and hallucination risk."
                ),
            },
        }
