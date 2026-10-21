"""
Audio Service for CrimeVisionAI.
Coordinates audio evidence processing for Epic 3:
- Activity 3.1: Automated Speech Recognition (Whisper)
- Activity 3.2: Key Event Extraction and Structured Timelines (Groq LLM)
"""

import json
from pathlib import Path
from typing import Any

from app.services.audio_transcriber import AudioTranscriber
from app.services.timeline_service import TimelineService


class AudioService:
    """Orchestrate audio transcription and timeline generation."""

    def __init__(
        self,
        transcriber: AudioTranscriber | None = None,
        timeline_service: TimelineService | None = None,
    ):
        self.transcriber = transcriber or AudioTranscriber()
        self.timeline_service = timeline_service or TimelineService()

    def process_audio(
        self,
        audio_path: str | Path,
        language: str | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Execute end-to-end audio analysis workflow:
        1. Validate and transcribe audio evidence into timestamped segments.
        2. Extract forensic anchor events and generate a chronologically sorted timeline.
        3. Optionally save the consolidated JSON output.

        Parameters
        ----------
        audio_path : str | Path
            Path to the target audio evidence file.
        language : str | None
            Optional language hint for transcription (e.g., 'te', 'hi', 'fr', 'en').
        output_path : str | Path | None
            Optional path to save the final JSON result.

        Returns
        -------
        dict[str, Any]
            Consolidated audio analysis containing transcription and structured timeline.
        """
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # 1. Activity 3.1: Speech Transcription
        transcription_result = self.transcriber.transcribe(
            audio_path=path,
            language=language,
        )

        # 2. Activity 3.2: Timeline & Event Extraction
        timeline_result = self.timeline_service.generate_timeline(
            transcription_data=transcription_result
        )

        # 3. Assemble consolidated workflow output
        analysis_result: dict[str, Any] = {
            "audio_file": path.name,
            "audio_path": str(path),
            "recording_duration_seconds": transcription_result.get("duration_seconds"),
            "transcription": transcription_result,
            "timeline": timeline_result,
            "status": "success",
        }

        # 4. Save output if requested
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(
                json.dumps(analysis_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return analysis_result
