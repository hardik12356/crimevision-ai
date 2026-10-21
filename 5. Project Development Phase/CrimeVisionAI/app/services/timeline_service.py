"""
Timeline Service for CrimeVisionAI.
Activity 3.2: Extract Key Events and Generate Structured Timelines.
"""

import json
import re
from typing import Any

from app.services.groq_service import GroqService


class TimelineService:
    """
    Service for extracting anchor events from speech transcripts and
    generating structured, chronologically aligned forensic timelines.
    """

    def __init__(self, groq_service: GroqService | None = None):
        self.groq_service = groq_service or GroqService()

    def _get_forensic_limitations(self) -> dict[str, str]:
        """Return standardized forensic limitations and disclaimers."""
        return {
            "timestamp_accuracy": (
                "Whisper timestamps represent approximate acoustic chunk boundaries "
                "and must not be treated as millisecond-accurate forensic measurements."
            ),
            "speaker_attribution": (
                "Speaker diarization is not present in the baseline transcription; "
                "speaker identities and attributions are unverified claims."
            ),
            "allegation_vs_fact": (
                "Statements made by speakers represent claims/allegations and are not "
                "verified forensic facts."
            ),
            "model_confidence": (
                "Confidence scores indicate model extraction confidence over the transcript text, "
                "not factual certainty or physical truth."
            ),
        }

    def _parse_groq_json(self, raw_content: str) -> list[dict[str, Any]]:
        """
        Robustly parse model-generated JSON, stripping markdown fences if present.
        Returns a list of raw event dictionaries, or an empty list if malformed.
        """
        text = raw_content.strip()
        if not text:
            return []

        # Strip markdown code blocks like ```json ... ``` or ``` ... ```
        if "```" in text:
            pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                text = matches[0].strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                events = data.get("events", [])
                if isinstance(events, list):
                    return events
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        return []

    def _derive_timestamps(
        self,
        source_segment_ids: list[int],
        segment_map: dict[int, dict[str, Any]],
    ) -> tuple[float | None, float | None]:
        """
        Derive start_raw and end_raw from source segment timestamps.
        Preserves None when timestamps are missing or unsupported.
        Never manufactures precise sub-second timings from whole segments.
        """
        valid_starts: list[float] = []
        valid_ends: list[float] = []

        for seg_id in source_segment_ids:
            seg = segment_map.get(seg_id)
            if seg:
                start = seg.get("start")
                end = seg.get("end")
                if isinstance(start, (int, float)):
                    valid_starts.append(float(start))
                if isinstance(end, (int, float)):
                    valid_ends.append(float(end))

        start_raw = min(valid_starts) if valid_starts else None
        end_raw = max(valid_ends) if valid_ends else None

        return (
            round(start_raw, 2) if start_raw is not None else None,
            round(end_raw, 2) if end_raw is not None else None,
        )

    def _deduplicate_events(
        self,
        events: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Detect and reconcile likely duplicate events arising from repeated or overlapping segments.
        Preserves conflicting statements, distinct events, and records deduplication audit notes.
        """
        if not events:
            return [], []

        deduplicated: list[dict[str, Any]] = []
        dedup_notes: list[str] = []

        for event in events:
            is_dup = False
            for existing in deduplicated:
                # Same event type with overlapping source segments or identical quotes
                same_type = event["event_type"].lower() == existing["event_type"].lower()
                common_segments = set(event["source_segment_ids"]).intersection(
                    set(existing["source_segment_ids"])
                )
                same_quote = bool(
                    set(event["supporting_quotes"]).intersection(
                        set(existing["supporting_quotes"])
                    )
                )

                if same_type and (common_segments or same_quote):
                    # Check if descriptions are substantially identical
                    desc_a = event["description"].lower().strip()
                    desc_b = existing["description"].lower().strip()
                    if desc_a == desc_b or desc_a in desc_b or desc_b in desc_a:
                        # Reconcile duplicate into existing record
                        is_dup = True
                        merged_segs = sorted(
                            list(set(existing["source_segment_ids"] + event["source_segment_ids"]))
                        )
                        existing["source_segment_ids"] = merged_segs

                        # Merge quotes preserving uniqueness
                        for q in event["supporting_quotes"]:
                            if q not in existing["supporting_quotes"]:
                                existing["supporting_quotes"].append(q)

                        # Update timestamps if new event broadens bounds
                        if event["start_raw"] is not None:
                            if existing["start_raw"] is None or event["start_raw"] < existing["start_raw"]:
                                existing["start_raw"] = event["start_raw"]
                        if event["end_raw"] is not None:
                            if existing["end_raw"] is None or event["end_raw"] > existing["end_raw"]:
                                existing["end_raw"] = event["end_raw"]

                        dedup_notes.append(
                            f"Merged duplicate event '{event['event_type']}' referencing segments "
                            f"{event['source_segment_ids']} into existing event."
                        )
                        break

            if not is_dup:
                deduplicated.append(event)

        return deduplicated, dedup_notes

    def generate_timeline(
        self,
        transcription_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a structured chronological timeline of forensic events from transcription data.

        Parameters
        ----------
        transcription_data : dict[str, Any]
            The output payload from AudioTranscriber.transcribe().

        Returns
        -------
        dict[str, Any]
            Standardized timeline with events, segment links, notes, and disclaimers.
        """
        audio_file = transcription_data.get("audio_file", "unknown")
        duration = transcription_data.get("duration_seconds")
        transcript = transcription_data.get("transcript", "").strip()
        segments = transcription_data.get("segments", [])
        has_overlapping = transcription_data.get("has_overlapping_segments", False)
        timestamp_quality = transcription_data.get("timestamp_quality", "none")

        # 1. Handle empty transcript gracefully
        if not transcript or not segments:
            return {
                "audio_file": audio_file,
                "recording_duration_seconds": duration,
                "timeline_summary": {
                    "total_events": 0,
                    "event_types_found": [],
                    "has_overlapping_segments": has_overlapping,
                    "timestamp_quality": timestamp_quality,
                    "status": "empty_transcript",
                    "note": "Transcript is empty or has no segments; no events extracted.",
                },
                "events": [],
                "deduplication_notes": [],
                "limitations": self._get_forensic_limitations(),
            }

        # 2. Build segment lookup table
        segment_map = {seg["id"]: seg for seg in segments if isinstance(seg.get("id"), int)}

        # 3. Call Groq service for anchor-event extraction
        try:
            groq_result = self.groq_service.extract_audio_events(transcription_data)
        except Exception as exc:
            raise RuntimeError(f"Event extraction failed: {exc}") from exc

        raw_content = groq_result.get("raw_content", "")
        raw_events = self._parse_groq_json(raw_content)

        # 4. Standardize and validate each extracted event
        validated_events: list[dict[str, Any]] = []

        for item in raw_events:
            if not isinstance(item, dict):
                continue

            event_type = str(item.get("event_type", "unclassified")).strip()
            description = str(item.get("description", "")).strip()

            if not description:
                continue

            # Validate source_segment_ids
            raw_seg_ids = item.get("source_segment_ids", [])
            if isinstance(raw_seg_ids, list):
                source_seg_ids = [
                    int(sid) for sid in raw_seg_ids if isinstance(sid, (int, str)) and str(sid).isdigit()
                ]
            else:
                source_seg_ids = []

            # Derive start_raw and end_raw strictly from source segment bounds
            start_raw, end_raw = self._derive_timestamps(source_seg_ids, segment_map)

            # Quotes
            quotes = item.get("supporting_quotes", [])
            if isinstance(quotes, list):
                supporting_quotes = [str(q).strip() for q in quotes if str(q).strip()]
            else:
                supporting_quotes = []

            # If model omitted quotes, pull from source segments
            if not supporting_quotes and source_seg_ids:
                for sid in source_seg_ids:
                    seg = segment_map.get(sid)
                    if seg and seg.get("text"):
                        supporting_quotes.append(seg["text"])

            # Confidence: model extraction confidence
            raw_conf = item.get("confidence")
            if isinstance(raw_conf, (int, float)):
                confidence = round(float(raw_conf), 2)
            else:
                confidence = 0.80

            # Allegation status
            is_allegation = bool(item.get("is_allegation", False))

            # Uncertainties
            uncertainties = item.get("uncertainties", [])
            if isinstance(uncertainties, list):
                event_uncertainties = [str(u).strip() for u in uncertainties if str(u).strip()]
            else:
                event_uncertainties = []

            event_uncertainties.append(
                "Source timestamps represent approximate acoustic boundaries."
            )
            if is_allegation:
                event_uncertainties.append("Event is based on speaker allegation, not verified physical evidence.")
            if start_raw is None or end_raw is None:
                event_uncertainties.append("Temporal boundary unsupported by segment timestamps.")

            stated_location = item.get("stated_location")
            if stated_location is not None:
                stated_location = str(stated_location).strip() or None

            entities = item.get("entities_mentioned", [])
            if isinstance(entities, list):
                entities_mentioned = [str(e).strip() for e in entities if str(e).strip()]
            else:
                entities_mentioned = []

            validated_events.append({
                "event_type": event_type,
                "description": description,
                "start_raw": start_raw,
                "end_raw": end_raw,
                "source_segment_ids": source_seg_ids,
                "supporting_quotes": supporting_quotes,
                "confidence": confidence,
                "stated_location": stated_location,
                "entities_mentioned": entities_mentioned,
                "is_allegation": is_allegation,
                "uncertainties": event_uncertainties,
            })

        # 5. Deduplication and reconcile overlapping/repeated events
        deduped_events, dedup_notes = self._deduplicate_events(validated_events)

        if has_overlapping:
            dedup_notes.append(
                "Source audio transcription contains overlapping segments; "
                "consecutive events may reflect concurrent or rapid-turn speech."
            )

        # 6. Chronological alignment:
        # Sort by start_raw. Events with start_raw = None are placed last as unsequenced.
        def sort_key(ev: dict[str, Any]) -> tuple[int, float]:
            s = ev.get("start_raw")
            if s is not None and isinstance(s, (int, float)):
                return (0, float(s))
            return (1, 0.0)

        aligned_events = sorted(deduped_events, key=sort_key)

        # Assign standardized sequential event_id
        for idx, ev in enumerate(aligned_events, start=1):
            ev["event_id"] = f"evt_{idx:03d}"

        # 7. Collect unique event types
        event_types = sorted(list(set(ev["event_type"] for ev in aligned_events)))

        return {
            "audio_file": audio_file,
            "recording_duration_seconds": duration,
            "timeline_summary": {
                "total_events": len(aligned_events),
                "event_types_found": event_types,
                "has_overlapping_segments": has_overlapping,
                "timestamp_quality": timestamp_quality,
                "status": "success" if aligned_events else "no_events_found",
            },
            "events": aligned_events,
            "deduplication_notes": dedup_notes,
            "limitations": self._get_forensic_limitations(),
        }
