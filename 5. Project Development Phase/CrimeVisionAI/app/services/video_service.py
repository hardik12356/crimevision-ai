"""
Video Service for CrimeVisionAI.
Implements Epic 4 (Activities 4.1 and 4.2):
- Frame extraction at regular intervals (1 FPS) using OpenCV.
- Sequential object detection (RT-DETR) and image captioning (BLIP) per frame.
- Temporal action synthesis, visual event extraction, and incident hypothesis classification via Groq.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import cv2
import numpy as np

from app.services.groq_service import GroqService
from app.services.image_captioner import ImageCaptioner
from app.services.object_detector import ObjectDetector
from app.services.scene_classifier import SceneClassifier


class VideoService:
    """
    Orchestrates video frame extraction, sequential visual analysis,
    and temporal event inference.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    DEFAULT_MAX_DURATION = 180.0  # 3 minutes maximum for laptop safety
    DEFAULT_MAX_FRAMES = 720      # maximum frames sampled (supports up to 180s at 4 FPS)
    DEFAULT_TARGET_FPS = 4.0      # 4 frames per second (~0.25s interval)
    MAX_FRAME_DIMENSION = 640     # resize max dimension for CPU efficiency

    def __init__(
        self,
        object_detector: ObjectDetector | None = None,
        image_captioner: ImageCaptioner | None = None,
        scene_classifier: SceneClassifier | None = None,
        groq_service: GroqService | None = None,
        max_duration_seconds: float = DEFAULT_MAX_DURATION,
        max_frames: int = DEFAULT_MAX_FRAMES,
        target_fps: float = DEFAULT_TARGET_FPS,
    ):
        self.object_detector = object_detector
        self.image_captioner = image_captioner
        self.scene_classifier = scene_classifier
        self.groq_service = groq_service
        self.max_duration_seconds = max_duration_seconds
        self.max_frames = max_frames
        self.target_fps = target_fps

    def _get_object_detector(self) -> ObjectDetector:
        if self.object_detector is None:
            self.object_detector = ObjectDetector()
        return self.object_detector

    def _get_image_captioner(self) -> ImageCaptioner:
        if self.image_captioner is None:
            self.image_captioner = ImageCaptioner()
        return self.image_captioner

    def _get_scene_classifier(self) -> SceneClassifier:
        if self.scene_classifier is None:
            self.scene_classifier = SceneClassifier()
        return self.scene_classifier

    def _get_groq_service(self) -> GroqService:
        if self.groq_service is None:
            self.groq_service = GroqService()
        return self.groq_service

    def _validate_video_file(self, video_path: str | Path) -> Path:
        """Validate existence, format, and size of video file."""
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported video format '{ext}'. Supported formats: {supported}"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"Video file is empty (0 bytes): {path.name}")

        return path

    def _resize_frame_preserving_aspect_ratio(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame so maximum dimension does not exceed MAX_FRAME_DIMENSION."""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > self.MAX_FRAME_DIMENSION:
            scale = self.MAX_FRAME_DIMENSION / float(max_dim)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def extract_sampled_frames(
        self, video_path: Path
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Extract video frames at target sampling rate, preserving timestamps.
        Releases all OpenCV resources in a finally block.

        Returns
        -------
        tuple[list[dict[str, Any]], dict[str, Any]]
            Extracted frame records (with image data) and metadata dictionary.
        """
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise ValueError(
                    f"Unable to open or decode video file '{video_path.name}'. File may be corrupt or missing a compatible codec."
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or np.isnan(fps):
                fps = 25.0  # safe standard fallback

            duration_seconds = round(total_frames / fps, 2) if total_frames > 0 else 0.0
            frame_step = max(1, int(round(fps / self.target_fps)))

            extracted_frames = []
            frame_idx = 0
            is_capped = False
            cap_reason = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    timestamp_sec = round(frame_idx / fps, 2)

                    if timestamp_sec > self.max_duration_seconds:
                        is_capped = True
                        cap_reason = f"Exceeded maximum processing duration of {self.max_duration_seconds}s"
                        break

                    if len(extracted_frames) >= self.max_frames:
                        is_capped = True
                        cap_reason = f"Exceeded maximum frame extraction limit of {self.max_frames} frames"
                        break

                    resized = self._resize_frame_preserving_aspect_ratio(frame)
                    extracted_frames.append({
                        "frame_index": frame_idx,
                        "timestamp_seconds": timestamp_sec,
                        "frame_data": resized,
                    })

                frame_idx += 1

            if not extracted_frames:
                raise ValueError(
                    f"No decodable video frames could be extracted from '{video_path.name}'."
                )

            last_ts = extracted_frames[-1]["timestamp_seconds"]
            status = "partial" if is_capped else "complete"

            metadata = {
                "total_video_frames": total_frames,
                "fps": round(fps, 2),
                "duration_seconds": duration_seconds or last_ts,
                "sampled_fps": self.target_fps,
                "extracted_count": len(extracted_frames),
                "analyzed_interval": {
                    "start_seconds": extracted_frames[0]["timestamp_seconds"],
                    "end_seconds": last_ts,
                },
                "status": status,
                "partial_reason": cap_reason,
            }

            return extracted_frames, metadata
        finally:
            cap.release()

    def _parse_groq_json(self, raw_content: str) -> dict[str, Any]:
        """Parse structured model JSON, handling markdown fences and truncated content."""
        text = raw_content.strip()
        if not text:
            return {}

        if "```" in text:
            pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                text = matches[0].strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        return {}

    def _generate_heuristic_events(
        self, observations: list[dict[str, Any]], max_time: float
    ) -> list[dict[str, Any]]:
        """
        Derive verifiable visual transition events directly from frame observations
        to guarantee grounded findings even if LLM synthesis is truncated or offline.
        """
        events = []
        evt_counter = 1

        prev_persons = 0
        prev_labels = set()

        for obs in observations:
            curr_persons = obs.get("person_count", 0)
            curr_labels = set(d["label"] for d in obs.get("detections", []))
            ts = obs.get("timestamp_seconds", 0.0)
            idx = obs.get("frame_index", 0)

            # 1. Person presence / count change observation
            if curr_persons > prev_persons:
                events.append({
                    "event_id": f"evt_v{evt_counter:03d}",
                    "timestamp_seconds": ts,
                    "end_timestamp_seconds": min(ts + 2.0, max_time),
                    "start_raw": ts,
                    "end_raw": min(ts + 2.0, max_time),
                    "event_type": "person_observation",
                    "description": f"Model detected person presence in sampled frame (detected count: {curr_persons}; physical entry unconfirmed without tracking)",
                    "observed_evidence": f"Frame {idx} model detection: {curr_persons} person(s)",
                    "source_frame_indices": [idx],
                    "source_timestamps": [ts],
                    "confidence": 0.85,
                    "reasoning_limitations": [
                        "Automated model prediction from sampled frame; does not confirm physical entry or persistent identity across frames"
                    ],
                })
                evt_counter += 1

            # 2. Person count decreased
            elif curr_persons < prev_persons and curr_persons == 0:
                events.append({
                    "event_id": f"evt_v{evt_counter:03d}",
                    "timestamp_seconds": ts,
                    "end_timestamp_seconds": min(ts + 2.0, max_time),
                    "start_raw": ts,
                    "end_raw": min(ts + 2.0, max_time),
                    "event_type": "person_observation",
                    "description": f"No persons detected in sampled frame (previously {prev_persons}; physical exit unconfirmed without tracking)",
                    "observed_evidence": f"Frame {idx} model detection: 0 person(s)",
                    "source_frame_indices": [idx],
                    "source_timestamps": [ts],
                    "confidence": 0.80,
                    "reasoning_limitations": [
                        "Absence of model detection in sampled frame does not confirm physical exit from scene (subject may be occluded, stationary, or out of camera bounds)"
                    ],
                })
                evt_counter += 1

            # 3. New object labels appeared in frame
            new_objs = curr_labels - prev_labels - {"person"}
            if new_objs:
                events.append({
                    "event_id": f"evt_v{evt_counter:03d}",
                    "timestamp_seconds": ts,
                    "end_timestamp_seconds": min(ts + 2.0, max_time),
                    "start_raw": ts,
                    "end_raw": min(ts + 2.0, max_time),
                    "event_type": "object_detected_in_frame",
                    "description": f"Model detection predicted object label(s): {', '.join(sorted(new_objs))}",
                    "observed_evidence": f"Frame {idx} model detections: {', '.join(sorted(new_objs))}",
                    "source_frame_indices": [idx],
                    "source_timestamps": [ts],
                    "confidence": 0.75,
                    "reasoning_limitations": [
                        "Automated model prediction; presence is not verified by human investigator"
                    ],
                })
                evt_counter += 1

            prev_persons = curr_persons
            prev_labels = curr_labels

        return events

    def analyze_video(
        self,
        video_path: str | Path,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """
        Execute end-to-end video analysis workflow:
        1. Extract frames at 1 FPS with OpenCV, releasing resources cleanly.
        2. Execute sequential frame detection and captioning (Activity 4.1).
        3. Synthesize chronological action events and crime hypothesis via Groq (Activity 4.2).
        4. Assemble consolidated findings with strict forensic disclaimers.

        Parameters
        ----------
        video_path : str | Path
            Path to input video evidence file.
        confidence_threshold : float
            Detection confidence threshold for RT-DETR (0.0 to 1.0).

        Returns
        -------
        dict[str, Any]
            Structured multimodal video analysis findings.
        """
        path = self._validate_video_file(video_path)
        extracted_frames, metadata = self.extract_sampled_frames(path)

        detector = self._get_object_detector()
        captioner = self._get_image_captioner()
        classifier = self._get_scene_classifier()

        frame_observations: list[dict[str, Any]] = []
        compact_summary: list[dict[str, Any]] = []

        # Activity 4.1: Sequential Frame Analysis
        for item in extracted_frames:
            f_idx = item["frame_index"]
            t_sec = item["timestamp_seconds"]
            frame_img = item["frame_data"]

            # Save temporary image for detector and captioner
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                cv2.imwrite(str(tmp_path), frame_img)

                # 1. Object Detection
                original_conf = getattr(detector, "confidence_threshold", 0.25)
                try:
                    detector.confidence_threshold = confidence_threshold
                    det_result = detector.detect(str(tmp_path))
                finally:
                    detector.confidence_threshold = original_conf

                # 2. Image Captioning
                caption_res = captioner.caption(str(tmp_path))
                caption_text = caption_res.get("caption", "").strip()

                # 3. Scene Classification (sample initial and key changes)
                scene_text = "surveillance view"
                if f_idx == 0 or (det_result.get("person_count", 0) > 0 and len(frame_observations) % 3 == 0):
                    try:
                        sc_res = classifier.classify(str(tmp_path))
                        scene_text = sc_res.get("top_scene", "surveillance view")
                    except Exception:
                        scene_text = "surveillance view"

                obs_record = {
                    "frame_index": f_idx,
                    "timestamp_seconds": t_sec,
                    "detection_count": det_result.get("detection_count", 0),
                    "person_count": det_result.get("person_count", 0),
                    "detections": det_result.get("detections", []),
                    "caption": caption_text,
                    "scene": scene_text,
                }
                frame_observations.append(obs_record)

                compact_summary.append({
                    "time_s": t_sec,
                    "frame": f_idx,
                    "persons": det_result.get("person_count", 0),
                    "objects": [d["label"] for d in det_result.get("detections", []) if d.get("label") != "person"],
                    "caption": caption_text,
                })
            finally:
                tmp_path.unlink(missing_ok=True)

        # Activity 4.2: Action Synthesis and Event Classification
        max_duration = metadata["analyzed_interval"]["end_seconds"]
        parsed_events: list[dict[str, Any]] = []
        incident_hypothesis = {
            "type": "insufficient_evidence",
            "reasoning": "No verified suspicious activity observed in sampled video frames.",
            "note": "AI-generated hypothesis. Requires independent investigator verification.",
        }

        # Select representative frames for Groq action synthesis to stay strictly within token limits:
        # Preserve frames with person or object detections, plus key transitions, capped at 45 frames.
        if len(compact_summary) <= 45:
            groq_frames = compact_summary
        else:
            # Always prioritize frames with detections
            frames_with_detections = [f for f in compact_summary if f.get("persons", 0) > 0 or len(f.get("objects", [])) > 0]
            if len(frames_with_detections) >= 30:
                step = max(1, len(frames_with_detections) // 40)
                groq_frames = frames_with_detections[::step][:45]
            else:
                step = max(1, len(compact_summary) // 40)
                groq_frames = compact_summary[::step][:45]
            # Ensure first and last frames are included
            if compact_summary[0] not in groq_frames:
                groq_frames.insert(0, compact_summary[0])
            if compact_summary[-1] not in groq_frames:
                groq_frames.append(compact_summary[-1])

        try:
            groq = self._get_groq_service()
            groq_response = groq.analyze_video_events({
                "video_file": path.name,
                "duration_seconds": metadata["duration_seconds"],
                "analyzed_duration_seconds": max_duration,
                "frames": groq_frames,
            })
            raw_content = groq_response.get("raw_content", "")
            groq_json = self._parse_groq_json(raw_content)

            if isinstance(groq_json, dict):
                # Extract hypothesis
                hyp_type = str(groq_json.get("incident_type_hypothesis", "unknown")).strip()
                hyp_reason = str(groq_json.get("incident_hypothesis_reasoning", "")).strip()
                incident_hypothesis["type"] = hyp_type or "unknown"
                if hyp_reason:
                    incident_hypothesis["reasoning"] = hyp_reason

                # Extract events
                model_events = groq_json.get("events", [])
                if isinstance(model_events, list):
                    for me in model_events:
                        if not isinstance(me, dict):
                            continue
                        t_start = me.get("timestamp_seconds")
                        t_end = me.get("end_timestamp_seconds", t_start)
                        if isinstance(t_start, (int, float)) and t_start >= 0.0:
                            clamped_start = round(min(float(t_start), max_duration), 2)
                            clamped_end = round(min(float(t_end or clamped_start), max_duration), 2)
                            if clamped_end < clamped_start:
                                clamped_end = clamped_start

                            parsed_events.append({
                                "event_id": str(me.get("event_id", f"evt_v{len(parsed_events)+1:03d}")),
                                "timestamp_seconds": clamped_start,
                                "end_timestamp_seconds": clamped_end,
                                "start_raw": clamped_start,
                                "end_raw": clamped_end,
                                "event_type": str(me.get("event_type", "visual_observation")),
                                "description": str(me.get("description", "Observed activity")),
                                "observed_evidence": str(me.get("observed_evidence", "")),
                                "source_frame_indices": me.get("source_frame_indices", []),
                                "source_timestamps": me.get("source_timestamps", [clamped_start]),
                                "confidence": me.get("confidence", 0.75),
                                "reasoning_limitations": me.get(
                                    "reasoning_limitations",
                                    ["Observation derived from sampled frames; identity untracked"],
                                ),
                            })
        except Exception:
            # Graceful fallback: maintain frame observations and heuristic events
            pass

        # If LLM yielded no parsed events, use grounded heuristic transition events
        if not parsed_events:
            parsed_events = self._generate_heuristic_events(frame_observations, max_duration)

        # Sort events chronologically
        parsed_events.sort(key=lambda ev: ev.get("timestamp_seconds", 0.0))

        # Standardize event IDs
        for idx, ev in enumerate(parsed_events, start=1):
            ev["event_id"] = f"evt_v{idx:03d}"

        unique_types = sorted(list(set(ev["event_type"] for ev in parsed_events)))

        return {
            "video_file": path.name,
            "video_path": str(path),
            "media_type": "video",
            "duration_seconds": metadata["duration_seconds"],
            "analyzed_interval": metadata["analyzed_interval"],
            "total_video_frames": metadata["total_video_frames"],
            "fps": metadata["fps"],
            "sampled_fps": metadata["sampled_fps"],
            "analyzed_frame_count": len(frame_observations),
            "status": metadata["status"],
            "partial_reason": metadata.get("partial_reason"),
            "frame_observations": frame_observations,
            "incident_hypothesis": incident_hypothesis,
            "timeline": {
                "summary": {
                    "total_events": len(parsed_events),
                    "event_types_found": unique_types,
                    "analyzed_duration_seconds": max_duration,
                    "status": metadata["status"],
                },
                "events": parsed_events,
            },
            "limitations": {
                "tracking_caveat": (
                    "RT-DETR detects objects per frame. It does not perform continuous biometric "
                    "tracking or guarantee persistent identity over time."
                ),
                "temporal_resolution": (
                    f"Frames were sampled at {self.target_fps} frame(s) per second. Brief events occurring "
                    "between sampled intervals may not be captured."
                ),
                "forensic_caution": (
                    "Video-derived event timelines and crime hypotheses are probabilistic AI interpretations. "
                    "Independent human review of original footage is strictly required."
                ),
            },
        }
