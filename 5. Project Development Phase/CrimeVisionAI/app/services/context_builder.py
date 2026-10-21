"""
Forensic Context Builder & Deterministic Fallback Generator.

Builds token-efficient, question-targeted evidence contexts for Groq LLM queries.
Prevents HTTP 413 / rate-limit failures by extracting only question-relevant
evidence, eliminating duplicate dumps, pruning raw frame payloads, and
enforcing a strict token budget.

Also generates evidence-grounded deterministic answers when LLM calls are
prevented by rate limits or service unavailability.
"""

import json
import logging
import re
from typing import Any

from app.services.token_budget import TokenBudgetManager, default_token_budget

logger = logging.getLogger(__name__)


def extract_query_target(question: str) -> dict[str, Any]:
    """Extract requested target timestamp (seconds) or frame index from query text."""
    q = question.lower()

    # 1. MM:SS or M:SS format (e.g. 01:23, 0:02)
    time_colon = re.search(r"\b(\d{1,2}):(\d{2}(?:\.\d+)?)\b", q)
    if time_colon:
        mins = float(time_colon.group(1))
        secs = float(time_colon.group(2))
        return {"target_timestamp": mins * 60.0 + secs, "type": "timestamp"}

    # 2. Explicit second markers: "at 1.5s", "second 2.0", "timestamp 3", "2.5 seconds", "at 5"
    sec_match = re.search(
        r"\b(?:at\s+|timestamp\s+|second\s+|time\s+|sec\s+)(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\b",
        q,
    )
    if sec_match:
        return {"target_timestamp": float(sec_match.group(1)), "type": "timestamp"}

    sec_suffix = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", q)
    if sec_suffix:
        return {"target_timestamp": float(sec_suffix.group(1)), "type": "timestamp"}

    # 3. Explicit frame index: "frame 0", "frame 25"
    frame_match = re.search(r"\bframe\s+(\d+)\b", q)
    if frame_match:
        return {"target_frame_index": int(frame_match.group(1)), "type": "frame"}

    return {}


class ContextBuilder:
    """
    Constructs compact, question-specific context for forensic queries
    and provides deterministic factual fallbacks.
    """

    _PERSON_PATTERNS = re.compile(
        r"\b(who|person|people|individual|suspect|bystander|human|man|woman|observed|seen|spotted|detected|appearance|identity|identities)\b",
        re.IGNORECASE,
    )

    _TIMELINE_PATTERNS = re.compile(
        r"\b(timeline|summarize|summary|chronology|sequence|key events|anchors|what happened|overview|events)\b",
        re.IGNORECASE,
    )

    _CONFLICT_PATTERNS = re.compile(
        r"\b(conflict\w*|caveat\w*|cross-modal|discrepan\w*|corroborat\w*|inconsisten\w*|contradict\w*|agreement|disagree\w*|alignment|synchroniz\w*)\b",
        re.IGNORECASE,
    )

    _HYPOTHESIS_PATTERNS = re.compile(
        r"\b("
        r"hypothes[ei]s|hypothesize|theory|theories|theorize"
        r"|what could have happened|what might have happened"
        r"|possible explanations?|possible scenarios?"
        r"|generate .{0,20}hypothes[ei]s"
        r"|crime .{0,15}hypothes[ei]s"
        r"|suggest .{0,15}explanations?"
        r"|what .{0,15}explanations?"
        r"|reconstruct .{0,15}events?"
        r"|scenario analysis"
        r")\b",
        re.IGNORECASE,
    )

    @classmethod
    def detect_query_focus(cls, question: str) -> str:
        """
        Detect question focus for targeted context extraction.
        Returns: 'timestamp', 'person', 'timeline', 'conflict', 'hypothesis', or 'general'.
        """
        target = extract_query_target(question)
        if target:
            return "timestamp"

        if cls._HYPOTHESIS_PATTERNS.search(question):
            return "hypothesis"

        if cls._PERSON_PATTERNS.search(question):
            return "person"

        if cls._CONFLICT_PATTERNS.search(question):
            return "conflict"

        if cls._TIMELINE_PATTERNS.search(question):
            return "timeline"

        return "general"

    @classmethod
    def build_compact_context(
        cls,
        case_data: dict,
        question: str,
        max_input_tokens: int = 4000,
    ) -> dict[str, Any]:
        """
        Build a concise, evidence-grounded context dictionary tailored specifically
        to the investigator's question.
        """
        query_target = extract_query_target(question)
        focus = cls.detect_query_focus(question)

        evidence_items = case_data.get("evidence", [])
        retrieval_limitations: list[str] = []

        if not evidence_items:
            retrieval_limitations.append("No evidence files have been recorded or analyzed for this case.")

        primary_frame_observations: list[dict[str, Any]] = []
        video_summaries: list[dict[str, Any]] = []
        audio_transcripts: list[dict[str, Any]] = []
        image_evidence: list[dict[str, Any]] = []
        timeline_events_subset: list[dict[str, Any]] = []
        direct_timestamp_events: list[dict[str, Any]] = []

        # ------------------------------------------------------------------
        # 1. TARGET: TIMESTAMP QUERY
        # ------------------------------------------------------------------
        if focus == "timestamp":
            target_sec = query_target.get("target_timestamp")
            target_idx = query_target.get("target_frame_index")

            for ev in evidence_items:
                ev_id = ev.get("evidence_id")
                filename = ev.get("original_filename")
                media_type = ev.get("media_type")
                analysis = ev.get("analysis") or {}

                if media_type == "video":
                    frames = analysis.get("frame_observations", [])
                    duration = float(analysis.get("duration_seconds", 0.0) or 0.0)
                    fps = float(analysis.get("sampled_fps", 1.0) or 1.0)

                    video_summaries.append({
                        "evidence_id": ev_id,
                        "video_filename": filename,
                        "duration_seconds": duration,
                        "sampled_fps": fps,
                    })

                    if target_sec is not None:
                        if not frames:
                            retrieval_limitations.append(f"Video '{filename}' has no stored frame observations.")
                        else:
                            closest = min(frames, key=lambda f: abs(float(f.get("timestamp_seconds", 0.0)) - target_sec))
                            diff = abs(float(closest.get("timestamp_seconds", 0.0)) - target_sec)
                            if diff <= 1.5:
                                primary_frame_observations.append(cls._clean_frame_obs(closest, ev_id, filename, fps, target_sec))
                            else:
                                retrieval_limitations.append(
                                    f"Sampled frames for {target_sec}s unavailable in '{filename}'. "
                                    f"Available sampled range: 0.0s - {frames[-1].get('timestamp_seconds', 0.0)}s."
                                )
                    elif target_idx is not None:
                        matched = next((f for f in frames if f.get("frame_index") == target_idx), None)
                        if matched:
                            primary_frame_observations.append(cls._clean_frame_obs(matched, ev_id, filename, fps))
                        else:
                            retrieval_limitations.append(f"Frame index {target_idx} is unavailable in '{filename}'.")

            if target_sec is not None:
                for ev in case_data.get("timeline_events", []):
                    s = float(ev.get("start_raw") or 0.0)
                    e = float(ev.get("end_raw") or s)
                    if s <= target_sec <= (e + 0.5):
                        direct_timestamp_events.append(cls._clean_timeline_event(ev))

        # ------------------------------------------------------------------
        # 2. TARGET: PERSON DETECTIONS QUERY
        # ------------------------------------------------------------------
        elif focus == "person":
            for ev in evidence_items:
                ev_id = ev.get("evidence_id")
                filename = ev.get("original_filename")
                media_type = ev.get("media_type")
                analysis = ev.get("analysis") or {}

                if media_type == "video":
                    frames = analysis.get("frame_observations", [])
                    fps = float(analysis.get("sampled_fps", 1.0) or 1.0)
                    duration = float(analysis.get("duration_seconds", 0.0) or 0.0)

                    # Extract ONLY frames where persons are detected
                    person_frames = [
                        f for f in frames
                        if f.get("person_count", 0) > 0
                        or any(d.get("label") == "person" for d in f.get("detections", []))
                    ]

                    video_summaries.append({
                        "evidence_id": ev_id,
                        "video_filename": filename,
                        "duration_seconds": duration,
                        "person_detected_frames_count": len(person_frames),
                        "total_sampled_frames": len(frames),
                    })

                    # Compact person frames: select key intervals to avoid 100 duplicate frames
                    selected_frames = cls._sample_person_frames(person_frames, max_samples=18)
                    for pf in selected_frames:
                        primary_frame_observations.append(cls._clean_frame_obs(pf, ev_id, filename, fps, focus="person"))

                elif media_type == "image":
                    obj_det = analysis.get("object_detection") or {}
                    person_dets = [d for d in obj_det.get("detections", []) if d.get("label") == "person"]
                    if person_dets:
                        image_evidence.append({
                            "evidence_id": ev_id,
                            "image_filename": filename,
                            "person_detections": [
                                {"label": "person", "confidence": round(float(d.get("confidence", 0.0)), 3)}
                                for d in person_dets
                            ],
                            "caption": analysis.get("image_caption", {}).get("caption"),
                        })

                elif media_type == "audio":
                    transcription = analysis.get("transcription") or {}
                    segments = transcription.get("segments", [])
                    person_segments = [
                        s for s in segments
                        if any(w in s.get("text", "").lower() for w in ("who", "he", "she", "man", "woman", "person", "someone", "suspect"))
                    ]
                    if person_segments:
                        audio_transcripts.append({
                            "evidence_id": ev_id,
                            "audio_filename": filename,
                            "relevant_segments": [
                                {
                                    "start": s.get("start"),
                                    "end": s.get("end"),
                                    "text": s.get("text", "").strip(),
                                }
                                for s in person_segments[:5]
                            ],
                        })

            for ev in case_data.get("timeline_events", []):
                desc = (ev.get("description") or "").lower()
                etype = (ev.get("event_type") or "").lower()
                if "person" in desc or "individual" in desc or "person" in etype or "suspect" in etype:
                    timeline_events_subset.append(cls._clean_timeline_event(ev))

        # ------------------------------------------------------------------
        # 3. TARGET: TIMELINE / EVENT SUMMARY QUERY
        # ------------------------------------------------------------------
        elif focus == "timeline":
            for ev in evidence_items:
                ev_id = ev.get("evidence_id")
                filename = ev.get("original_filename")
                media_type = ev.get("media_type")
                analysis = ev.get("analysis") or {}

                if media_type == "video":
                    video_summaries.append({
                        "evidence_id": ev_id,
                        "video_filename": filename,
                        "duration_seconds": analysis.get("duration_seconds"),
                        "incident_hypothesis": analysis.get("incident_hypothesis", {}),
                    })

            for ev in case_data.get("timeline_events", [])[:25]:
                timeline_events_subset.append(cls._clean_timeline_event(ev))

        # ------------------------------------------------------------------
        # 4. TARGET: CONFLICTS / CAVEATS / CROSS-MODAL QUERY
        # ------------------------------------------------------------------
        elif focus == "conflict":
            for ev in evidence_items:
                ev_id = ev.get("evidence_id")
                filename = ev.get("original_filename")
                media_type = ev.get("media_type")
                analysis = ev.get("analysis") or {}

                video_summaries.append({
                    "evidence_id": ev_id,
                    "filename": filename,
                    "media_type": media_type,
                    "status": ev.get("status"),
                })

                # Include representative frame summaries for cross-modal comparison
                if media_type == "video":
                    frames = analysis.get("frame_observations", [])
                    fps = float(analysis.get("sampled_fps", 1.0) or 1.0)
                    duration = float(analysis.get("duration_seconds", 0.0) or 0.0)
                    video_summaries[-1]["duration_seconds"] = duration
                    video_summaries[-1]["analyzed_frame_count"] = len(frames)
                    # Sample up to 4 representative frames for conflict review
                    rep_frames = cls._sample_person_frames(frames, max_samples=4)
                    for rf in rep_frames:
                        primary_frame_observations.append(cls._clean_frame_obs(rf, ev_id, filename, fps, focus="conflict"))

                elif media_type == "audio":
                    transcription = analysis.get("transcription") or {}
                    segments = transcription.get("segments", [])
                    if segments:
                        audio_transcripts.append({
                            "evidence_id": ev_id,
                            "audio_filename": filename,
                            "relevant_segments": [
                                {
                                    "start": s.get("start"),
                                    "end": s.get("end"),
                                    "text": s.get("text", "").strip(),
                                }
                                for s in segments[:6]
                            ],
                        })

            for ev in case_data.get("timeline_events", [])[:15]:
                timeline_events_subset.append(cls._clean_timeline_event(ev))

        # ------------------------------------------------------------------
        # 5. TARGET: HYPOTHESIS OR GENERAL QUERY
        # ------------------------------------------------------------------
        else:
            for ev in evidence_items:
                ev_id = ev.get("evidence_id")
                filename = ev.get("original_filename")
                media_type = ev.get("media_type")
                analysis = ev.get("analysis") or {}

                if media_type == "video":
                    frames = analysis.get("frame_observations", [])
                    fps = float(analysis.get("sampled_fps", 1.0) or 1.0)
                    video_summaries.append({
                        "evidence_id": ev_id,
                        "video_filename": filename,
                        "duration_seconds": analysis.get("duration_seconds"),
                        "sampled_fps": fps,
                        "incident_hypothesis": analysis.get("incident_hypothesis", {}),
                    })
                    # Downsample key frames to at most 8 representative transition points
                    downsampled = cls._sample_person_frames(frames, max_samples=8)
                    for df in downsampled:
                        primary_frame_observations.append(cls._clean_frame_obs(df, ev_id, filename, fps, focus=focus))

                elif media_type == "image":
                    obj_det = analysis.get("object_detection") or {}
                    image_evidence.append({
                        "evidence_id": ev_id,
                        "image_filename": filename,
                        "detections": [
                            {"label": d.get("label"), "confidence": round(float(d.get("confidence", 0.0)), 2)}
                            for d in obj_det.get("detections", [])[:6]
                        ],
                        "caption": analysis.get("image_caption", {}).get("caption"),
                    })

                elif media_type == "audio":
                    transcription = analysis.get("transcription") or {}
                    raw_text = transcription.get("transcript", "")
                    if raw_text:
                        audio_transcripts.append({
                            "evidence_id": ev_id,
                            "audio_filename": filename,
                            "transcript": raw_text[:350] + ("..." if len(raw_text) > 350 else ""),
                        })

            for ev in case_data.get("timeline_events", [])[:18]:
                timeline_events_subset.append(cls._clean_timeline_event(ev))

        # ------------------------------------------------------------------
        # Assemble core case context
        # ------------------------------------------------------------------
        case_context: dict[str, Any] = {
            "case_id": case_data.get("case_id"),
            "title": case_data.get("title"),
            "query_focus": focus,
            "query_target": query_target,
            "retrieval_limitations": retrieval_limitations,
        }

        if primary_frame_observations:
            case_context["primary_frame_observations"] = primary_frame_observations
        if video_summaries:
            case_context["video_summaries"] = video_summaries
        if image_evidence:
            case_context["image_evidence"] = image_evidence
        if audio_transcripts:
            case_context["audio_transcripts"] = audio_transcripts
        if timeline_events_subset:
            case_context["derived_timeline_events"] = timeline_events_subset
        if direct_timestamp_events:
            case_context["direct_timestamp_timeline_events"] = direct_timestamp_events

        # ------------------------------------------------------------------
        # Attach compact fusion context if available
        # ------------------------------------------------------------------
        fusion_data = case_data.get("fusion")
        if fusion_data:
            compact_fusion: dict[str, Any] = {}
            if fusion_data.get("alignment_status"):
                compact_fusion["alignment_status"] = fusion_data["alignment_status"]

            if fusion_data.get("events"):
                compact_fusion["events"] = [
                    {
                        "event_id": fe.get("event_id"),
                        "modality": fe.get("modality"),
                        "time": fe.get("timestamp_normalized"),
                        "description": fe.get("description"),
                    }
                    for fe in fusion_data["events"][:15]
                ]

            if fusion_data.get("grounded_narrative"):
                narrative = fusion_data["grounded_narrative"]
                compact_fusion["grounded_narrative"] = narrative[:800] + ("..." if len(narrative) > 800 else "")

            if fusion_data.get("relationships"):
                compact_fusion["relationships"] = [
                    {
                        "rel_id": r.get("rel_id"),
                        "type": r.get("relationship_type"),
                        "source_a": r.get("source_a_evidence_id"),
                        "source_b": r.get("source_b_evidence_id"),
                        "rationale": r.get("rationale"),
                        "caveats": r.get("caveats"),
                    }
                    for r in fusion_data["relationships"][:12]
                ]

            if fusion_data.get("anchors"):
                compact_fusion["anchors"] = [
                    {
                        "anchor_id": a.get("anchor_id"),
                        "time": a.get("timestamp_normalized"),
                        "type": a.get("event_type"),
                        "reason": a.get("reason"),
                        "significance": a.get("significance"),
                    }
                    for a in fusion_data["anchors"][:10]
                ]

            if fusion_data.get("hash_index"):
                compact_fusion["hash_index"] = fusion_data["hash_index"]

            case_context["fusion_context"] = compact_fusion

        # ------------------------------------------------------------------
        # Token Budget Enforcement & Progressive Pruning
        # ------------------------------------------------------------------
        cls._enforce_token_budget(case_context, max_tokens=max_input_tokens)

        return case_context

    @staticmethod
    def _clean_frame_obs(obs: dict, ev_id: str, filename: str, fps: float, req_t: float | None = None, focus: str | None = None) -> dict:
        t_sec = obs.get("timestamp_seconds", 0.0)
        raw_dets = obs.get("detections", [])
        # Filter detections to focus-relevant labels to reduce token count
        if focus == "person":
            raw_dets = [d for d in raw_dets if d.get("label") in ("person", "man", "woman", "individual")]
        elif focus == "conflict":
            # For conflict analysis, prioritize persons, weapons, or top 6 detections
            p_or_w = [d for d in raw_dets if d.get("label") in ("person", "knife", "gun", "weapon", "pistol", "rifle", "dagger")]
            raw_dets = p_or_w if p_or_w else raw_dets[:5]
        detections = [
            {
                "label": d.get("label"),
                "confidence": round(float(d.get("confidence", 0.0)), 3) if d.get("confidence") is not None else None,
            }
            for d in raw_dets
        ]
        res: dict[str, Any] = {
            "evidence_id": ev_id,
            "video_filename": filename,
            "frame_index": obs.get("frame_index"),
            "timestamp_seconds": t_sec,
            "sampled_fps": fps,
            "model_detections": detections,
            "detected_person_count": obs.get("person_count", len([d for d in detections if d.get("label") == "person"])),
            "blip_caption_prediction": obs.get("caption"),
            "status": "available",
        }
        if req_t is not None:
            res["requested_timestamp_seconds"] = req_t
            res["is_exact_timestamp_match"] = abs(float(t_sec) - req_t) < 0.05
        return res

    @staticmethod
    def _clean_timeline_event(ev: dict) -> dict:
        s = ev.get("start_raw") or 0.0
        e = ev.get("end_raw") or s
        return {
            "event_id": ev.get("event_id"),
            "time": f"{s}s - {e}s",
            "type": ev.get("event_type"),
            "desc": ev.get("description"),
            "source_filename": ev.get("source_filename"),
            "source_evidence_id": ev.get("source_evidence_id"),
            "media_type": ev.get("media_type"),
            "confidence": ev.get("certainty_score") or ev.get("confidence"),
        }

    @staticmethod
    def _sample_person_frames(frames: list[dict], max_samples: int = 15) -> list[dict]:
        """Select representative frames with person detections to avoid bloated prompts."""
        if not frames:
            return []
        if len(frames) <= max_samples:
            return frames

        step = max(1, len(frames) // max_samples)
        sampled = [frames[i] for i in range(0, len(frames), step)]
        if frames[-1] not in sampled:
            sampled.append(frames[-1])
        return sampled[:max_samples]

    @classmethod
    def _enforce_token_budget(cls, context: dict, max_tokens: int) -> None:
        """Progressively prune context if it exceeds the input token limit."""
        serialized = json.dumps(context, ensure_ascii=False, default=str)
        tokens = default_token_budget.count_tokens(serialized)

        if tokens <= max_tokens:
            return

        logger.info(f"[ContextBuilder] Context ({tokens} tokens) exceeds limit ({max_tokens}). Pruning...")

        # Stage 1: Truncate grounded narrative in fusion
        fc = context.get("fusion_context")
        if fc and "grounded_narrative" in fc:
            fc["grounded_narrative"] = fc["grounded_narrative"][:300] + "... [pruned]"

        # Stage 2: Reduce frame observations to max 8
        frames = context.get("primary_frame_observations")
        if frames and len(frames) > 8:
            context["primary_frame_observations"] = frames[:8]

        # Stage 3: Reduce timeline events to max 10
        events = context.get("derived_timeline_events")
        if events and len(events) > 10:
            context["derived_timeline_events"] = events[:10]

        # Re-check tokens
        serialized = json.dumps(context, ensure_ascii=False, default=str)
        tokens = default_token_budget.count_tokens(serialized)
        if tokens <= max_tokens:
            return

        # Stage 4: Aggressive pruning of fusion events and relationships
        if fc:
            if "events" in fc:
                fc["events"] = fc["events"][:5]
            if "relationships" in fc:
                fc["relationships"] = fc["relationships"][:5]

        logger.info(f"[ContextBuilder] Context pruned down to ~{default_token_budget.count_tokens(json.dumps(context))} tokens.")

    @classmethod
    def build_deterministic_answer(
        cls,
        case_data: dict,
        question: str,
        focus: str | None = None,
        advisory_note: str = "",
    ) -> dict[str, Any]:
        """
        Generate a factual, grounded markdown answer directly from stored database records
        when cloud LLM calls are throttled or unavailable.
        """
        if not focus:
            focus = cls.detect_query_focus(question)

        lines: list[str] = []
        if advisory_note:
            lines.append(f"> ⚠️ **Investigator Notice: Rate Budget Protection Active**\n> *{advisory_note}*\n")

        evidence_list = case_data.get("evidence", [])
        timeline_events = case_data.get("timeline_events", [])
        fusion = case_data.get("fusion") or {}

        if focus == "person":
            lines.append("### Chronological Person Detections (Recorded Evidence)\n")
            lines.append(
                "Automated object detection (RT-DETR) identified the following person observations across recorded media:\n"
            )

            person_records: list[dict] = []
            for ev in evidence_list:
                mtype = ev.get("media_type")
                fname = ev.get("original_filename")
                ev_id = ev.get("evidence_id")
                analysis = ev.get("analysis") or {}

                if mtype == "video":
                    for obs in analysis.get("frame_observations", []):
                        p_dets = [d for d in obs.get("detections", []) if d.get("label") == "person"]
                        if p_dets or obs.get("person_count", 0) > 0:
                            person_records.append({
                                "source": f"Video: {fname} (Evidence ID: {ev_id})",
                                "time": f"{obs.get('timestamp_seconds', 0.0):.1f}s",
                                "frame": obs.get("frame_index"),
                                "count": obs.get("person_count", len(p_dets)),
                                "conf": max([d.get("confidence", 0.0) for d in p_dets] or [0.0]),
                                "caption": obs.get("caption", ""),
                            })
                elif mtype == "image":
                    obj_det = analysis.get("object_detection") or {}
                    p_dets = [d for d in obj_det.get("detections", []) if d.get("label") == "person"]
                    if p_dets:
                        person_records.append({
                            "source": f"Image: {fname} (Evidence ID: {ev_id})",
                            "time": "Static Image (N/A)",
                            "frame": "N/A",
                            "count": len(p_dets),
                            "conf": max([d.get("confidence", 0.0) for d in p_dets] or [0.0]),
                            "caption": analysis.get("image_caption", {}).get("caption", ""),
                        })

            if person_records:
                lines.append("| Timestamp | Source Evidence | Frame | Count | Max Conf | Model Scene Caption |")
                lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
                for pr in person_records[:20]:
                    conf_str = f"{pr['conf']:.2f}" if pr['conf'] else "—"
                    cap = pr['caption'].replace("|", "-") if pr['caption'] else "—"
                    lines.append(f"| {pr['time']} | {pr['source']} | {pr['frame']} | {pr['count']} | {conf_str} | {cap} |")
            else:
                lines.append("No person detections were recorded across the analyzed evidence files.\n")

            lines.append("\n**Forensic Observation Limitation:**")
            lines.append(
                "- Detections represent automated frame predictions, not verified human identities.\n"
                "- Continuous biometric or temporal tracking is not established; physical entry or exit cannot be confirmed from object counts alone."
            )

        elif focus == "timeline":
            lines.append("### Key Timeline Events & Milestones\n")
            if timeline_events:
                lines.append("| Time Window | Modality | Event Type | Description | Source |")
                lines.append("| :--- | :--- | :--- | :--- | :--- |")
                for ev in timeline_events[:25]:
                    s = ev.get("start_raw", 0.0)
                    e = ev.get("end_raw", s)
                    t_str = f"{s:.1f}s - {e:.1f}s" if isinstance(s, (int, float)) else f"{s} - {e}"
                    etype = ev.get("event_type", "observation")
                    desc = (ev.get("description") or "—").replace("|", "-")
                    src = ev.get("source_filename") or ev.get("source_evidence_id") or "—"
                    mtype = ev.get("media_type") or "unknown"
                    lines.append(f"| {t_str} | {mtype} | {etype} | {desc} | {src} |")
            else:
                lines.append("No structured timeline events have been generated for this case yet.\n")

            anchors = fusion.get("anchors", [])
            if anchors:
                lines.append("\n#### Forensic Anchors:")
                for a in anchors[:8]:
                    lines.append(
                        f"- **Anchor `{a.get('anchor_id')}`** ({a.get('timestamp_normalized', 'N/A')}): "
                        f"{a.get('event_type')} — *{a.get('significance', a.get('reason', ''))}*"
                    )

        elif focus == "conflict":
            lines.append("### Cross-Modal Consistency & Caveats Analysis\n")
            align_status = fusion.get("alignment_status", "unknown")
            if align_status == "independent":
                lines.append(
                    "> ⚠️ **Temporal Caveat:** Analyzed evidence sources have **independent time bases**. "
                    "Timestamp co-occurrence between distinct files does not imply simultaneous physical occurrence.\n"
                )
            elif align_status == "aligned":
                lines.append("✓ **Temporal Alignment:** Evidence streams share a synchronized timeline.\n")

            relationships = fusion.get("relationships", [])
            if relationships:
                lines.append("#### Cross-Modal Relationships:")
                for rel in relationships[:12]:
                    rtype = rel.get("relationship_type", "unspecified").replace("_", " ").title()
                    icon = "⚠️" if "Conflict" in rtype else ("🔗" if "Corroboration" in rtype else "⏱")
                    lines.append(
                        f"- {icon} **{rtype}** (Between `{rel.get('source_a_evidence_id')}` & `{rel.get('source_b_evidence_id')}`):\n"
                        f"  - **Rationale:** {rel.get('rationale', 'N/A')}\n"
                        f"  - **Caveats:** {rel.get('caveats', 'None recorded.')}"
                    )
            else:
                lines.append(
                    "No cross-modal conflicts or corroborations have been generated yet. "
                    "Ensure Multimodal Fusion has been executed on case evidence."
                )

        elif focus == "timestamp":
            target = extract_query_target(question)
            t_val = target.get("target_timestamp")
            lines.append(f"### Evidence Findings for Timestamp {t_val}s\n")
            matches = 0
            for ev in evidence_list:
                mtype = ev.get("media_type")
                fname = ev.get("original_filename")
                analysis = ev.get("analysis") or {}

                if mtype == "video":
                    frames = analysis.get("frame_observations", [])
                    if frames and t_val is not None:
                        closest = min(frames, key=lambda f: abs(float(f.get("timestamp_seconds", 0.0)) - t_val))
                        if abs(float(closest.get("timestamp_seconds", 0.0)) - t_val) <= 1.5:
                            matches += 1
                            lines.append(f"**Sampled Frame in `{fname}` at {closest.get('timestamp_seconds')}s (Frame {closest.get('frame_index')}):**")
                            dets = closest.get("detections", [])
                            det_summary = ", ".join([f"{d.get('label')} (conf {d.get('confidence', 0.0):.2f})" for d in dets]) or "No objects detected"
                            lines.append(f"- **Detections:** {det_summary}")
                            lines.append(f"- **BLIP Caption:** {closest.get('caption', 'None')}")
                            lines.append(f"- **CLIP Scene:** {closest.get('scene', 'None')}\n")

            if matches == 0:
                lines.append(f"No sampled frame observation found within ±1.5s of timestamp {t_val}s.")

        else:
            lines.append("### Case Evidence Summary Overview\n")
            lines.append(f"- **Case ID:** `{case_data.get('case_id')}`")
            lines.append(f"- **Title:** {case_data.get('title')}")
            lines.append(f"- **Recorded Evidence Count:** {len(evidence_list)} item(s)")
            for ev in evidence_list:
                lines.append(f"  - `{ev.get('original_filename')}` ({ev.get('media_type')}, Status: {ev.get('status')})")
            lines.append(f"- **Consolidated Timeline Events:** {len(timeline_events)} event(s)")
            if fusion.get("grounded_narrative"):
                lines.append(f"\n**Fusion Narrative:**\n{fusion.get('grounded_narrative')}")

        return {
            "answer": "\n".join(lines),
            "note": (
                "Deterministic findings compiled directly from stored case evidence and model detections. "
                "Automated model predictions are unverified by human investigators."
            ),
            "is_deterministic_fallback": bool(advisory_note),
        }
