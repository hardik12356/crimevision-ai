"""
Fusion Service for CrimeVisionAI.
Implements Epic 5 (Activity 5.1):
Fuse image, audio, and video outputs into a unified, source-traceable forensic timeline.

Capabilities:
1. Temporal float normalization & synchronization preserving millisecond precision.
2. Common typed forensic event representation across image, audio, and video modalities.
3. Deterministic cross-modal hash indexing (SHA-256) mapping time buckets and terms to event lists.
4. Cross-modal audit: corroboration matching, explicit conflict detection, and explanation.
5. Anchor-event highlighting (weapons detected vs. mentioned, person entries, significant sounds, conflicts).
6. Transparent weighted certainty and evidence-consistency scoring with strict safeguards.
7. Unified chronological timeline and evidence-grounded factual narrative.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


# ===========================================================================
# Domain Lexicons & Recognition Rules
# ===========================================================================

WEAPON_LABELS = {
    "knife",
    "dagger",
    "gun",
    "pistol",
    "rifle",
    "firearm",
    "weapon",
    "revolver",
    "shotgun",
    "blade",
    "machete",
    "baseball bat",
}

WEAPON_TEXT_TERMS = {
    "knife",
    "gun",
    "pistol",
    "rifle",
    "firearm",
    "weapon",
    "shoot",
    "shooting",
    "shot",
    "stab",
    "stabbing",
    "stabbed",
    "armed",
    "blade",
}

UNARMED_INDICATORS = [
    "unarmed",
    "no weapon",
    "without a weapon",
    "didn't have a weapon",
    "did not have a weapon",
    "no gun",
    "no knife",
    "hands were empty",
    "empty hands",
]

SOLITUDE_INDICATORS = [
    "alone",
    "nobody",
    "no one",
    "empty room",
    "no one was there",
    "nobody was there",
    "completely empty",
    "by myself",
]

AGGRESSION_LABELS = {
    "physical_struggle",
    "aggression",
    "threat",
    "possible_physical_altercation",
    "fight",
    "assault",
}


# ===========================================================================
# 1. Temporal Normalization Utilities
# ===========================================================================

def normalize_timestamp(val: Any) -> float | None:
    """
    Convert supported timestamp representations into a consistent float in seconds,
    strictly preserving fractional subsecond precision (e.g. 5.250s).
    Never converts to an integer or truncates precision.
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        if math.isnan(val) or val < 0:
            return None
        return round(float(val), 3)

    if isinstance(val, str):
        val = val.strip().lower()
        if not val or val in ("null", "none", "unknown", "--"):
            return None

        # Format: MM:SS or M:SS.sss (e.g. "01:23.45", "0:05.25")
        colon_match = re.match(r"^(\d{1,3}):(\d{2}(?:\.\d+)?)$", val)
        if colon_match:
            mins = float(colon_match.group(1))
            secs = float(colon_match.group(2))
            return round(mins * 60.0 + secs, 3)

        # Format: "5.25s", "5.25 seconds", "at 5.25"
        sec_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?$", val)
        if sec_match:
            try:
                sec_val = float(sec_match.group(1))
                return round(sec_val, 3)
            except ValueError:
                return None

    return None


def to_ms(seconds: float | None) -> int | None:
    """Convert fractional seconds to integer milliseconds, preserving precision."""
    if seconds is None:
        return None
    return int(round(seconds * 1000.0))


# ===========================================================================
# 2. Typed Forensic Event & Relationship Models
# ===========================================================================

@dataclass
class ForensicEvent:
    """Standardized representation of an observation from any evidence modality."""
    event_id: str
    case_id: str
    evidence_id: str
    original_filename: str
    modality: str  # 'image' | 'audio' | 'video'
    source_event_id: str
    event_type: str
    observed_labels: list[str] = field(default_factory=list)
    extracted_entities: list[str] = field(default_factory=list)
    description: str = ""
    source_confidence: float | None = None

    # Temporal coordinates
    original_timestamp: Any = None
    original_time_basis: str = "unknown"  # 'media_relative' | 'absolute_utc' | 'unknown'
    timestamp_normalized: float | None = None
    timestamp_ms: int | None = None
    end_timestamp_normalized: float | None = None
    end_timestamp_ms: int | None = None
    temporal_uncertainty_seconds: float | None = None
    alignment_status: str = "temporally_unknown"  # 'aligned_common_timeline' | 'unaligned_independent_time_basis' | 'temporally_uncertain' | 'temporally_unknown'

    # Cross-modal audit & scoring
    corroboration_links: list[str] = field(default_factory=list)
    conflict_links: list[str] = field(default_factory=list)
    relationship_status: str = "single_source"  # 'corroborated' | 'potentially_consistent' | 'conflicting' | 'single_source' | 'insufficient_evidence'
    evidence_consistency_score: float = 0.50
    certainty_level: str = "low"  # 'high' | 'medium' | 'low'
    scoring_explanation: str = ""

    # Anchor highlighting
    is_anchor: bool = False
    anchor_reason: str | None = None

    # Provenance metadata
    source_reference: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossModalRelationship:
    """Represents a validated corroboration, consistency, or conflict relationship between events."""
    relationship_id: str
    source_event_ids: list[str]
    relationship_type: str  # 'conflict' | 'qualified_corroboration' | 'thematic_match' | 'potentially_related' | 'corroboration' | 'potential_consistency' | 'insufficient_information'
    matching_rule: str
    time_alignment_basis: str  # 'synchronized_timeline' | 'overlapping_temporal_window' | 'semantic_unaligned'
    explanation: str
    uncertainties: list[str] = field(default_factory=list)
    source_a_evidence_id: str | None = None
    source_b_evidence_id: str | None = None
    source_a_filename: str | None = None
    source_b_filename: str | None = None
    source_a_modality: str | None = None
    source_b_modality: str | None = None
    source_a_timestamp: float | None = None
    source_b_timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ===========================================================================
# 3. Deterministic Cross-Modal Hash Index (SHA-256)
# ===========================================================================

class CrossModalHashIndex:
    """
    Deterministic lookup structure for event matching and timeline retrieval.
    Maps SHA-256 hash keys to lists of event references, strictly preserving
    all observations within the same second or term bucket without collision loss.
    """

    def __init__(self):
        # time_buckets: sha256 -> list of event reference dicts
        self.time_buckets: dict[str, list[dict[str, Any]]] = {}
        # term_buckets: sha256 -> list of event reference dicts
        self.term_buckets: dict[str, list[dict[str, Any]]] = {}
        # key_map: sha256 -> human-readable key string
        self.key_map: dict[str, str] = {}

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """Compute stable SHA-256 hex digest for key."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def index_event(self, event: ForensicEvent) -> None:
        """Index a single forensic event by time bucket and normalized terms."""
        ref = {
            "event_id": event.event_id,
            "evidence_id": event.evidence_id,
            "modality": event.modality,
            "original_filename": event.original_filename,
            "timestamp_normalized": event.timestamp_normalized,
            "event_type": event.event_type,
            "labels": event.observed_labels,
        }

        # 1. Index by time bucket if timestamp is known
        if event.timestamp_normalized is not None:
            bucket_sec = int(math.floor(event.timestamp_normalized))
            raw_time_key = f"time:{event.original_time_basis}:{bucket_sec}"
            h_time = self._hash_key(raw_time_key)
            self.key_map[h_time] = raw_time_key
            if h_time not in self.time_buckets:
                self.time_buckets[h_time] = []
            self.time_buckets[h_time].append(ref)

        # 2. Index by normalized labels and entities
        terms_to_index = set()
        for label in event.observed_labels:
            clean = label.lower().strip()
            if len(clean) >= 2:
                terms_to_index.add(clean)

        for ent in event.extracted_entities:
            clean = ent.lower().strip()
            if len(clean) >= 2:
                terms_to_index.add(clean)

        # Also index specific significant keywords from event_type
        terms_to_index.add(event.event_type.lower().strip())

        for term in terms_to_index:
            raw_term_key = f"term:{term}"
            h_term = self._hash_key(raw_term_key)
            self.key_map[h_term] = raw_term_key
            if h_term not in self.term_buckets:
                self.term_buckets[h_term] = []
            self.term_buckets[h_term].append(ref)

    def lookup_by_time(
        self,
        time_seconds: float,
        time_basis: str = "media_relative:common",
        tolerance_seconds: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all event references within the temporal tolerance window around time_seconds.
        Handles adjacent second buckets so boundary events are never missed.
        """
        results: list[dict[str, Any]] = []
        seen_ids = set()

        min_sec = int(math.floor(time_seconds - tolerance_seconds))
        max_sec = int(math.floor(time_seconds + tolerance_seconds))

        for sec in range(min_sec, max_sec + 1):
            raw_key = f"time:{time_basis}:{sec}"
            h_key = self._hash_key(raw_key)
            bucket = self.time_buckets.get(h_key, [])
            for item in bucket:
                ts = item.get("timestamp_normalized")
                if ts is not None and abs(ts - time_seconds) <= tolerance_seconds:
                    if item["event_id"] not in seen_ids:
                        seen_ids.add(item["event_id"])
                        results.append(item)

        return results

    def lookup_by_term(self, term: str) -> list[dict[str, Any]]:
        """Retrieve all event references indexed under a normalized term."""
        raw_key = f"term:{term.lower().strip()}"
        h_key = self._hash_key(raw_key)
        return list(self.term_buckets.get(h_key, []))

    def to_dict(self) -> dict[str, Any]:
        """Serialize index structure metadata for persistence and audit review."""
        return {
            "total_time_buckets": len(self.time_buckets),
            "total_term_buckets": len(self.term_buckets),
            "time_bucket_keys": [self.key_map.get(k, k) for k in self.time_buckets.keys()],
            "term_bucket_keys": [self.key_map.get(k, k) for k in self.term_buckets.keys()],
        }


def check_temporal_proximity(
    ev_a: ForensicEvent,
    ev_b: ForensicEvent,
    tolerance_seconds: float = 2.0,
) -> tuple[bool, float, str]:
    """
    Check if two events overlap or co-occur within a configurable temporal tolerance,
    accounting for event intervals [start, end] and source timestamp uncertainties.
    Returns (is_proximate, distance_seconds, explanation).
    """
    if ev_a.timestamp_normalized is None or ev_b.timestamp_normalized is None:
        return False, float("inf"), "One or both events lack normalized timestamps."

    if ev_a.alignment_status != "aligned_common_timeline" or ev_b.alignment_status != "aligned_common_timeline":
        return False, float("inf"), "Events do not share a verified common timeline."

    start_a = ev_a.timestamp_normalized
    end_a = ev_a.end_timestamp_normalized if ev_a.end_timestamp_normalized is not None else start_a
    start_b = ev_b.timestamp_normalized
    end_b = ev_b.end_timestamp_normalized if ev_b.end_timestamp_normalized is not None else start_b

    # Ensure start <= end
    if end_a < start_a:
        end_a = start_a
    if end_b < start_b:
        end_b = start_b

    # Check for direct interval overlap
    if max(start_a, start_b) <= min(end_a, end_b):
        overlap_len = min(end_a, end_b) - max(start_a, start_b)
        return True, 0.0, f"Intervals directly overlap by {overlap_len:.2f}s."

    # Distance between non-overlapping intervals
    gap = max(0.0, max(start_a, start_b) - min(end_a, end_b))
    unc_a = ev_a.temporal_uncertainty_seconds or 0.5
    unc_b = ev_b.temporal_uncertainty_seconds or 0.5
    effective_tolerance = tolerance_seconds + unc_a + unc_b

    if gap <= effective_tolerance:
        return True, gap, f"Interval gap of {gap:.2f}s is within effective tolerance ({effective_tolerance:.2f}s)."

    return False, gap, f"Interval gap of {gap:.2f}s exceeds effective tolerance ({effective_tolerance:.2f}s)."


# ===========================================================================
# 4. Certainty & Consistency Scoring Heuristic
# ===========================================================================

class CertaintyScorer:
    """
    Transparent, explainable evidence-consistency and certainty scoring.
    Accounts for source model confidence, independent corroboration,
    temporal alignment quality, and explicit contradictions.

    Safeguards:
    - Never assigns calibrated probability or proof claims.
    - Single-source findings cannot receive 'high' certainty.
    - Unresolved conflicts force 'low' certainty and lower consistency score.
    - Missing confidence is handled conservatively without inventing fake numbers.
    """

    @classmethod
    def score_event(
        cls,
        event: ForensicEvent,
        has_corroboration: bool,
        has_conflict: bool,
        corroborating_modalities: set[str],
    ) -> tuple[float, str, str]:
        """
        Compute evidence consistency score (0.05 to 0.95), certainty level ('high'|'medium'|'low'),
        and human-readable explanation.
        """
        reasons = []

        # 1. Base source component
        # Requirement 1: Do not silently substitute 0.50 when confidence is unavailable
        has_source_conf = event.source_confidence is not None
        if has_source_conf:
            c_source = min(max(float(event.source_confidence), 0.0), 1.0)
            reasons.append(f"Source confidence: {c_source:.2f} (weighted at 60%)")
            source_weight = 0.60
            align_weight = 0.40
        else:
            c_source = 0.0
            source_weight = 0.0
            align_weight = 0.50
            reasons.append("Source confidence: unavailable/unrated (evaluated solely on temporal alignment quality; not defaulted)")

        # 2. Temporal alignment weight
        alignment_weights = {
            "aligned_common_timeline": 1.0,
            "unaligned_independent_time_basis": 0.75,
            "temporally_uncertain": 0.65,
            "temporally_unknown": 0.50,
        }
        w_align = alignment_weights.get(event.alignment_status, 0.50)
        reasons.append(f"Temporal alignment ({event.alignment_status}): factor {w_align:.2f}")

        # Baseline combination
        if has_source_conf:
            base_score = (c_source * source_weight) + (w_align * align_weight)
        else:
            base_score = w_align * align_weight

        # 3. Corroboration modifier
        mod_corrob = 0.0
        if has_corroboration and len(corroborating_modalities) >= 1:
            mod_corrob = 0.20
            mods_str = ", ".join(sorted(corroborating_modalities))
            reasons.append(f"Independent corroboration bonus (+0.20) from modality: [{mods_str}]")
        elif event.relationship_status == "potentially_consistent":
            mod_corrob = 0.08
            reasons.append("Potential consistency bonus (+0.08)")
        else:
            reasons.append("Single-source observation (no independent corroboration)")

        # 4. Conflict penalty
        mod_conflict = 0.0
        if has_conflict:
            mod_conflict = -0.35
            reasons.append("Explicit cross-modal contradiction penalty (-0.35)")

        raw_score = base_score + mod_corrob + mod_conflict
        final_score = round(min(max(raw_score, 0.05), 0.95), 3)

        # 5. Determine Certainty Level Category with strict safeguards
        if has_conflict:
            certainty = "low"
            reasons.append("Certainty forced to LOW due to unresolved cross-modal contradiction.")
        elif event.alignment_status in ("temporally_unknown", "temporally_uncertain"):
            certainty = "low" if final_score < 0.60 else "medium"
            reasons.append("High certainty barred due to unanchored or uncertain temporal coordinates.")
        elif not has_source_conf:
            certainty = "medium" if final_score >= 0.45 else "low"
            reasons.append("High certainty barred because source model confidence is unavailable/unrated.")
        elif not has_corroboration:
            # Single-source safeguard: single-source cannot be high certainty
            certainty = "medium" if final_score >= 0.50 else "low"
            reasons.append("High certainty barred for single-source observation without independent confirmation.")
        else:
            # Corroborated with aligned timeline, verified source confidence, and no conflicts
            if final_score >= 0.75 and event.alignment_status == "aligned_common_timeline":
                certainty = "high"
            elif final_score >= 0.45:
                certainty = "medium"
            else:
                certainty = "low"

        explanation = "; ".join(reasons)
        return final_score, certainty, explanation


# ===========================================================================
# 5. Main Fusion Orchestration Service
# ===========================================================================

class FusionService:
    """
    Core service for Activity 5.1: Cross-modal evidence fusion,
    audit, conflict resolution, certainty scoring, and timeline generation.
    """

    def __init__(self):
        pass

    # -----------------------------------------------------------------------
    # Evidence Ingestion & Event Extraction
    # -----------------------------------------------------------------------
    def extract_events_from_case_evidence(
        self,
        case_data: dict[str, Any],
        sync_groups: dict[str, str] | None = None,
    ) -> list[ForensicEvent]:
        """
        Extract and normalize observations from all image, audio, and video evidence in a case.
        Determines legitimate common time bases vs. unaligned independent evidence.
        """
        case_id = case_data.get("case_id", "CASE-UNKNOWN")
        evidence_items = case_data.get("evidence", [])
        events: list[ForensicEvent] = []
        event_counter = 1

        sync_map = sync_groups or {}

        for ev in evidence_items:
            ev_id = ev.get("evidence_id", f"EVD-{event_counter:04d}")
            filename = ev.get("original_filename", "unnamed_evidence")
            media_type = ev.get("media_type", "unknown")
            analysis = ev.get("analysis") or {}
            metadata = ev.get("metadata") or {}

            # Explicit synchronization group or offset (Requirement 2: do not align solely by filename stem)
            sync_group = sync_map.get(ev_id) or sync_map.get(filename) or ev.get("sync_group") or metadata.get("sync_group")
            time_offset = ev.get("temporal_offset_seconds") or metadata.get("temporal_offset_seconds")
            offset_val = float(time_offset) if isinstance(time_offset, (int, float)) else 0.0

            # Determine temporal alignment basis
            if media_type == "image":
                time_basis = "unknown"
                alignment_status = "temporally_unknown"
            elif sync_group:
                time_basis = f"sync_group:{sync_group}"
                alignment_status = "aligned_common_timeline"
            elif time_offset is not None:
                time_basis = f"offset_aligned:{offset_val:.2f}s"
                alignment_status = "aligned_common_timeline"
            else:
                # Preserve independent time basis: do not force onto common timeline
                time_basis = f"media_relative:{ev_id}"
                alignment_status = "unaligned_independent_time_basis"

            # ---------------------------------------------------------------
            # A. Video Evidence Parsing
            # ---------------------------------------------------------------
            if media_type == "video":
                # 1. Timeline visual events
                vid_timeline = analysis.get("timeline") or {}
                vid_events = vid_timeline.get("events") or []

                for ve in vid_events:
                    raw_start = ve.get("timestamp_seconds", ve.get("start_raw"))
                    raw_end = ve.get("end_timestamp_seconds", ve.get("end_raw", raw_start))
                    norm_start = normalize_timestamp(raw_start)
                    norm_end = normalize_timestamp(raw_end)
                    if norm_start is not None and offset_val != 0.0:
                        norm_start = round(norm_start + offset_val, 4)
                    if norm_end is not None and offset_val != 0.0:
                        norm_end = round(norm_end + offset_val, 4)

                    desc = str(ve.get("description", ""))
                    ev_type = str(ve.get("event_type", "visual_observation")).lower()

                    observed_labels = []
                    for w in WEAPON_LABELS:
                        if w in desc.lower():
                            observed_labels.append(w)
                    if "person" in desc.lower() or "person" in ev_type:
                        observed_labels.append("person")

                    # Extract person count if available
                    person_count = ve.get("person_count")
                    if person_count is None:
                        m = re.search(r"(\d+)\s+person", desc.lower() + " " + str(ve.get("observed_evidence", "")).lower())
                        if m:
                            person_count = int(m.group(1))
                        elif "person" in desc.lower() or "person" in ev_type:
                            person_count = 1
                        else:
                            person_count = 0

                    conf = ve.get("confidence")
                    src_conf = float(conf) if isinstance(conf, (int, float)) else None

                    fevt = ForensicEvent(
                        event_id=f"fevt_{event_counter:03d}",
                        case_id=case_id,
                        evidence_id=ev_id,
                        original_filename=filename,
                        modality="video",
                        source_event_id=str(ve.get("event_id", f"vevt_{event_counter}")),
                        event_type=ev_type,
                        observed_labels=observed_labels,
                        extracted_entities=[],
                        description=desc,
                        source_confidence=src_conf,
                        original_timestamp=raw_start,
                        original_time_basis=time_basis,
                        timestamp_normalized=norm_start,
                        timestamp_ms=to_ms(norm_start),
                        end_timestamp_normalized=norm_end,
                        end_timestamp_ms=to_ms(norm_end),
                        temporal_uncertainty_seconds=1.0,
                        alignment_status=alignment_status,
                        source_reference={
                            "observed_evidence": ve.get("observed_evidence"),
                            "source_frame_indices": ve.get("source_frame_indices", []),
                            "person_count": person_count,
                        },
                    )
                    events.append(fevt)
                    event_counter += 1

                # 2. Key Frame Observations (e.g. frames with weapon or notable detections)
                frame_obs = analysis.get("frame_observations") or []
                for fo in frame_obs:
                    dets = fo.get("detections") or []
                    f_ts = normalize_timestamp(fo.get("timestamp_seconds"))
                    if f_ts is not None and offset_val != 0.0:
                        f_ts = round(f_ts + offset_val, 4)
                    f_idx = fo.get("frame_index", 0)
                    fo_person_cnt = fo.get("person_count")
                    if fo_person_cnt is None:
                        fo_person_cnt = sum(1 for d in dets if d.get("label", "").lower() == "person")

                    weapon_dets = [d for d in dets if d.get("label", "").lower() in WEAPON_LABELS]
                    for wd in weapon_dets:
                        w_label = wd.get("label", "weapon").lower()
                        w_conf = wd.get("confidence")
                        src_conf = float(w_conf) if isinstance(w_conf, (int, float)) else None

                        fevt = ForensicEvent(
                            event_id=f"fevt_{event_counter:03d}",
                            case_id=case_id,
                            evidence_id=ev_id,
                            original_filename=filename,
                            modality="video",
                            source_event_id=f"frame_{f_idx}_det_{w_label}",
                            event_type="weapon_detected",
                            observed_labels=[w_label],
                            extracted_entities=[],
                            description=f"Model detected weapon '{w_label}' in video frame {f_idx} (confidence: {w_conf})",
                            source_confidence=src_conf,
                            original_timestamp=fo.get("timestamp_seconds"),
                            original_time_basis=time_basis,
                            timestamp_normalized=f_ts,
                            timestamp_ms=to_ms(f_ts),
                            end_timestamp_normalized=f_ts,
                            end_timestamp_ms=to_ms(f_ts),
                            temporal_uncertainty_seconds=0.5,
                            alignment_status=alignment_status,
                            source_reference={
                                "frame_index": f_idx,
                                "box": wd.get("box") or wd.get("bbox_xyxy"),
                                "caption": fo.get("caption"),
                                "person_count": fo_person_cnt,
                            },
                        )
                        events.append(fevt)
                        event_counter += 1

            # ---------------------------------------------------------------
            # B. Audio Evidence Parsing
            # ---------------------------------------------------------------
            elif media_type == "audio":
                # 1. Timeline events
                aud_timeline = analysis.get("timeline") or {}
                aud_events = aud_timeline.get("events") or []

                for ae in aud_events:
                    raw_start = ae.get("start_raw")
                    raw_end = ae.get("end_raw", raw_start)
                    norm_start = normalize_timestamp(raw_start)
                    norm_end = normalize_timestamp(raw_end)
                    if norm_start is not None and offset_val != 0.0:
                        norm_start = round(norm_start + offset_val, 4)
                    if norm_end is not None and offset_val != 0.0:
                        norm_end = round(norm_end + offset_val, 4)

                    desc = str(ae.get("description", ""))
                    ev_type = str(ae.get("event_type", "audio_statement")).lower()

                    observed_labels = []
                    # Check for weapon mentions
                    has_weapon_mention = False
                    for w in WEAPON_TEXT_TERMS:
                        if w in desc.lower() or any(w in str(q).lower() for q in ae.get("supporting_quotes", [])):
                            observed_labels.append(w)
                            has_weapon_mention = True

                    if has_weapon_mention and ev_type != "weapon_mentioned":
                        ev_type = "weapon_mentioned"

                    conf = ae.get("confidence")
                    src_conf = float(conf) if isinstance(conf, (int, float)) else None

                    fevt = ForensicEvent(
                        event_id=f"fevt_{event_counter:03d}",
                        case_id=case_id,
                        evidence_id=ev_id,
                        original_filename=filename,
                        modality="audio",
                        source_event_id=str(ae.get("event_id", f"aevt_{event_counter}")),
                        event_type=ev_type,
                        observed_labels=observed_labels,
                        extracted_entities=ae.get("entities_mentioned", []),
                        description=desc,
                        source_confidence=src_conf,
                        original_timestamp=raw_start,
                        original_time_basis=time_basis,
                        timestamp_normalized=norm_start,
                        timestamp_ms=to_ms(norm_start),
                        end_timestamp_normalized=norm_end,
                        end_timestamp_ms=to_ms(norm_end),
                        temporal_uncertainty_seconds=1.5,
                        alignment_status=alignment_status,
                        source_reference={
                            "supporting_quotes": ae.get("supporting_quotes", []),
                            "source_segment_ids": ae.get("source_segment_ids", []),
                            "is_allegation": ae.get("is_allegation", False),
                            "stated_location": ae.get("stated_location"),
                        },
                    )
                    events.append(fevt)
                    event_counter += 1

                # 2. Significant acoustic segments from transcript
                transcription = analysis.get("transcription") or {}
                _SPEECH_ENTITY_KEYWORDS = {
                    "person", "customer", "man", "woman", "individual", "suspect",
                    "counter", "doorway", "jacket", "box", "bag", "car", "vehicle",
                    "store", "shop", "entrance", "exit", "door", "window",
                }
                for seg in transcription.get("segments", []):
                    seg_text = seg.get("text", "")

                    # 2a. Fallback: ingest ALL transcript segments as speech_statement
                    #     events when no LLM-extracted audio events exist
                    if not aud_events and seg_text.strip():
                        seg_ts = normalize_timestamp(seg.get("start"))
                        seg_end = normalize_timestamp(seg.get("end"))
                        if seg_ts is not None and offset_val != 0.0:
                            seg_ts = round(seg_ts + offset_val, 4)
                        if seg_end is not None and offset_val != 0.0:
                            seg_end = round(seg_end + offset_val, 4)

                        seg_lower = seg_text.lower()
                        matched_entities = [kw for kw in sorted(_SPEECH_ENTITY_KEYWORDS) if kw in seg_lower]
                        obs_labels = ["speaker_statement"] + matched_entities

                        conf = seg.get("confidence")
                        src_conf = float(conf) if isinstance(conf, (int, float)) else None

                        fevt = ForensicEvent(
                            event_id=f"fevt_{event_counter:03d}",
                            case_id=case_id,
                            evidence_id=ev_id,
                            original_filename=filename,
                            modality="audio",
                            source_event_id=f"seg_{seg.get('id', event_counter)}",
                            event_type="speech_statement",
                            observed_labels=obs_labels,
                            extracted_entities=matched_entities,
                            description=f'Witness statement: "{seg_text.strip()}"',
                            source_confidence=src_conf,
                            original_timestamp=seg.get("start"),
                            original_time_basis=time_basis,
                            timestamp_normalized=seg_ts,
                            timestamp_ms=to_ms(seg_ts),
                            end_timestamp_normalized=seg_end,
                            end_timestamp_ms=to_ms(seg_end),
                            temporal_uncertainty_seconds=1.0,
                            alignment_status=alignment_status,
                            source_reference={
                                "quote": seg_text.strip(),
                                "segment_id": seg.get("id"),
                                "is_allegation": True,
                            },
                        )
                        events.append(fevt)
                        event_counter += 1

                    # 2b. Check for unarmed or solitude alibi statements
                    is_unarmed_claim = any(u in seg_text.lower() for u in UNARMED_INDICATORS)
                    is_solitude_claim = any(s in seg_text.lower() for s in SOLITUDE_INDICATORS)

                    if is_unarmed_claim or is_solitude_claim:
                        seg_ts = normalize_timestamp(seg.get("start"))
                        seg_end = normalize_timestamp(seg.get("end"))
                        if seg_ts is not None and offset_val != 0.0:
                            seg_ts = round(seg_ts + offset_val, 4)
                        if seg_end is not None and offset_val != 0.0:
                            seg_end = round(seg_end + offset_val, 4)
                        claim_type = "unarmed_claim" if is_unarmed_claim else "solitude_claim"

                        conf = seg.get("confidence")
                        src_conf = float(conf) if isinstance(conf, (int, float)) else None

                        fevt = ForensicEvent(
                            event_id=f"fevt_{event_counter:03d}",
                            case_id=case_id,
                            evidence_id=ev_id,
                            original_filename=filename,
                            modality="audio",
                            source_event_id=f"seg_{seg.get('id', event_counter)}",
                            event_type=claim_type,
                            observed_labels=["speaker_claim", claim_type],
                            extracted_entities=[],
                            description=f"Speaker stated: \"{seg_text}\"",
                            source_confidence=src_conf,
                            original_timestamp=seg.get("start"),
                            original_time_basis=time_basis,
                            timestamp_normalized=seg_ts,
                            timestamp_ms=to_ms(seg_ts),
                            end_timestamp_normalized=seg_end,
                            end_timestamp_ms=to_ms(seg_end),
                            temporal_uncertainty_seconds=1.0,
                            alignment_status=alignment_status,
                            source_reference={
                                "quote": seg_text,
                                "segment_id": seg.get("id"),
                            },
                        )
                        events.append(fevt)
                        event_counter += 1

            # ---------------------------------------------------------------
            # C. Image Evidence Parsing (Static / Unknown Time)
            # ---------------------------------------------------------------
            elif media_type == "image":
                obj_det = analysis.get("object_detection") or {}
                dets = obj_det.get("detections") or []
                scene_res = analysis.get("scene_classification") or {}
                top_scene = scene_res.get("top_scene")
                caption_res = analysis.get("image_caption") or {}
                caption_text = caption_res.get("caption", "")

                # 1. Object Detections (weapons, persons)
                for det_idx, d in enumerate(dets):
                    label = d.get("label", "object").lower()
                    conf = d.get("confidence")
                    src_conf = float(conf) if isinstance(conf, (int, float)) else None

                    is_weapon = label in WEAPON_LABELS
                    ev_type = "weapon_detected" if is_weapon else f"{label}_detection"

                    fevt = ForensicEvent(
                        event_id=f"fevt_{event_counter:03d}",
                        case_id=case_id,
                        evidence_id=ev_id,
                        original_filename=filename,
                        modality="image",
                        source_event_id=f"img_det_{det_idx}",
                        event_type=ev_type,
                        observed_labels=[label],
                        extracted_entities=[],
                        description=f"Static photograph visual detection: '{label}' (confidence: {conf})",
                        source_confidence=src_conf,
                        original_timestamp=None,
                        original_time_basis="unknown",
                        timestamp_normalized=None,
                        timestamp_ms=None,
                        end_timestamp_normalized=None,
                        end_timestamp_ms=None,
                        temporal_uncertainty_seconds=None,
                        alignment_status="temporally_unknown",
                        source_reference={
                            "bbox": d.get("bbox_xyxy"),
                            "class_id": d.get("class_id"),
                        },
                    )
                    events.append(fevt)
                    event_counter += 1

                # 2. Scene Context
                if top_scene:
                    top_score = scene_res.get("confidence") or scene_res.get("top_score")
                    src_conf = float(top_score) if isinstance(top_score, (int, float)) else None

                    fevt = ForensicEvent(
                        event_id=f"fevt_{event_counter:03d}",
                        case_id=case_id,
                        evidence_id=ev_id,
                        original_filename=filename,
                        modality="image",
                        source_event_id="img_scene",
                        event_type="scene_classification",
                        observed_labels=[top_scene],
                        extracted_entities=[],
                        description=f"Classified scene environment: '{top_scene}'",
                        source_confidence=src_conf,
                        original_timestamp=None,
                        original_time_basis="unknown",
                        timestamp_normalized=None,
                        timestamp_ms=None,
                        alignment_status="temporally_unknown",
                        source_reference={
                            "candidates": scene_res.get("candidates", [])[:3],
                            "caption": caption_text,
                        },
                    )
                    events.append(fevt)
                    event_counter += 1

        return events

    # -----------------------------------------------------------------------
    # Cross-Modal Audit & Relationship Resolution
    # -----------------------------------------------------------------------
    def audit_cross_modal_relationships(
        self,
        events: list[ForensicEvent],
        hash_index: CrossModalHashIndex,
        temporal_tolerance: float = 2.0,
    ) -> list[CrossModalRelationship]:
        """
        Cross-modal audit comparing visual detections, captions, audio events, and transcripts.
        Detects corroborations, potential consistencies, thematic matches, and explicit contradictions.
        Deduplicates repeated relationships between the same evidence sources for the same semantic concept.
        """
        relationships: list[CrossModalRelationship] = []
        rel_counter = 1
        seen_dedup_keys: set[tuple[str, str, str, Any]] = set()

        def add_relationship(
            ev1: ForensicEvent,
            ev2: ForensicEvent,
            rel_type: str,
            rule: str,
            alignment_basis: str,
            expl: str,
            uncs: list[str],
            dedup_window: Any = None,
        ) -> CrossModalRelationship | None:
            nonlocal rel_counter
            # Order evidence IDs deterministically for deduplication key
            e1_id, e2_id = sorted([ev1.evidence_id, ev2.evidence_id])
            d_key = (e1_id, e2_id, rule, dedup_window)
            if d_key in seen_dedup_keys:
                return None
            seen_dedup_keys.add(d_key)

            rel = CrossModalRelationship(
                relationship_id=f"rel_{rel_counter:03d}",
                source_event_ids=[ev1.event_id, ev2.event_id],
                relationship_type=rel_type,
                matching_rule=rule,
                time_alignment_basis=alignment_basis,
                explanation=expl,
                uncertainties=uncs,
                source_a_evidence_id=ev1.evidence_id,
                source_b_evidence_id=ev2.evidence_id,
                source_a_filename=ev1.original_filename,
                source_b_filename=ev2.original_filename,
                source_a_modality=ev1.modality,
                source_b_modality=ev2.modality,
                source_a_timestamp=ev1.timestamp_normalized,
                source_b_timestamp=ev2.timestamp_normalized,
            )
            relationships.append(rel)
            rel_counter += 1
            return rel

        # Pairwise comparison across distinct evidence sources
        for i, ev_a in enumerate(events):
            for ev_b in events[i + 1 :]:
                # Only compare observations originating from distinct evidence records or distinct modalities
                if ev_a.evidence_id == ev_b.evidence_id and ev_a.modality == ev_b.modality:
                    continue

                # -----------------------------------------------------------
                # Rule 1: Weapon Corroboration vs. Potential Consistency
                # -----------------------------------------------------------
                is_weapon_a = ev_a.event_type in ("weapon_detected", "weapon_mentioned") or any(
                    w in ev_a.observed_labels for w in WEAPON_LABELS or w in WEAPON_TEXT_TERMS
                )
                is_weapon_b = ev_b.event_type in ("weapon_detected", "weapon_mentioned") or any(
                    w in ev_b.observed_labels for w in WEAPON_LABELS or w in WEAPON_TEXT_TERMS
                )

                if is_weapon_a and is_weapon_b:
                    is_prox, dist, prox_expl = check_temporal_proximity(ev_a, ev_b, tolerance_seconds=temporal_tolerance)
                    weapons_a = {w for w in ev_a.observed_labels if w in WEAPON_LABELS or w in WEAPON_TEXT_TERMS}
                    weapons_b = {w for w in ev_b.observed_labels if w in WEAPON_LABELS or w in WEAPON_TEXT_TERMS}
                    shared_weapons = weapons_a.intersection(weapons_b)
                    type_compatible = bool(shared_weapons) or ("weapon" in (weapons_a | weapons_b))

                    if is_prox and type_compatible:
                        matched_label = ", ".join(sorted(shared_weapons)) if shared_weapons else "weapon"
                        window = int(ev_a.timestamp_normalized // 3.0) if ev_a.timestamp_normalized is not None else 0
                        rel = add_relationship(
                            ev_a, ev_b,
                            rel_type="corroboration",
                            rule="weapon_presence_cross_modal_corroboration",
                            alignment_basis="synchronized_timeline",
                            expl=(
                                f"Visual observation ({ev_a.modality} • {ev_a.original_filename}) and audio reference "
                                f"({ev_b.modality} • {ev_b.original_filename}) co-occur within temporal proximity ({prox_expl}) "
                                f"regarding weapon presence ('{matched_label}'). "
                                f"Note: Specific physical possession and bearer identity remain unverified without forensic confirmation."
                            ),
                            uncs=["Automated co-occurrence; continuous physical possession and bearer identity not verified"],
                            dedup_window=window,
                        )
                        if rel:
                            ev_a.corroboration_links.append(ev_b.event_id)
                            ev_b.corroboration_links.append(ev_a.event_id)
                            ev_a.relationship_status = "corroborated"
                            ev_b.relationship_status = "corroborated"
                    elif is_prox and not type_compatible:
                        window = int(ev_a.timestamp_normalized // 3.0) if ev_a.timestamp_normalized is not None else 0
                        rel = add_relationship(
                            ev_a, ev_b,
                            rel_type="potential_consistency",
                            rule="discrepant_weapon_types_unconfirmed",
                            alignment_basis="synchronized_timeline",
                            expl=(
                                f"Sources ({ev_a.original_filename} and {ev_b.original_filename}) co-occur in time ({prox_expl}), "
                                f"but report disparate weapon types ({weapons_a or 'unknown'} vs {weapons_b or 'unknown'}). "
                                f"Corroboration withheld pending weapon classification verification."
                            ),
                            uncs=["Mismatched weapon categories; could represent distinct objects or model classification error"],
                            dedup_window=window,
                        )
                        if rel:
                            if ev_a.relationship_status == "single_source":
                                ev_a.relationship_status = "potentially_consistent"
                            if ev_b.relationship_status == "single_source":
                                ev_b.relationship_status = "potentially_consistent"
                    else:
                        rel = add_relationship(
                            ev_a, ev_b,
                            rel_type="potential_consistency",
                            rule="weapon_mention_unaligned_consistency",
                            alignment_basis="semantic_unaligned",
                            expl=(
                                f"Both {ev_a.modality} ({ev_a.original_filename}) and {ev_b.modality} ({ev_b.original_filename}) "
                                f"reference weapon presence, but lack verified temporal synchronization ({prox_expl})."
                            ),
                            uncs=["Temporal alignment unconfirmed; observations may refer to distinct moments or items"],
                            dedup_window="all",
                        )
                        if rel:
                            if ev_a.relationship_status == "single_source":
                                ev_a.relationship_status = "potentially_consistent"
                            if ev_b.relationship_status == "single_source":
                                ev_b.relationship_status = "potentially_consistent"

                # -----------------------------------------------------------
                # Rule 2: Explicit Weapon Contradiction (Claim vs. Detection)
                # -----------------------------------------------------------
                has_weapon_det = (
                    (ev_a.event_type == "weapon_detected" and ev_a.modality in ("image", "video"))
                    or (ev_b.event_type == "weapon_detected" and ev_b.modality in ("image", "video"))
                )
                has_unarmed_claim = (
                    ev_a.event_type == "unarmed_claim" or ev_b.event_type == "unarmed_claim"
                )

                if has_weapon_det and has_unarmed_claim:
                    weapon_ev = ev_a if ev_a.event_type == "weapon_detected" else ev_b
                    claim_ev = ev_a if ev_a.event_type == "unarmed_claim" else ev_b

                    is_prox, dist, prox_expl = check_temporal_proximity(weapon_ev, claim_ev, tolerance_seconds=temporal_tolerance)

                    if is_prox:
                        window = int(weapon_ev.timestamp_normalized // 3.0) if weapon_ev.timestamp_normalized is not None else 0
                        rel = add_relationship(
                            ev_a, ev_b,
                            rel_type="conflict",
                            rule="unarmed_claim_vs_weapon_detection_conflict",
                            alignment_basis="synchronized_conflict",
                            expl=(
                                f"Material contradiction on common timeline: Audio statement asserts subject was unarmed ({claim_ev.original_filename}), "
                                f"but concurrent visual detection ({weapon_ev.original_filename}) detects weapon "
                                f"('{', '.join(weapon_ev.observed_labels)}'). "
                                f"Note: Investigator must verify whether the statement refers to the specific individual detected."
                            ),
                            uncs=["Speaker identity vs person holding detected weapon requires forensic confirmation"],
                            dedup_window=window,
                        )
                        if rel:
                            ev_a.conflict_links.append(ev_b.event_id)
                            ev_b.conflict_links.append(ev_a.event_id)
                            ev_a.relationship_status = "conflicting"
                            ev_b.relationship_status = "conflicting"
                    else:
                        rel = add_relationship(
                            ev_a, ev_b,
                            rel_type="potential_consistency",
                            rule="unaligned_weapon_discrepancy",
                            alignment_basis="unaligned_discrepancy",
                            expl=(
                                f"Audio statement asserts subject was unarmed ({claim_ev.original_filename}), while visual evidence "
                                f"({weapon_ev.original_filename}) detects a weapon; however, recordings lack synchronized common alignment "
                                f"or occur in different time intervals ({prox_expl}). Direct contradiction cannot be established."
                            ),
                            uncs=["Temporal alignment unconfirmed; subject may have been unarmed at a different time"],
                            dedup_window="all",
                        )
                        if rel:
                            if ev_a.relationship_status == "single_source":
                                ev_a.relationship_status = "potentially_consistent"
                            if ev_b.relationship_status == "single_source":
                                ev_b.relationship_status = "potentially_consistent"

                # -----------------------------------------------------------
                # Rule 3: Person Presence / Alibi Solitude Contradiction
                # -----------------------------------------------------------
                is_solitude_a = ev_a.event_type == "solitude_claim"
                is_solitude_b = ev_b.event_type == "solitude_claim"
                if is_solitude_a or is_solitude_b:
                    solitude_ev = ev_a if is_solitude_a else ev_b
                    visual_ev = ev_b if is_solitude_a else ev_a

                    is_vis_person = (
                        visual_ev.modality in ("image", "video")
                        and ("person" in visual_ev.observed_labels or visual_ev.event_type in ("person_entered_frame", "person_detection", "visual_observation"))
                    )
                    if is_vis_person:
                        is_prox, dist, prox_expl = check_temporal_proximity(solitude_ev, visual_ev, tolerance_seconds=temporal_tolerance)
                        person_cnt = visual_ev.source_reference.get("person_count")
                        if person_cnt is None:
                            m = re.search(r"(\d+)\s+person", visual_ev.description.lower())
                            person_cnt = int(m.group(1)) if m else 1

                        is_multi_person = person_cnt > 1
                        is_entry = visual_ev.event_type in ("person_entered_frame", "arrival")

                        if is_prox and (is_multi_person or is_entry):
                            window = int(visual_ev.timestamp_normalized // 3.0) if visual_ev.timestamp_normalized is not None else 0
                            rel = add_relationship(
                                ev_a, ev_b,
                                rel_type="conflict",
                                rule="solitude_claim_vs_person_detection_conflict",
                                alignment_basis="synchronized_conflict",
                                expl=(
                                    f"Contradiction on common timeline: Audio statement asserts subject was alone / nobody was present "
                                    f"({solitude_ev.original_filename}), but concurrent visual evidence ({visual_ev.original_filename}) "
                                    f"indicates {'multiple persons (' + str(person_cnt) + ')' if is_multi_person else 'a person arrival/entry transition'}."
                                ),
                                uncs=["Speaker statement contradicts multi-person observation; verify speaker presence"],
                                dedup_window=window,
                            )
                            if rel:
                                ev_a.conflict_links.append(ev_b.event_id)
                                ev_b.conflict_links.append(ev_a.event_id)
                                ev_a.relationship_status = "conflicting"
                                ev_b.relationship_status = "conflicting"
                        elif is_prox and not is_multi_person and not is_entry:
                            window = int(visual_ev.timestamp_normalized // 3.0) if visual_ev.timestamp_normalized is not None else 0
                            rel = add_relationship(
                                ev_a, ev_b,
                                rel_type="potential_consistency",
                                rule="solitude_claim_consistent_with_single_person",
                                alignment_basis="synchronized_timeline",
                                expl=(
                                    f"Audio statement asserts subject was alone ({solitude_ev.original_filename}), and concurrent visual "
                                    f"observation ({visual_ev.original_filename}) detects exactly 1 person. This observation is consistent "
                                    f"with the presence of the solitary speaker alone."
                                ),
                                uncs=["Presumes detected individual is the speaker; biometric confirmation required"],
                                dedup_window=window,
                            )
                            if rel:
                                if ev_a.relationship_status == "single_source":
                                    ev_a.relationship_status = "potentially_consistent"
                                if ev_b.relationship_status == "single_source":
                                    ev_b.relationship_status = "potentially_consistent"
                        else:
                            rel = add_relationship(
                                ev_a, ev_b,
                                rel_type="potential_consistency",
                                rule="unaligned_solitude_observation",
                                alignment_basis="semantic_unaligned",
                                expl=(
                                    f"Audio solitude statement ({solitude_ev.original_filename}) and visual person observation "
                                    f"({visual_ev.original_filename}) lack common temporal synchronization or occur in different windows ({prox_expl})."
                                ),
                                uncs=["Observations may refer to separate moments"],
                                dedup_window="all",
                            )
                            if rel:
                                if ev_a.relationship_status == "single_source":
                                    ev_a.relationship_status = "potentially_consistent"
                                if ev_b.relationship_status == "single_source":
                                    ev_b.relationship_status = "potentially_consistent"

                # -----------------------------------------------------------
                # Rule 4: Aggression / Altercation Corroboration
                # -----------------------------------------------------------
                is_aggr_a = any(a in ev_a.event_type or a in ev_a.description.lower() for a in AGGRESSION_LABELS)
                is_aggr_b = any(a in ev_b.event_type or a in ev_b.description.lower() for a in AGGRESSION_LABELS)
                if is_aggr_a and is_aggr_b and ev_a.modality != ev_b.modality:
                    is_prox_aggr, dist_aggr, prox_expl_aggr = check_temporal_proximity(ev_a, ev_b, tolerance_seconds=temporal_tolerance)
                    if is_prox_aggr:
                        window = int(ev_a.timestamp_normalized // 3.0) if ev_a.timestamp_normalized is not None else 0
                        rel = add_relationship(
                            ev_a, ev_b,
                            rel_type="corroboration",
                            rule="physical_altercation_cross_modal_corroboration",
                            alignment_basis="synchronized_timeline",
                            expl=(
                                f"Acoustic aggression finding ({ev_a.modality}) and visual altercation "
                                f"observation ({ev_b.modality}) mutually corroborate an altercation ({prox_expl_aggr})."
                            ),
                            uncs=[],
                            dedup_window=window,
                        )
                        if rel:
                            ev_a.corroboration_links.append(ev_b.event_id)
                            ev_b.corroboration_links.append(ev_a.event_id)
                            ev_a.relationship_status = "corroborated"
                            ev_b.relationship_status = "corroborated"

                # -----------------------------------------------------------
                # Rule 5: Cross-Modal Person Mention Matching
                # (spoken person references vs visual person detections)
                # Deduplicated: one consolidated thematic match relationship per evidence pair.
                # Classified as 'thematic_match' rather than unqualified corroboration.
                # -----------------------------------------------------------
                _PERSON_ENTITY_LABELS = {"person", "customer", "man", "woman", "individual", "suspect"}
                is_speech_person_a = (
                    ev_a.modality == "audio"
                    and ev_a.event_type in ("speech_statement", "audio_statement")
                    and (any(p in ev_a.observed_labels for p in _PERSON_ENTITY_LABELS) or any(p in ev_a.description.lower() for p in _PERSON_ENTITY_LABELS))
                )
                is_visual_person_a = (
                    ev_a.modality in ("image", "video")
                    and ("person" in ev_a.observed_labels or "person" in ev_a.description.lower() or ev_a.event_type in ("person_observation", "person_entered_frame", "person_detection", "visual_observation"))
                )
                is_speech_person_b = (
                    ev_b.modality == "audio"
                    and ev_b.event_type in ("speech_statement", "audio_statement")
                    and (any(p in ev_b.observed_labels for p in _PERSON_ENTITY_LABELS) or any(p in ev_b.description.lower() for p in _PERSON_ENTITY_LABELS))
                )
                is_visual_person_b = (
                    ev_b.modality in ("image", "video")
                    and ("person" in ev_b.observed_labels or "person" in ev_b.description.lower() or ev_b.event_type in ("person_observation", "person_entered_frame", "person_detection", "visual_observation"))
                )

                cross_person = (is_speech_person_a and is_visual_person_b) or (is_speech_person_b and is_visual_person_a)
                if cross_person:
                    speech_ev = ev_a if is_speech_person_a else ev_b
                    visual_ev = ev_b if is_speech_person_a else ev_a
                    is_prox_p, dist_p, prox_expl_p = check_temporal_proximity(speech_ev, visual_ev, tolerance_seconds=temporal_tolerance)

                    terms = ", ".join(speech_ev.extracted_entities or speech_ev.observed_labels) or "person"
                    rel = add_relationship(
                        speech_ev, visual_ev,
                        rel_type="thematic_match",
                        rule="spoken_person_visual_person_cross_modal",
                        alignment_basis="synchronized_timeline" if is_prox_p else "semantic_unaligned",
                        expl=(
                            f"Audio witness statement ({speech_ev.original_filename}) mentions an individual ('{terms}'), "
                            f"and visual evidence ({visual_ev.original_filename}) contains corresponding person detection(s). "
                            f"Temporal alignment: {prox_expl_p}. "
                            f"Classified as thematic match only: recordings lack common clock synchronization and biometric identity matching; "
                            f"mere co-presence of a person in audio transcript and video/image does not establish identity or corroboration."
                        ),
                        uncs=[
                            "Independent time bases; spoken reference and visual detections lack synchronized temporal alignment",
                            "No biometric or clothing confirmation; spoken person and detected individual may be different individuals",
                            "Generic presence of individuals is insufficient to establish investigative corroboration",
                        ],
                        dedup_window="all",
                    )
                    if rel:
                        if ev_a.relationship_status == "single_source":
                            ev_a.relationship_status = "potentially_consistent"
                        if ev_b.relationship_status == "single_source":
                            ev_b.relationship_status = "potentially_consistent"

        return relationships

    # -----------------------------------------------------------------------
    # Anchor Event Identification & Highlighting
    # -----------------------------------------------------------------------
    def highlight_anchors(
        self,
        events: list[ForensicEvent],
        relationships: list[CrossModalRelationship],
    ) -> list[dict[str, Any]]:
        """
        Flag and highlight critical anchor events:
        - Weapons detected or mentioned (strictly distinguished).
        - Person entry or arrival events.
        - Significant distress or aggression events.
        - Flagged cross-modal contradictions.
        """
        anchors: list[dict[str, Any]] = []
        anchor_counter = 1

        for ev in events:
            is_anchor = False
            reason = None
            category = None

            # 1. Weapon detected visually
            if ev.event_type == "weapon_detected":
                is_anchor = True
                category = "weapon"
                w_labels = ", ".join(ev.observed_labels) or "weapon"
                reason = f"Visual model detection of weapon ({w_labels}) in {ev.modality}"

            # 2. Weapon mentioned in audio
            elif ev.event_type == "weapon_mentioned":
                is_anchor = True
                category = "weapon"
                reason = f"Weapon mentioned in audio recording transcript ({ev.original_filename})"

            # 3. Person entry
            elif ev.event_type in ("person_entered_frame", "arrival"):
                is_anchor = True
                category = "person_transition"
                reason = f"Person presence transition reported by source analysis in {ev.modality}; physical entry unconfirmed without continuous multi-camera tracking"

            # 4. Distress / Altercation
            elif any(a in ev.event_type for a in ("struggle", "aggression", "distress_call", "altercation")):
                is_anchor = True
                category = "incident_event"
                reason = f"High-significance incident event ({ev.event_type}) in {ev.modality}"

            # 5. Cross-modal conflict
            elif ev.conflict_links:
                is_anchor = True
                category = "cross_modal_conflict"
                reason = "Event is subject to an active cross-modal evidentiary conflict"

            # 5b. Witness speech statement with entity mentions
            elif ev.event_type == "speech_statement" and ev.extracted_entities:
                is_anchor = True
                category = "witness_statement"
                entities_str = ", ".join(ev.extracted_entities[:5])
                reason = f"Witness speech statement mentioning: {entities_str} ({ev.original_filename})"

            # 6. Corroborated event
            elif ev.corroboration_links and ev.certainty_level == "high":
                is_anchor = True
                category = "corroborated_incident"
                reason = "Multi-source corroborated event with high cross-modal consistency"

            if is_anchor:
                ev.is_anchor = True
                ev.anchor_reason = reason
                anchors.append({
                    "anchor_id": f"anc_{anchor_counter:03d}",
                    "event_id": ev.event_id,
                    "category": category,
                    "reason": reason,
                    "modality": ev.modality,
                    "original_filename": ev.original_filename,
                    "timestamp_normalized": ev.timestamp_normalized,
                    "evidence_consistency_score": ev.evidence_consistency_score,
                    "certainty_level": ev.certainty_level,
                })
                anchor_counter += 1

        return anchors

    # -----------------------------------------------------------------------
    # Grounded Narrative Generation
    # -----------------------------------------------------------------------
    def generate_grounded_narrative(
        self,
        case_title: str,
        events: list[ForensicEvent],
        relationships: list[CrossModalRelationship],
        anchors: list[dict[str, Any]],
    ) -> str:
        """
        Generate a strictly factual, chronological, evidence-grounded textual narrative.
        Never fabricates claims absent from evidence records.
        """
        lines = []
        lines.append(f"### Forensic Timeline Narrative: {case_title}")
        lines.append("")

        aligned = [e for e in events if e.alignment_status == "aligned_common_timeline" and e.timestamp_normalized is not None]
        aligned.sort(key=lambda x: x.timestamp_normalized or 0.0)

        unaligned = [e for e in events if e.alignment_status != "aligned_common_timeline"]

        # 1. Aligned Timeline Chronology
        if aligned:
            lines.append("#### Chronological Sequence (Common Time Axis)")
            for e in aligned:
                ts_str = f"[{e.timestamp_normalized:.2f}s"
                if e.end_timestamp_normalized and e.end_timestamp_normalized > e.timestamp_normalized:
                    ts_str += f" - {e.end_timestamp_normalized:.2f}s]"
                else:
                    ts_str += "]"
                anchor_tag = f" [ANCHOR: {e.anchor_reason}]" if e.is_anchor else ""
                lines.append(f"- **{ts_str}** ({e.modality.upper()} • {e.original_filename}): {e.description} (Certainty: {e.certainty_level.upper()}){anchor_tag}")
            lines.append("")

        # 2. Unaligned or Static Evidence
        if unaligned:
            lines.append("#### Independent & Unaligned Evidence Observations")
            for e in unaligned:
                time_note = f"at media-relative offset {e.timestamp_normalized:.2f}s" if e.timestamp_normalized is not None else "temporal coordinate unknown"
                lines.append(f"- ({e.modality.upper()} • {e.original_filename}): {e.description} ({time_note}; Certainty: {e.certainty_level.upper()})")
            lines.append("")

        # 3. Cross-Modal Corroboration Summary
        corrobs = [r for r in relationships if r.relationship_type == "corroboration"]
        if corrobs:
            lines.append("#### Cross-Modal Corroborations")
            for c in corrobs:
                lines.append(f"- **{c.matching_rule}**: {c.explanation}")
            lines.append("")

        # 4. Critical Conflicts & Contradictions
        conflicts = [r for r in relationships if r.relationship_type == "conflict"]
        if conflicts:
            lines.append("#### Evidentiary Inconsistencies & Contradictions")
            for cf in conflicts:
                lines.append(f"- ⚠️ **CONFLICT ({cf.matching_rule})**: {cf.explanation}")
            lines.append("")

        lines.append("> **FORENSIC NOTICE**: Narrative is synthesized from algorithmic model outputs. Model detections and extracted speech claims are probabilistic and do not constitute verified facts.")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Main Fusion Workflow
    # -----------------------------------------------------------------------
    def fuse_case(
        self,
        case_data: dict[str, Any],
        sync_groups: dict[str, str] | None = None,
        default_temporal_tolerance: float = 2.0,
    ) -> dict[str, Any]:
        """
        Execute end-to-end Activity 5.1 Multimodal Fusion:
        1. Extract and normalize events with fractional timestamps.
        2. Construct deterministic SHA-256 hash index mapping to event lists.
        3. Conduct cross-modal audit (corroborations & conflicts).
        4. Calculate weighted certainty and consistency scores.
        5. Identify and highlight anchor events.
        6. Generate structured chronological timeline and grounded narrative.
        """
        case_id = case_data.get("case_id", "CASE-UNKNOWN")
        case_title = case_data.get("title", "Forensic Investigation")

        # 1. Extract raw normalized events
        events = self.extract_events_from_case_evidence(case_data, sync_groups=sync_groups)

        # 2. Build cross-modal hash index
        hash_index = CrossModalHashIndex()
        for ev in events:
            hash_index.index_event(ev)

        # 3. Audit cross-modal relationships (corroboration and conflicts)
        relationships = self.audit_cross_modal_relationships(
            events, hash_index, temporal_tolerance=default_temporal_tolerance
        )

        # 4. Calculate certainty scores with safeguards
        for ev in events:
            corrob_mods = set()
            for other in events:
                if other.event_id in ev.corroboration_links:
                    corrob_mods.add(other.modality)

            score, certainty, explanation = CertaintyScorer.score_event(
                event=ev,
                has_corroboration=bool(ev.corroboration_links),
                has_conflict=bool(ev.conflict_links),
                corroborating_modalities=corrob_mods,
            )
            ev.evidence_consistency_score = score
            ev.certainty_level = certainty
            ev.scoring_explanation = explanation

        # 5. Highlight anchor events
        anchors = self.highlight_anchors(events, relationships)

        # 6. Sort chronological events: aligned events first (by normalized timestamp), followed by unaligned
        def event_sort_key(e: ForensicEvent) -> tuple[int, float]:
            if e.timestamp_normalized is not None:
                # 0 for aligned common timeline, 1 for unaligned with timestamp
                group = 0 if e.alignment_status == "aligned_common_timeline" else 1
                return (group, float(e.timestamp_normalized))
            return (2, 0.0)  # temporally unknown last

        events.sort(key=event_sort_key)

        # Re-assign sequential stable event IDs in sorted order for consistent export.
        # Use single-pass mapping to prevent ID collision and double-replacement corruption.
        id_map = {ev.event_id: f"fevt_{idx:03d}" for idx, ev in enumerate(events, start=1)}

        for ev in events:
            old_id = ev.event_id
            ev.event_id = id_map.get(old_id, old_id)
            ev.corroboration_links = [id_map.get(cid, cid) for cid in ev.corroboration_links]
            ev.conflict_links = [id_map.get(cid, cid) for cid in ev.conflict_links]

        for r in relationships:
            r.source_event_ids = [id_map.get(sid, sid) for sid in r.source_event_ids]

        for a in anchors:
            a["event_id"] = id_map.get(a["event_id"], a["event_id"])

        # 7. Generate grounded narrative
        narrative = self.generate_grounded_narrative(case_title, events, relationships, anchors)

        # Segregate aligned and unaligned for dashboard consumers
        aligned_events = [e.to_dict() for e in events if e.alignment_status == "aligned_common_timeline"]
        unaligned_events = [e.to_dict() for e in events if e.alignment_status != "aligned_common_timeline"]

        now_iso = datetime.now(timezone.utc).isoformat()

        return {
            "schema_version": "1.0.0",
            "case_id": case_id,
            "title": case_title,
            "fused_at": now_iso,
            "summary": {
                "total_events": len(events),
                "aligned_events_count": len(aligned_events),
                "unaligned_events_count": len(unaligned_events),
                "corroborations_count": sum(1 for r in relationships if r.relationship_type in ("corroboration", "qualified_corroboration")),
                "thematic_matches_count": sum(1 for r in relationships if r.relationship_type == "thematic_match"),
                "potentially_related_count": sum(1 for r in relationships if r.relationship_type in ("potentially_related", "potential_consistency")),
                "conflicts_count": sum(1 for r in relationships if r.relationship_type == "conflict"),
                "anchor_events_count": len(anchors),
            },
            "events": [e.to_dict() for e in events],
            "timeline_aligned_events": aligned_events,
            "unaligned_events": unaligned_events,
            "relationships": [r.to_dict() for r in relationships],
            "anchors": anchors,
            "hash_index": hash_index.to_dict(),
            "grounded_narrative": narrative,
            "forensic_disclaimer": (
                "Evidence consistency scores and cross-modal links are automated heuristic "
                "decision-support markers. They do not constitute scientific proof or legal conclusions. "
                "Independent human review of original recordings and imagery is strictly required."
            ),
        }
