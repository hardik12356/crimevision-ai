"""
Unit and integration tests for CrimeVisionAI Epic 5.1:
Activity 5.1: Fuse image, audio, and video outputs into a unified forensic timeline.

Validation coverage:
1. Fractional timestamps and millisecond precision are preserved throughout normalization.
2. Multiple events within the same second are retained without collision loss.
3. Audio/video synchronization aligns on common timeline when source/offset are established.
4. Unrelated/unaligned evidence is preserved with independent time basis, NOT forced onto common timeline.
5. Static images with unknown capture times remain temporally unknown (None).
6. Cross-modal corroborations are established with source references and explanation.
7. Material contradictions (unarmed claim vs weapon detection, solitude vs person) are flagged as conflicts.
8. Deterministic SHA-256 hash index maps to lists of event references and supports time/term lookups.
9. Transparent certainty scoring applies safeguards (single-source cannot be high, conflicts force low).
10. Full JSON serialization/deserialization round-trip maintains schema and numerical precision.
11. Flask API endpoints GET /api/cases/<id>/fusion and POST /api/cases/<id>/fusion return valid data.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app import create_app
from app.services.case_service import CaseService
from app.services.fusion_service import (
    CertaintyScorer,
    CrossModalHashIndex,
    ForensicEvent,
    FusionService,
    normalize_timestamp,
    to_ms,
)


# ===========================================================================
# 1. Temporal Normalization & Precision Tests
# ===========================================================================

def test_normalize_timestamp_preserves_fractional_seconds():
    """Verify fractional seconds are preserved, not truncated or converted to integers."""
    assert normalize_timestamp(5.250) == 5.250
    assert normalize_timestamp(0.123) == 0.123
    assert normalize_timestamp("01:23.450") == 83.450
    assert normalize_timestamp("0:05.25") == 5.250
    assert normalize_timestamp("4.875s") == 4.875
    assert normalize_timestamp(None) is None
    assert normalize_timestamp("unknown") is None
    assert normalize_timestamp(-1.5) is None


def test_to_ms_conversion():
    """Verify integer millisecond conversion preserves precision."""
    assert to_ms(5.250) == 5250
    assert to_ms(0.005) == 5
    assert to_ms(None) is None


# ===========================================================================
# 2. Same-Second Multiple Event Retention & Hash Index Tests
# ===========================================================================

def test_same_second_events_retained_in_hash_index():
    """Verify that multiple events occurring in the exact same second are not lost."""
    index = CrossModalHashIndex()

    ev1 = ForensicEvent(
        event_id="fevt_001",
        case_id="CASE-01",
        evidence_id="EVD-01",
        original_filename="cctv.mp4",
        modality="video",
        source_event_id="det_1",
        event_type="person_detection",
        observed_labels=["person"],
        timestamp_normalized=5.25,
        original_time_basis="media_relative:common",
    )
    ev2 = ForensicEvent(
        event_id="fevt_002",
        case_id="CASE-01",
        evidence_id="EVD-01",
        original_filename="cctv.mp4",
        modality="video",
        source_event_id="det_2",
        event_type="weapon_detected",
        observed_labels=["knife"],
        timestamp_normalized=5.75,
        original_time_basis="media_relative:common",
    )

    index.index_event(ev1)
    index.index_event(ev2)

    # Both events fall into second bucket 5
    looked_up = index.lookup_by_time(5.5, time_basis="media_relative:common", tolerance_seconds=1.0)
    assert len(looked_up) == 2
    event_ids = [item["event_id"] for item in looked_up]
    assert "fevt_001" in event_ids
    assert "fevt_002" in event_ids


def test_hash_index_deterministic_keys():
    """Verify that hashing uses deterministic SHA-256 and retrieves by term."""
    index = CrossModalHashIndex()
    ev = ForensicEvent(
        event_id="fevt_001",
        case_id="CASE-01",
        evidence_id="EVD-01",
        original_filename="audio.wav",
        modality="audio",
        source_event_id="evt_01",
        event_type="weapon_mentioned",
        observed_labels=["firearm"],
        timestamp_normalized=12.5,
        original_time_basis="media_relative:common",
    )
    index.index_event(ev)

    results = index.lookup_by_term("firearm")
    assert len(results) == 1
    assert results[0]["event_id"] == "fevt_001"

    meta = index.to_dict()
    assert meta["total_time_buckets"] >= 1
    assert meta["total_term_buckets"] >= 1


# ===========================================================================
# 3. Synchronization & Alignment Rules Tests
# ===========================================================================

def test_temporal_alignment_distinguishes_shared_vs_independent():
    """
    Verify synchronization rules:
    - Matching file stems alone do NOT establish alignment (preserves independent time bases).
    - Explicit sync groups or offsets align on common timeline.
    - Unrelated audio remains independent.
    - Static image remains temporally unknown.
    """
    service = FusionService()

    case_data = {
        "case_id": "CASE-TEST-SYNC",
        "title": "Synchronization Test Case",
        "evidence": [
            {
                "evidence_id": "EVD-VID",
                "original_filename": "surveillance.mp4",
                "media_type": "video",
                "analysis": {
                    "timeline": {
                        "events": [
                            {"event_id": "vevt_1", "start_raw": 5.25, "description": "Person entered view"}
                        ]
                    }
                },
            },
            {
                "evidence_id": "EVD-AUD-SHARED",
                "original_filename": "surveillance.wav",  # Matches surveillance stem, but no sync group yet
                "media_type": "audio",
                "analysis": {
                    "timeline": {
                        "events": [
                            {"event_id": "aevt_1", "start_raw": 5.30, "description": "Footsteps heard"}
                        ]
                    }
                },
            },
            {
                "evidence_id": "EVD-AUD-UNRELATED",
                "original_filename": "dispatch_call.wav",
                "media_type": "audio",
                "analysis": {
                    "timeline": {
                        "events": [
                            {"event_id": "aevt_2", "start_raw": 5.25, "description": "Radio transmission"}
                        ]
                    }
                },
            },
            {
                "evidence_id": "EVD-IMG",
                "original_filename": "evidence_photo.jpg",
                "media_type": "image",
                "analysis": {
                    "object_detection": {
                        "detections": [{"label": "knife", "confidence": 0.95}]
                    }
                },
            },
        ],
    }

    # Requirement 2 check: matching file stems alone do NOT align recordings
    fusion_unaligned = service.fuse_case(case_data)
    unaligned_by_fn = {e["original_filename"]: e for e in fusion_unaligned["events"]}
    assert unaligned_by_fn["surveillance.mp4"]["alignment_status"] == "unaligned_independent_time_basis"
    assert unaligned_by_fn["surveillance.wav"]["alignment_status"] == "unaligned_independent_time_basis"
    assert unaligned_by_fn["dispatch_call.wav"]["alignment_status"] == "unaligned_independent_time_basis"

    # Explicit synchronization establishes common timeline
    sync_groups = {
        "EVD-VID": "cctv_sync_1",
        "EVD-AUD-SHARED": "cctv_sync_1",
    }
    fusion_aligned = service.fuse_case(case_data, sync_groups=sync_groups)
    events_by_filename = {e["original_filename"]: e for e in fusion_aligned["events"]}

    # Synchronized video & audio now share common timeline
    assert events_by_filename["surveillance.mp4"]["alignment_status"] == "aligned_common_timeline"
    assert events_by_filename["surveillance.wav"]["alignment_status"] == "aligned_common_timeline"

    # Unrelated dispatch audio remains independent
    assert events_by_filename["dispatch_call.wav"]["alignment_status"] == "unaligned_independent_time_basis"

    # Static photo remains temporally unknown
    assert events_by_filename["evidence_photo.jpg"]["alignment_status"] == "temporally_unknown"
    assert events_by_filename["evidence_photo.jpg"]["timestamp_normalized"] is None


# ===========================================================================
# 4. Cross-Modal Corroboration & Conflict Resolution Tests
# ===========================================================================

def test_cross_modal_weapon_corroboration():
    """Verify visual weapon detection and audio weapon mention create traceable corroboration with qualified findings."""
    service = FusionService()

    case_data = {
        "case_id": "CASE-WEAPON-CORROB",
        "title": "Weapon Corroboration Test",
        "evidence": [
            {
                "evidence_id": "EVD-01",
                "original_filename": "camera1.mp4",
                "media_type": "video",
                "analysis": {
                    "frame_observations": [
                        {
                            "frame_index": 25,
                            "timestamp_seconds": 10.0,
                            "detections": [{"label": "knife", "confidence": 0.88}],
                        }
                    ]
                },
            },
            {
                "evidence_id": "EVD-02",
                "original_filename": "camera1.wav",
                "media_type": "audio",
                "analysis": {
                    "timeline": {
                        "events": [
                            {
                                "event_id": "aevt_01",
                                "start_raw": 10.5,
                                "description": "Subject shouted: He has a knife!",
                                "supporting_quotes": ["He has a knife!"],
                                "confidence": 0.92,
                            }
                        ]
                    }
                },
            },
        ],
    }

    sync_groups = {"EVD-01": "cam1_sync", "EVD-02": "cam1_sync"}
    fusion = service.fuse_case(case_data, sync_groups=sync_groups)
    corrobs = [r for r in fusion["relationships"] if r["relationship_type"] == "corroboration"]
    assert len(corrobs) >= 1
    assert "weapon" in corrobs[0]["matching_rule"]
    # Explanations qualify that subject identity / physical possession remain unverified
    assert "unverified" in corrobs[0]["explanation"].lower() or "unconfirmed" in corrobs[0]["explanation"].lower()

    # Verify anchor highlighting distinguishes detection vs mention
    anchors = fusion["anchors"]
    anchor_categories = [a["category"] for a in anchors]
    assert "weapon" in anchor_categories


def test_cross_modal_conflict_surfacing_unarmed_claim_vs_detection():
    """
    Verify conflict resolution:
    Audio claims subject was unarmed, but synchronized video detected a weapon.
    Surfaces conflict when synchronized, but does not declare conflict when unaligned.
    """
    service = FusionService()

    case_data = {
        "case_id": "CASE-CONFLICT",
        "title": "Conflict Test",
        "evidence": [
            {
                "evidence_id": "EVD-VID",
                "original_filename": "cctv.mp4",
                "media_type": "video",
                "analysis": {
                    "frame_observations": [
                        {
                            "frame_index": 10,
                            "timestamp_seconds": 4.0,
                            "detections": [{"label": "gun", "confidence": 0.91}],
                        }
                    ]
                },
            },
            {
                "evidence_id": "EVD-AUD",
                "original_filename": "interrogation.wav",
                "media_type": "audio",
                "analysis": {
                    "transcription": {
                        "segments": [
                            {"id": 1, "start": 4.0, "end": 6.0, "text": "I swear I was completely unarmed, no gun!"}
                        ]
                    }
                },
            },
        ],
    }

    # 1. When unaligned, cannot declare direct conflict
    unaligned_fusion = service.fuse_case(case_data)
    unaligned_conflicts = [r for r in unaligned_fusion["relationships"] if r["relationship_type"] == "conflict"]
    assert len(unaligned_conflicts) == 0

    # 2. When synchronized on common timeline, surfaces conflict
    sync_groups = {"EVD-VID": "common_sync", "EVD-AUD": "common_sync"}
    fusion = service.fuse_case(case_data, sync_groups=sync_groups)

    conflicts = [r for r in fusion["relationships"] if r["relationship_type"] == "conflict"]
    assert len(conflicts) >= 1
    assert "unarmed_claim" in conflicts[0]["matching_rule"]

    # Both observations must be preserved
    assert len(fusion["events"]) >= 2

    # Certainty level must be forced to LOW due to active conflict
    for ev in fusion["events"]:
        if ev["conflict_links"]:
            assert ev["certainty_level"] == "low"
            assert ev["relationship_status"] == "conflicting"


def test_cross_modal_solitude_conflict():
    """
    Verify solitude conflict logic:
    - Speaker claims solitude/alone, and synchronized video shows person entry / multiple persons -> CONFLICT.
    - Speaker claims solitude/alone, and synchronized video shows exactly 1 person -> CONSISTENT (speaker is 1 person).
    """
    service = FusionService()

    # Case A: Entry transition occurs during solitude claim -> Conflict
    case_data_entry = {
        "case_id": "CASE-SOLITUDE-ENTRY",
        "title": "Solitude Conflict",
        "evidence": [
            {
                "evidence_id": "EVD-VID",
                "original_filename": "room.mp4",
                "media_type": "video",
                "analysis": {
                    "timeline": {
                        "events": [
                            {
                                "event_id": "vevt_1",
                                "start_raw": 8.0,
                                "event_type": "person_entered_frame",
                                "description": "Person entered room",
                            }
                        ]
                    }
                },
            },
            {
                "evidence_id": "EVD-AUD",
                "original_filename": "statement.wav",
                "media_type": "audio",
                "analysis": {
                    "transcription": {
                        "segments": [
                            {"id": 0, "start": 8.0, "end": 10.0, "text": "Nobody was there, I was alone."}
                        ]
                    }
                },
            },
        ],
    }

    sync_groups = {"EVD-VID": "room_sync", "EVD-AUD": "room_sync"}
    fusion_entry = service.fuse_case(case_data_entry, sync_groups=sync_groups)
    conflicts = [r for r in fusion_entry["relationships"] if r["relationship_type"] == "conflict"]
    assert len(conflicts) >= 1
    assert "solitude_claim" in conflicts[0]["matching_rule"]

    # Case B: Exactly 1 person detected while claiming solitude -> NO conflict (consistent with lone speaker)
    case_data_single = {
        "case_id": "CASE-SOLITUDE-SINGLE",
        "title": "Solitude Single Person",
        "evidence": [
            {
                "evidence_id": "EVD-VID",
                "original_filename": "room.mp4",
                "media_type": "video",
                "analysis": {
                    "timeline": {
                        "events": [
                            {
                                "event_id": "vevt_1",
                                "start_raw": 8.0,
                                "event_type": "visual_observation",
                                "description": "1 person present in room",
                                "person_count": 1,
                            }
                        ]
                    }
                },
            },
            {
                "evidence_id": "EVD-AUD",
                "original_filename": "statement.wav",
                "media_type": "audio",
                "analysis": {
                    "transcription": {
                        "segments": [
                            {"id": 0, "start": 8.0, "end": 10.0, "text": "Nobody was there, I was alone."}
                        ]
                    }
                },
            },
        ],
    }
    fusion_single = service.fuse_case(case_data_single, sync_groups=sync_groups)
    conflicts_single = [r for r in fusion_single["relationships"] if r["relationship_type"] == "conflict"]
    assert len(conflicts_single) == 0
    consistencies = [r for r in fusion_single["relationships"] if "solitude_claim_consistent_with_single_person" in r["matching_rule"]]
    assert len(consistencies) >= 1


# ===========================================================================
# 5. Certainty Scoring & Safeguards Tests
# ===========================================================================

def test_certainty_scorer_safeguards():
    """Verify certainty scoring rules and strict safeguards."""
    # 1. Single source with high confidence CANNOT be 'high' certainty
    single_event = ForensicEvent(
        event_id="fevt_01",
        case_id="C-1",
        evidence_id="E-1",
        original_filename="audio.wav",
        modality="audio",
        source_event_id="s1",
        event_type="speech_statement",
        source_confidence=0.99,
        alignment_status="aligned_common_timeline",
    )
    score, certainty, expl = CertaintyScorer.score_event(
        event=single_event,
        has_corroboration=False,
        has_conflict=False,
        corroborating_modalities=set(),
    )
    assert certainty != "high"  # Safeguard: single-source cannot be high!
    assert certainty == "medium"

    # 2. Corroborated event on common timeline CAN reach 'high' certainty
    corrob_event = ForensicEvent(
        event_id="fevt_02",
        case_id="C-1",
        evidence_id="E-1",
        original_filename="audio.wav",
        modality="audio",
        source_event_id="s2",
        event_type="weapon_mentioned",
        source_confidence=0.88,
        alignment_status="aligned_common_timeline",
    )
    score2, certainty2, expl2 = CertaintyScorer.score_event(
        event=corrob_event,
        has_corroboration=True,
        has_conflict=False,
        corroborating_modalities={"video"},
    )
    assert certainty2 == "high"
    assert score2 >= 0.75

    # 3. Conflicting event must be forced to 'low' certainty
    conflict_event = ForensicEvent(
        event_id="fevt_03",
        case_id="C-1",
        evidence_id="E-1",
        original_filename="audio.wav",
        modality="audio",
        source_event_id="s3",
        event_type="unarmed_claim",
        source_confidence=0.95,
        alignment_status="aligned_common_timeline",
    )
    score3, certainty3, expl3 = CertaintyScorer.score_event(
        event=conflict_event,
        has_corroboration=False,
        has_conflict=True,
        corroborating_modalities=set(),
    )
    assert certainty3 == "low"
    assert "contradiction" in expl3.lower()

    # 4. Missing confidence does not crash or invent fake confidence
    missing_conf_event = ForensicEvent(
        event_id="fevt_04",
        case_id="C-1",
        evidence_id="E-1",
        original_filename="audio.wav",
        modality="audio",
        source_event_id="s4",
        event_type="speech_statement",
        source_confidence=None,
        alignment_status="temporally_unknown",
    )
    score4, certainty4, expl4 = CertaintyScorer.score_event(
        event=missing_conf_event,
        has_corroboration=False,
        has_conflict=False,
        corroborating_modalities=set(),
    )
    assert 0.05 <= score4 <= 0.95
    assert "not defaulted" in expl4.lower()
    assert "unavailable/unrated" in expl4.lower()

    # Even with corroboration, missing source confidence bars high certainty
    corrob_missing = ForensicEvent(
        event_id="fevt_05",
        case_id="C-1",
        evidence_id="E-1",
        original_filename="audio.wav",
        modality="audio",
        source_event_id="s5",
        event_type="speech_statement",
        source_confidence=None,
        alignment_status="aligned_common_timeline",
    )
    score5, certainty5, expl5 = CertaintyScorer.score_event(
        event=corrob_missing,
        has_corroboration=True,
        has_conflict=False,
        corroborating_modalities={"video"},
    )
    assert certainty5 != "high"  # High certainty barred because source confidence is unrated
    assert "barred" in expl5.lower()


# ===========================================================================
# 6. JSON Serialization & Schema Conformity Tests
# ===========================================================================

def test_fusion_result_json_serialization_round_trip():
    """Verify that fusion output conforms to schema and survives JSON round-trip."""
    service = FusionService()
    case_data = {
        "case_id": "CASE-ROUND-TRIP",
        "title": "Round Trip Test Case",
        "evidence": [
            {
                "evidence_id": "EVD-01",
                "original_filename": "surveillance.mp4",
                "media_type": "video",
                "analysis": {
                    "timeline": {
                        "events": [
                            {"event_id": "vevt_1", "start_raw": 1.234, "description": "Person entered view"}
                        ]
                    }
                },
            }
        ],
    }

    fusion = service.fuse_case(case_data)
    serialized = json.dumps(fusion, ensure_ascii=False)
    deserialized = json.loads(serialized)

    assert deserialized["schema_version"] == "1.0.0"
    assert deserialized["case_id"] == "CASE-ROUND-TRIP"
    assert "summary" in deserialized
    assert "events" in deserialized
    assert "hash_index" in deserialized
    assert "grounded_narrative" in deserialized
    assert "forensic_disclaimer" in deserialized

    # Numeric precision check
    event = deserialized["events"][0]
    assert event["timestamp_normalized"] == 1.234
    assert event["timestamp_ms"] == 1234


# ===========================================================================
# 7. Persistence & Flask API Integration Tests
# ===========================================================================

def test_case_service_fusion_persistence(tmp_path: Path):
    """Verify SQLite persistence of fusion result via CaseService."""
    db_file = tmp_path / "fusion_test.db"
    case_service = CaseService(db_path=db_file)

    case = case_service.create_case(title="Fusion Persistence Test", case_id="CASE-FUSE-01")

    fusion_data = {
        "schema_version": "1.0.0",
        "case_id": "CASE-FUSE-01",
        "summary": {"total_events": 3},
        "events": [],
    }

    save_res = case_service.save_case_fusion("CASE-FUSE-01", fusion_data)
    assert save_res["status"] == "success"

    # Reload from separate service instance simulating restart
    case_service2 = CaseService(db_path=db_file)
    loaded_fusion = case_service2.get_case_fusion("CASE-FUSE-01")
    assert loaded_fusion is not None
    assert loaded_fusion["schema_version"] == "1.0.0"

    # get_case includes fusion
    reloaded_case = case_service2.get_case("CASE-FUSE-01")
    assert reloaded_case["fusion"] is not None
    assert reloaded_case["fusion"]["case_id"] == "CASE-FUSE-01"


def test_api_fusion_endpoints(tmp_path: Path):
    """Verify GET and POST /api/cases/<case_id>/fusion endpoints."""
    db_file = tmp_path / "api_test.db"
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    app = create_app({
        "TESTING": True,
        "DATABASE_PATH": str(db_file),
        "UPLOAD_FOLDER": str(uploads_dir),
    })

    client = app.test_client()

    # 1. Create a case
    create_res = client.post("/api/cases", json={"title": "API Fusion Test", "case_id": "CASE-API-FUSE"})
    assert create_res.status_code == 201

    # 2. GET fusion triggers computation and returns 200
    get_res = client.get("/api/cases/CASE-API-FUSE/fusion")
    assert get_res.status_code == 200
    data = get_res.get_json()
    assert data["status"] == "success"
    assert data["case_id"] == "CASE-API-FUSE"
    assert "fusion" in data
    assert data["fusion"]["schema_version"] == "1.0.0"

    # 3. POST fusion re-runs and returns 200
    post_res = client.post(
        "/api/cases/CASE-API-FUSE/fusion",
        json={"sync_groups": {"EVD-1": "common"}},
    )
    assert post_res.status_code == 200
    post_data = post_res.get_json()
    assert post_data["status"] == "success"
    assert post_data["fusion"]["case_id"] == "CASE-API-FUSE"
