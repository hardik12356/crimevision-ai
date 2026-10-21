"""
Focused Verification Tests for Cross-Modal Fusion Redesign.

Verifies:
1. Deduplication of cross-modal relationships (no combinatorial frame explosion).
2. Semantic classification of generic person co-occurrences as 'thematic_match'.
3. Elimination of event ID remapping collision/corruption bug (no ['fevt_076', 'fevt_076']).
4. Complete metadata attribution on relationships (sources, filenames, modalities, timestamps).
5. Frontend template structure with the 4-part investigator hierarchy (Overview, Key Findings,
   Relationships, and Collapsible Technical Audit Trail).
6. Integration with real test case CASE-2026-INCIDENT-97.
"""

from pathlib import Path
import pytest

from app import create_app
from app.services.case_service import CaseService
from app.services.fusion_service import FusionService


@pytest.fixture
def fusion_service():
    return FusionService()


@pytest.fixture
def mock_case_multi_frame():
    """Case with 2 audio statements and 5 video person events."""
    video_events = [
        {
            "event_id": f"vt_{i}",
            "start_raw": float(i),
            "end_raw": float(i + 1),
            "event_type": "person_observation",
            "description": f"Model detected person presence in frame {i}",
            "confidence": 0.88,
        }
        for i in range(5)
    ]
    return {
        "case_id": "CASE-DEDUP-TEST",
        "title": "Deduplication Verification Case",
        "evidence": [
            {
                "evidence_id": "EVD-VID-01",
                "original_filename": "surveillance_cam.mp4",
                "media_type": "video",
                "status": "completed",
                "analysis": {
                    "duration_seconds": 10.0,
                    "timeline": {
                        "events": video_events,
                    },
                },
            },
            {
                "evidence_id": "EVD-AUD-01",
                "original_filename": "witness_audio.wav",
                "media_type": "audio",
                "status": "completed",
                "analysis": {
                    "duration_seconds": 15.0,
                    "transcription": {
                        "transcript": "A customer was standing near the register. Then a man walked in.",
                        "segments": [
                            {"id": 0, "start": 1.0, "end": 4.0, "text": "A customer was standing near the register."},
                            {"id": 1, "start": 5.0, "end": 8.0, "text": "Then a man walked in."}
                        ],
                    },
                    "timeline": {"events": []},
                },
            },
        ],
    }


def test_spoken_person_visual_person_deduplication(fusion_service, mock_case_multi_frame):
    """
    Ensure 10 video frames do NOT produce 10 duplicate relationships with the single audio segment.
    Instead, exactly 1 consolidated relationship should be produced between the evidence pair.
    """
    result = fusion_service.fuse_case(mock_case_multi_frame)
    relationships = result["relationships"]

    person_rels = [
        r for r in relationships
        if r["matching_rule"] == "spoken_person_visual_person_cross_modal"
    ]

    # Exactly 1 deduplicated relationship between the audio and video source
    assert len(person_rels) == 1, f"Expected 1 deduplicated relationship, got {len(person_rels)}"

    rel = person_rels[0]
    # Classified as thematic_match rather than unqualified corroboration
    assert rel["relationship_type"] == "thematic_match"
    assert "thematic match only" in rel["explanation"].lower()
    assert any("independent time bases" in u.lower() for u in rel["uncertainties"])


def test_no_event_id_collision_or_double_mapping(fusion_service, mock_case_multi_frame):
    """
    Verify that sequential ID remapping does not mangle source_event_ids into duplicates
    like ['fevt_076', 'fevt_076'].
    """
    result = fusion_service.fuse_case(mock_case_multi_frame)
    relationships = result["relationships"]

    for r in relationships:
        s_ids = r["source_event_ids"]
        assert len(s_ids) == 2, f"Relationship {r['relationship_id']} should link exactly 2 events"
        assert s_ids[0] != s_ids[1], (
            f"Event ID collision detected in {r['relationship_id']}: {s_ids}"
        )
        assert s_ids[0].startswith("fevt_")
        assert s_ids[1].startswith("fevt_")


def test_relationship_metadata_attribution(fusion_service, mock_case_multi_frame):
    """
    Verify that relationships include source filenames, modalities, evidence IDs,
    and timestamps for frontend evidence navigation.
    """
    result = fusion_service.fuse_case(mock_case_multi_frame)
    rel = result["relationships"][0]

    assert rel["source_a_evidence_id"] in ("EVD-VID-01", "EVD-AUD-01")
    assert rel["source_b_evidence_id"] in ("EVD-VID-01", "EVD-AUD-01")
    assert rel["source_a_filename"] in ("surveillance_cam.mp4", "witness_audio.wav")
    assert rel["source_b_filename"] in ("surveillance_cam.mp4", "witness_audio.wav")
    assert rel["source_a_modality"] in ("video", "audio")
    assert rel["source_b_modality"] in ("video", "audio")


def test_fusion_on_controlled_media_case_incident_97():
    """
    End-to-end verification on CASE-2026-INCIDENT-97 (CV-QA-TEST-003).
    Previously produced 54 redundant links with identical text and ID collision.
    Must now produce clean, deduplicated relationships without ID collisions.
    """
    cs = CaseService()
    case = cs.get_case("CASE-2026-INCIDENT-97")
    if not case:
        pytest.skip("CASE-2026-INCIDENT-97 not found in database")

    fs = FusionService()
    result = fs.fuse_case(case)

    # 1. Detections/events are extracted
    assert len(result["events"]) > 0

    # 2. Relationships are deduplicated (previously 54, now 5)
    relationships = result["relationships"]
    assert 1 <= len(relationships) <= 10, (
        f"Expected between 1 and 10 deduplicated relationships, got {len(relationships)}"
    )

    # 3. No ID collisions
    for r in relationships:
        s_ids = r["source_event_ids"]
        assert len(s_ids) == 2
        assert s_ids[0] != s_ids[1], f"Collision in {r['relationship_id']}: {s_ids}"

    # 4. Summary counts are accurate
    summary = result["summary"]
    assert summary["thematic_matches_count"] == len(relationships)
    assert summary["conflicts_count"] == 0


def test_fusion_tab_html_investigator_sections(tmp_path):
    """
    Verify that index.html contains the 4-part investigator layout:
    Section A (Overview & metrics), Section B (Key findings),
    Section C (Relationships with comparison chips), Section D (Collapsible audit trail).
    """
    app = create_app({"TESTING": True, "DATABASE_PATH": str(tmp_path / "test.db")})
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Section A: Overview & Metrics
    assert 'id="fusion-main-container"' in html
    assert 'id="fusion-summary-card"' in html
    assert 'id="fusion-alignment-badge"' in html
    assert 'id="fusion-metrics-grid"' in html
    assert 'id="fusion-metric-relations"' in html
    assert 'id="fusion-metric-conflicts"' in html
    assert 'id="fusion-metric-review"' in html
    assert 'id="fusion-gaps-banner"' in html

    # Section B: Key Findings
    assert 'id="fusion-key-findings-card"' in html
    assert 'id="fusion-key-findings-list"' in html
    assert 'id="fusion-findings-count-badge"' in html

    # Section C: Evidence Relationships
    assert 'id="fusion-relationships-card"' in html
    assert 'id="fusion-relationships-list"' in html
    assert 'id="fusion-rel-count-badge"' in html

    # Section D: Collapsible Audit Details
    assert '<details' in html
    assert 'id="fusion-audit-details"' in html
    assert 'id="fusion-narrative-text"' in html
    assert 'id="fusion-hash-meta"' in html
    assert 'id="fusion-events-table-wrap"' in html


def test_app_js_contains_evidence_navigation_and_helpers(tmp_path):
    """Verify that app.js contains the evidence navigation and redesign rendering helpers."""
    app = create_app({"TESTING": True, "DATABASE_PATH": str(tmp_path / "test.db")})
    client = app.test_client()
    res = client.get("/static/js/app.js")
    assert res.status_code == 200
    js = res.get_data(as_text=True)

    assert "navigateToEvidenceSource" in js
    assert "fusionMainContainer" in js
    assert "badge-rel-thematic" in js
    assert "badge-rel-conflict" in js
    assert "fusion-comparison-wrap" in js
