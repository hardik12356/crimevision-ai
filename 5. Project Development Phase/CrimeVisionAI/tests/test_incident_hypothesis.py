"""
Tests for Incident Hypothesis Engine in CrimeVisionAI.

Verifies:
1. UI placement directly below the Primary Inspection Canvas and presence of all 6 required components.
2. Empty case / insufficient evidence handling (no fabricated hypotheses).
3. Evidence-grounded candidate incident generation with provenance (filename, modality, timestamp).
4. Witness allegation distinguished from visual observation (no false claim of video confirmation).
5. Cross-modal conflicting findings preserved in contradictions.
6. Independent time bases flagged without false synchronization claims.
7. Case switching isolation (case A hypothesis does not leak to case B).
8. API endpoints (GET/POST /api/cases/<case_id>/hypothesis).
9. Graceful fallback on LLM failure / rate limit.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app import create_app
from app.services.case_service import CaseService
from app.services.fusion_service import FusionService
from app.services.groq_service import GroqService
from app.services.hypothesis_service import HypothesisService
from app.services.token_budget import GroqRateLimitExceeded


@pytest.fixture
def mock_groq_service():
    groq = MagicMock(spec=GroqService)
    groq.classify_query_intent = GroqService.classify_query_intent
    groq.generate_hypotheses.return_value = {
        "model": "openai/gpt-oss-20b",
        "primary_incident_type": "Suspected Jewellery Theft",
        "hypotheses": [
            {
                "title": "Suspected Jewellery Theft",
                "narrative": "Witness statements allege a theft from the counter. Surveillance footage confirms human presence but does not show item removal.",
                "supporting_evidence": [
                    "[Audio] witness_statement.wav @ 00:02: Witness stated someone took a necklace",
                    "[Video] store_cam.mp4 @ 1.25s: RT-DETR detected person near counter",
                ],
                "contradicting_evidence": [
                    "Surveillance video does not independently confirm physical item removal."
                ],
                "assumptions": ["Individual in frame is associated with the reported interaction."],
                "uncertainty_factors": [
                    "Camera angle does not capture counter surface clearly.",
                    "Recordings operate on independent time bases.",
                ],
                "plausibility": "Moderate",
            },
            {
                "title": "Routine Store Interaction",
                "narrative": "Individuals engaged in authorized browsing without criminal removal of items.",
                "supporting_evidence": ["[Video] store_cam.mp4: Normal movement observed"],
                "contradicting_evidence": ["[Audio] witness_statement.wav: Witness allegations"],
                "assumptions": ["Customer departed without merchandise."],
                "uncertainty_factors": ["Unrecorded transaction details."],
                "plausibility": "Limited",
            },
        ],
        "contradictory_evidence": [
            "Witness reports theft, but video surveillance frames do not show item concealment."
        ],
        "evidence_gaps": [
            "Recordings operate on independent, unsynchronized time bases.",
            "Counter surface obscured from view.",
        ],
        "overall_assessment": "Witness claims a theft occurred; however, video evidence confirms presence without independent verification of theft.",
        "evidence_sufficiency": "limited",
        "overall_limitations": "Unsynchronized sources and partial visual coverage.",
        "note": "AI-generated hypotheses grounded in recorded case evidence.",
        "truncated": False,
    }
    groq.answer_case_question.return_value = {
        "model": "openai/gpt-oss-20b",
        "answer": "Grounded answer to question.",
        "note": "AI-generated response.",
    }
    return groq


@pytest.fixture
def app(tmp_path, mock_groq_service):
    db_path = tmp_path / "test_hypothesis.db"
    case_service = CaseService(db_path=db_path)
    fusion_service = FusionService()
    hypothesis_service = HypothesisService(
        case_service=case_service,
        fusion_service=fusion_service,
        groq_service=mock_groq_service,
    )

    app_instance = create_app({
        "TESTING": True,
        "DATABASE_PATH": str(db_path),
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "CASE_SERVICE": case_service,
        "FUSION_SERVICE": fusion_service,
        "HYPOTHESIS_SERVICE": hypothesis_service,
        "GROQ_SERVICE": mock_groq_service,
    })
    return app_instance


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# 1. UI Placement and Layout Verification
# ---------------------------------------------------------------------------
def test_ui_placement_below_primary_inspection_canvas(client):
    """Verify incident-hypothesis-panel is located directly below media-viewer-panel in index.html."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # 1. Verify existence of Primary Inspection Canvas and Incident Hypothesis Panel
    assert "media-stage-canvas" in html
    assert "Primary Inspection Canvas" in html
    assert "incident-hypothesis-panel" in html
    assert "Incident Hypothesis" in html

    # 2. Verify placement order: media-viewer-panel comes before incident-hypothesis-panel
    canvas_pos = html.find('id="media-stage-canvas"')
    hyp_pos = html.find('id="incident-hypothesis-panel"')
    video_res_pos = html.find('id="video-results-panel"')

    assert canvas_pos != -1
    assert hyp_pos != -1
    assert canvas_pos < hyp_pos, "Incident Hypothesis section must be placed after Primary Inspection Canvas"
    assert hyp_pos < video_res_pos, "Incident Hypothesis section should precede single-modality findings"

    # 3. Verify all 6 required content sections exist in HTML
    assert "hyp-primary-type" in html, "Primary candidate incident type element missing"
    assert "hyp-plausibility-badge" in html, "Plausibility badge element missing"
    assert "hyp-alternatives-list" in html, "Alternative hypotheses list element missing"
    assert "hyp-supporting-list" in html, "Supporting evidence list element missing"
    assert "hyp-contradictions-list" in html, "Contradictions list element missing"
    assert "hyp-gaps-list" in html, "Evidence gaps element missing"
    assert "hyp-overall-assessment" in html, "Overall assessment element missing"

    # 4. Verify states exist
    assert "hypothesis-loading" in html
    assert "hypothesis-empty" in html
    assert "hypothesis-error" in html
    assert "btn-refresh-hypothesis" in html


# ---------------------------------------------------------------------------
# 2. Empty Case / Insufficient Evidence (No Fabrication)
# ---------------------------------------------------------------------------
def test_empty_case_returns_undetermined_without_fabrication(client):
    """Empty cases with no evidence must return Undetermined / Insufficient Evidence without calling LLM."""
    case_res = client.post("/api/cases", json={"title": "Empty Cold Case"})
    case_id = case_res.get_json()["case"]["case_id"]

    res = client.get(f"/api/cases/{case_id}/hypothesis")
    assert res.status_code == 200
    data = res.get_json()["hypothesis"]

    assert data["status"] == "insufficient_evidence"
    assert "Undetermined" in data["incident_type"] or "Insufficient" in data["incident_type"]
    assert data["plausibility"] == "Undetermined"
    assert len(data["supporting_evidence"]) == 0
    assert len(data["alternative_hypotheses"]) == 0
    assert "No analyzed evidence" in data["narrative"]
    assert data["evidence_sufficiency"] == "insufficient"


# ---------------------------------------------------------------------------
# 3. Sufficient Supporting Evidence with Citations & Provenance
# ---------------------------------------------------------------------------
def test_sufficient_evidence_generates_grounded_hypothesis(client, mock_groq_service):
    """A case with analyzed evidence calls Groq and returns evidence-grounded hypothesis with provenance."""
    case_res = client.post("/api/cases", json={"title": "Store Burglary Investigation"})
    case_id = case_res.get_json()["case"]["case_id"]
    cs: CaseService = client.application.config["CASE_SERVICE"]

    # Add and analyze video evidence
    ev = cs.add_evidence(case_id, "store_cam.mp4", "video", "uploads/store_cam.mp4", 1024)
    ev_id = ev["evidence_id"]
    cs.save_analysis_result(case_id, ev_id, "video", {
        "duration_seconds": 8.0,
        "sampled_fps": 4.0,
        "frame_observations": [
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "persons": 1,
                "detections": [{"label": "person", "confidence": 0.94}],
            },
            {
                "frame_index": 5,
                "timestamp_seconds": 1.25,
                "persons": 1,
                "detections": [{"label": "person", "confidence": 0.92}, {"label": "knife", "confidence": 0.78}],
            },
        ],
        "incident_hypothesis": {
            "type": "unauthorized_entry",
            "reasoning": "Person detected in frame with knife.",
        },
    })

    res = client.post(f"/api/cases/{case_id}/hypothesis")
    assert res.status_code == 200
    hyp = res.get_json()["hypothesis"]

    assert hyp["status"] == "completed"
    assert hyp["incident_type"] == "Suspected Jewellery Theft"
    assert hyp["plausibility"] == "Moderate"
    assert len(hyp["supporting_evidence"]) >= 1
    assert any("store_cam.mp4" in s for s in hyp["supporting_evidence"])
    assert len(hyp["alternative_hypotheses"]) >= 1
    assert "Routine Store Interaction" in hyp["alternative_hypotheses"][0]["title"]
    assert len(hyp["evidence_gaps"]) >= 1
    assert "independent" in hyp["evidence_gaps"][0].lower() or "camera" in hyp["evidence_gaps"][0].lower()


# ---------------------------------------------------------------------------
# 4. Witness Allegation Distinguished from Visual Observation
# ---------------------------------------------------------------------------
def test_witness_allegation_distinguished_from_visual_confirmation(app):
    """When a witness alleges theft but video shows no concealment, distinction is explicitly preserved."""
    cs: CaseService = app.config["CASE_SERVICE"]
    fs: FusionService = app.config["FUSION_SERVICE"]
    # Test heuristic fallback directly (offline/deterministic mode)
    hs = HypothesisService(case_service=cs, fusion_service=fs, groq_service=None)

    case_data = cs.create_case("Theft Allegation Case", "Investigating reported necklace snatching.")
    case_id = case_data["case_id"]

    # Audio with witness allegation
    ev_audio = cs.add_evidence(case_id, "witness_call.wav", "audio", "uploads/witness.wav", 500)
    cs.save_analysis_result(case_id, ev_audio["evidence_id"], "audio", {
        "transcription": {
            "transcript": "A man snatched a gold necklace from the counter and ran out the front door!",
        },
    })

    # Video showing person but no theft action
    ev_video = cs.add_evidence(case_id, "counter_cctv.mp4", "video", "uploads/cctv.mp4", 2000)
    cs.save_analysis_result(case_id, ev_video["evidence_id"], "video", {
        "duration_seconds": 6.0,
        "sampled_fps": 4.0,
        "frame_observations": [
            {"timestamp_seconds": 0.0, "persons": 1, "detections": [{"label": "person", "confidence": 0.91}]},
            {"timestamp_seconds": 2.0, "persons": 1, "detections": [{"label": "person", "confidence": 0.88}]},
        ],
        "incident_hypothesis": {
            "type": "routine_presence",
            "reasoning": "Person observed in lobby area.",
        },
    })

    full_case = cs.get_case(case_id)
    hyp = hs.generate_case_hypothesis(full_case, force_refresh=True)

    assert hyp["status"] == "completed"
    # Verify primary candidate reflects witness-reported status
    assert "Witness-Reported" in hyp["incident_type"] or "Suspected" in hyp["incident_type"]
    assert hyp["plausibility"] in ("Moderate", "Limited")

    # Verify contradiction explicitly identifies video lacks independent confirmation of theft
    contra_text = " ".join(hyp["contradictions"]).lower()
    assert "witness" in contra_text or "surveillance" in contra_text
    assert "removal" in contra_text or "not independently" in contra_text or "theft" in contra_text

    # Verify alternative hypothesis considers routine presence
    assert any("Routine" in alt["title"] for alt in hyp["alternative_hypotheses"])


# ---------------------------------------------------------------------------
# 5. Cross-Modal Conflicting Findings Preserved in Contradictions
# ---------------------------------------------------------------------------
def test_conflicting_audio_and_video_findings_preserved(app):
    """Audio witness claiming suspect unarmed while detector finds knife produces explicit conflict."""
    cs: CaseService = app.config["CASE_SERVICE"]
    fs: FusionService = app.config["FUSION_SERVICE"]
    hs = HypothesisService(case_service=cs, fusion_service=fs, groq_service=None)

    case_data = cs.create_case("Weapon Conflict Case")
    case_id = case_data["case_id"]

    ev_audio = cs.add_evidence(case_id, "statement.wav", "audio", "uploads/s.wav", 100)
    cs.save_analysis_result(case_id, ev_audio["evidence_id"], "audio", {
        "transcription": {"transcript": "The customer was unarmed and had completely empty hands."},
    })

    ev_video = cs.add_evidence(case_id, "cam1.mp4", "video", "uploads/c1.mp4", 500)
    cs.save_analysis_result(case_id, ev_video["evidence_id"], "video", {
        "duration_seconds": 4.0,
        "sampled_fps": 4.0,
        "frame_observations": [
            {"timestamp_seconds": 1.0, "persons": 1, "detections": [{"label": "knife", "confidence": 0.85}]},
        ],
    })

    full_case = cs.get_case(case_id)
    hyp = hs.generate_case_hypothesis(full_case, force_refresh=True)

    # Cross-modal conflict must appear in contradictions
    assert len(hyp["contradictions"]) >= 1
    contra_str = " ".join(hyp["contradictions"]).lower()
    assert "conflict" in contra_str or "unarmed" in contra_str or "knife" in contra_str


# ---------------------------------------------------------------------------
# 6. Independent Time Bases Flagged (No False Synchronization)
# ---------------------------------------------------------------------------
def test_independent_time_bases_flags_unsynchronized_gap(app):
    """Separate unsynchronized recordings must not claim synchronized timestamps."""
    cs: CaseService = app.config["CASE_SERVICE"]
    fs: FusionService = app.config["FUSION_SERVICE"]
    hs = HypothesisService(case_service=cs, fusion_service=fs, groq_service=None)

    case_data = cs.create_case("Unsynced Case")
    case_id = case_data["case_id"]

    ev1 = cs.add_evidence(case_id, "cam_north.mp4", "video", "uploads/n.mp4", 100)
    cs.save_analysis_result(case_id, ev1["evidence_id"], "video", {
        "duration_seconds": 5.0,
        "frame_observations": [{"timestamp_seconds": 0.5, "persons": 1, "detections": []}],
    })

    ev2 = cs.add_evidence(case_id, "cam_south.mp4", "video", "uploads/s.mp4", 100)
    cs.save_analysis_result(case_id, ev2["evidence_id"], "video", {
        "duration_seconds": 5.0,
        "frame_observations": [{"timestamp_seconds": 0.5, "persons": 1, "detections": []}],
    })

    full_case = cs.get_case(case_id)
    hyp = hs.generate_case_hypothesis(full_case, force_refresh=True)

    gaps_str = " ".join(hyp["evidence_gaps"]).lower()
    assert "independent" in gaps_str or "unsynchronized" in gaps_str


# ---------------------------------------------------------------------------
# 7. Case Switching Isolation
# ---------------------------------------------------------------------------
def test_case_switching_isolation(client):
    """Case A hypothesis does not leak into Case B."""
    cs: CaseService = client.application.config["CASE_SERVICE"]

    case_a = cs.create_case("Case Alpha")
    case_b = cs.create_case("Case Beta")

    # Save distinct hypothesis for Case A
    hyp_a = {
        "case_id": case_a["case_id"],
        "incident_type": "Case A Exclusive Hypothesis",
        "status": "completed",
        "plausibility": "Moderate",
        "narrative": "Narrative for Alpha.",
        "primary_hypothesis": {"title": "Alpha Candidate", "plausibility": "Moderate"},
        "alternative_hypotheses": [],
        "supporting_evidence": ["Evidence Alpha"],
        "contradictions": [],
        "evidence_gaps": [],
        "overall_assessment": "Alpha assessment.",
    }
    cs.save_case_hypothesis(case_a["case_id"], hyp_a)

    # Fetch Case A
    res_a = client.get(f"/api/cases/{case_a['case_id']}/hypothesis")
    assert res_a.status_code == 200
    assert res_a.get_json()["hypothesis"]["incident_type"] == "Case A Exclusive Hypothesis"

    # Fetch Case B (which is empty)
    res_b = client.get(f"/api/cases/{case_b['case_id']}/hypothesis")
    assert res_b.status_code == 200
    hyp_b_data = res_b.get_json()["hypothesis"]
    assert hyp_b_data["incident_type"] != "Case A Exclusive Hypothesis"
    assert hyp_b_data["status"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# 8. Graceful Fallback on LLM Failure / Rate Limit
# ---------------------------------------------------------------------------
def test_rate_limit_fallback_to_heuristic_without_500(client, mock_groq_service):
    """When Groq raises GroqRateLimitExceeded, the engine falls back to heuristic synthesis."""
    mock_groq_service.generate_hypotheses.side_effect = GroqRateLimitExceeded("Rate limit reached")

    cs: CaseService = client.application.config["CASE_SERVICE"]
    case_res = client.post("/api/cases", json={"title": "Rate Limit Fallback Case"})
    case_id = case_res.get_json()["case"]["case_id"]

    ev = cs.add_evidence(case_id, "video.mp4", "video", "uploads/v.mp4", 100)
    cs.save_analysis_result(case_id, ev["evidence_id"], "video", {
        "duration_seconds": 5.0,
        "frame_observations": [{"timestamp_seconds": 1.0, "persons": 1, "detections": []}],
        "incident_hypothesis": {"type": "routine_entry", "reasoning": "Person entered."},
    })

    res = client.post(f"/api/cases/{case_id}/hypothesis")
    assert res.status_code == 200
    hyp = res.get_json()["hypothesis"]
    assert hyp["status"] == "completed"
    assert hyp["model"] == "heuristic_synthesizer"
    assert "Routine Entry" in hyp["incident_type"]
