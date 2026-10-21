"""
Incident Hypothesis Service for CrimeVisionAI.

Synthesizes multimodal case evidence (video observations, audio transcripts,
image detections, cross-modal timeline events, and fusion relationships)
to generate evidence-grounded candidate incident types, supporting evidence,
contradictions, limitations, and overall investigative assessments.

Strict Forensic Principles:
1. Distinguish direct observations, witness claims, model interpretations, and unverified hypotheses.
2. Do not infer theft merely because a person and a bag appear in the same scene.
3. Do not infer a weapon's use from detection alone.
4. Do not treat generic matches across modalities as proof of the same person or event.
5. Do not treat source-relative timestamps from separate recordings as synchronized.
6. Preserve contradictions instead of forcing all evidence into one narrative.
7. If evidence is insufficient, return Undetermined / Insufficient Evidence.
8. Qualitative assessments: 'Limited', 'Moderate', 'Strongly Supported'.
"""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any

from app.services.case_service import CaseService
from app.services.context_builder import ContextBuilder
from app.services.fusion_service import FusionService
from app.services.groq_service import GroqService
from app.services.token_budget import GroqRateLimitExceeded

logger = logging.getLogger(__name__)

FORENSIC_DISCLAIMER = (
    "Hypotheses are probabilistic investigative decision-support aids. "
    "They do not constitute scientific proof or legal conclusions. "
    "Independent human review of original recordings and imagery is strictly required."
)

PLAUSIBILITY_MAP = {
    "high": "Strongly Supported",
    "strongly supported": "Strongly Supported",
    "strong": "Strongly Supported",
    "moderate": "Moderate",
    "medium": "Moderate",
    "low": "Limited",
    "limited": "Limited",
    "undetermined": "Undetermined",
    "insufficient": "Undetermined",
}


def normalize_plausibility(raw_val: str | None) -> str:
    if not raw_val:
        return "Limited"
    clean = str(raw_val).strip().lower()
    return PLAUSIBILITY_MAP.get(clean, "Limited")


class HypothesisService:
    """Service for formulating and persisting evidence-grounded incident hypotheses."""

    def __init__(
        self,
        case_service: CaseService | None = None,
        fusion_service: FusionService | None = None,
        groq_service: Any = "default",
    ):
        self.case_service = case_service or CaseService()
        self.fusion_service = fusion_service or FusionService()
        self._groq_service = groq_service

    def _get_groq_service(self) -> GroqService | None:
        if self._groq_service is None or self._groq_service is False:
            return None
        if self._groq_service != "default":
            return self._groq_service
        try:
            return GroqService()
        except Exception as err:
            logger.warning(f"[HypothesisService] GroqService unavailable: {err}")
            return None

    def generate_case_hypothesis(
        self,
        case_data: dict[str, Any],
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Synthesize available case evidence and generate evidence-grounded candidate incident types.

        Parameters
        ----------
        case_data : dict[str, Any]
            Full case record including evidence items, timeline events, and fusion.
        force_refresh : bool
            If True, regenerates the hypothesis even if a cached result exists.

        Returns
        -------
        dict[str, Any]
            Structured incident hypothesis dictionary with primary candidate, alternatives,
            supporting evidence, contradictions, limitations, and overall assessment.
        """
        case_id = case_data.get("case_id", "CASE-UNKNOWN")

        # 1. Check persistence cache if not forcing refresh
        if not force_refresh:
            cached = self.case_service.get_case_hypothesis(case_id)
            if cached:
                return cached

        # 2. Check for analyzed evidence items
        evidence_items = case_data.get("evidence", [])
        analyzed_items = [
            e for e in evidence_items
            if e.get("status") == "completed" and e.get("analysis")
        ]

        now_iso = datetime.now(timezone.utc).isoformat()

        # 3. Empty or unanalyzed case handling (Never fabricate hypotheses)
        if not analyzed_items:
            empty_result = {
                "case_id": case_id,
                "incident_type": "Undetermined / Insufficient Evidence",
                "status": "insufficient_evidence",
                "plausibility": "Undetermined",
                "narrative": "No analyzed evidence is currently recorded in this case locker to formulate hypotheses.",
                "primary_hypothesis": {
                    "title": "Undetermined / Insufficient Evidence",
                    "incident_type": "Undetermined / Insufficient Evidence",
                    "narrative": "No analyzed evidence is currently available to substantiate any candidate incident hypotheses.",
                    "plausibility": "Undetermined",
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "assumptions": [],
                    "uncertainty_factors": ["No analyzed video, audio, or image evidence available in case."],
                },
                "alternative_hypotheses": [],
                "supporting_evidence": [],
                "contradictions": [],
                "evidence_gaps": [
                    "No analyzed recordings or imagery available in case evidence locker."
                ],
                "overall_assessment": "Cannot formulate forensic incident hypotheses without analyzed evidence. Ingest and analyze evidence files first.",
                "overall_limitations": "Zero analyzed evidence items in case.",
                "evidence_sufficiency": "insufficient",
                "generated_at": now_iso,
                "model": "rule_based_filter",
                "forensic_disclaimer": FORENSIC_DISCLAIMER,
            }
            self.case_service.save_case_hypothesis(case_id, empty_result)
            return empty_result

        # 4. Ensure multimodal fusion data exists for the case
        fusion_data = case_data.get("fusion")
        if not fusion_data:
            try:
                fusion_data = self.fusion_service.fuse_case(case_data)
                self.case_service.save_case_fusion(case_id, fusion_data)
                case_data["fusion"] = fusion_data
            except Exception as f_err:
                logger.warning(f"[HypothesisService] Cross-modal fusion error: {f_err}")
                fusion_data = {}

        # 5. Attempt LLM-grounded hypothesis generation
        groq = self._get_groq_service()
        llm_result: dict[str, Any] | None = None

        if groq is not None:
            try:
                case_context = ContextBuilder.build_compact_context(
                    case_data,
                    question="Synthesize case evidence into candidate incident hypotheses",
                    max_input_tokens=3500,
                )
                llm_result = groq.generate_hypotheses(
                    case_context=case_context,
                    question="Synthesize case evidence into candidate incident hypotheses",
                )
            except GroqRateLimitExceeded as rle:
                logger.warning(f"[HypothesisService] Groq rate limit exceeded: {rle}")
                llm_result = None
            except Exception as groq_err:
                logger.warning(f"[HypothesisService] Groq hypothesis generation failed: {groq_err}")
                llm_result = None

        # 6. Normalize LLM response or fall back to deterministic grounded heuristic
        if llm_result and (llm_result.get("hypotheses") or llm_result.get("primary_incident_type")):
            final_hypothesis = self._normalize_llm_response(case_id, case_data, fusion_data, llm_result, now_iso)
        else:
            final_hypothesis = self._generate_heuristic_hypothesis(case_id, case_data, fusion_data, now_iso)

        # 7. Persist hypothesis
        self.case_service.save_case_hypothesis(case_id, final_hypothesis)

        # 8. Sync with fusion hypotheses for PDF report consistency
        if fusion_data and isinstance(fusion_data, dict):
            hyp_list = [final_hypothesis["primary_hypothesis"]] + final_hypothesis["alternative_hypotheses"]
            fusion_data["hypotheses"] = hyp_list
            try:
                self.case_service.save_case_fusion(case_id, fusion_data)
            except Exception:
                pass

        return final_hypothesis

    def _normalize_llm_response(
        self,
        case_id: str,
        case_data: dict[str, Any],
        fusion_data: dict[str, Any],
        llm_result: dict[str, Any],
        now_iso: str,
    ) -> dict[str, Any]:
        """Normalize Groq JSON hypotheses into standardized 6-section structure."""
        raw_hypotheses = llm_result.get("hypotheses", [])

        if not raw_hypotheses:
            # Undetermined / Insufficient evidence returned by LLM
            return {
                "case_id": case_id,
                "incident_type": "Undetermined / Insufficient Evidence",
                "status": "insufficient_evidence",
                "plausibility": "Undetermined",
                "narrative": llm_result.get("overall_limitations") or "Available evidence is insufficient to formulate grounded hypotheses.",
                "primary_hypothesis": {
                    "title": "Undetermined / Insufficient Evidence",
                    "incident_type": "Undetermined / Insufficient Evidence",
                    "narrative": llm_result.get("overall_limitations") or "Insufficient evidence.",
                    "plausibility": "Undetermined",
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "assumptions": [],
                    "uncertainty_factors": [llm_result.get("overall_limitations") or "Evidence gaps prevent classification."],
                },
                "alternative_hypotheses": [],
                "supporting_evidence": [],
                "contradictions": llm_result.get("contradictory_evidence", []),
                "evidence_gaps": llm_result.get("evidence_gaps") or [llm_result.get("overall_limitations", "Insufficient evidence")],
                "overall_assessment": llm_result.get("overall_assessment") or "Evidence insufficient to establish incident type.",
                "overall_limitations": llm_result.get("overall_limitations", ""),
                "evidence_sufficiency": "insufficient",
                "generated_at": now_iso,
                "model": llm_result.get("model", "openai/gpt-oss-20b"),
                "forensic_disclaimer": FORENSIC_DISCLAIMER,
            }

        primary_raw = raw_hypotheses[0]
        primary_title = primary_raw.get("title", "Candidate Incident")
        primary_type = (
            llm_result.get("primary_incident_type")
            or primary_raw.get("incident_type")
            or primary_title
        )

        primary_plausibility = normalize_plausibility(primary_raw.get("plausibility"))

        primary_hypothesis = {
            "title": primary_title,
            "incident_type": primary_type,
            "narrative": primary_raw.get("narrative", ""),
            "plausibility": primary_plausibility,
            "supporting_evidence": primary_raw.get("supporting_evidence", []),
            "contradicting_evidence": primary_raw.get("contradicting_evidence", []),
            "assumptions": primary_raw.get("assumptions", []),
            "uncertainty_factors": primary_raw.get("uncertainty_factors", []),
        }

        alt_hypotheses = []
        for alt_raw in raw_hypotheses[1:]:
            alt_plaus = normalize_plausibility(alt_raw.get("plausibility"))
            alt_hypotheses.append({
                "title": alt_raw.get("title", "Alternative Hypothesis"),
                "incident_type": alt_raw.get("incident_type", alt_raw.get("title", "Alternative Hypothesis")),
                "narrative": alt_raw.get("narrative", ""),
                "plausibility": alt_plaus,
                "supporting_evidence": alt_raw.get("supporting_evidence", []),
                "contradicting_evidence": alt_raw.get("contradicting_evidence", []),
                "assumptions": alt_raw.get("assumptions", []),
                "uncertainty_factors": alt_raw.get("uncertainty_factors", []),
            })

        # Gather supporting evidence citations
        supporting_evidence = list(primary_raw.get("supporting_evidence", []))

        # Gather contradictions from LLM and cross-modal fusion conflicts
        contradictions = list(llm_result.get("contradictory_evidence", []))
        for item in primary_raw.get("contradicting_evidence", []):
            if item not in contradictions and item != "None identified in available evidence":
                contradictions.append(item)

        if fusion_data and "relationships" in fusion_data:
            for r in fusion_data["relationships"]:
                if r.get("relationship_type") == "conflict":
                    desc = f"Conflict between {r.get('source_a_evidence_id')} and {r.get('source_b_evidence_id')}: {r.get('rationale')}"
                    if desc not in contradictions:
                        contradictions.append(desc)

        # Gather evidence gaps
        evidence_gaps = list(llm_result.get("evidence_gaps", []))
        for u in primary_raw.get("uncertainty_factors", []):
            if u not in evidence_gaps:
                evidence_gaps.append(u)

        # Check alignment status caveat
        alignment = fusion_data.get("alignment_status", "independent") if fusion_data else "independent"
        if alignment != "aligned":
            sync_gap = "Recordings operate on independent, unsynchronized time bases. Source timestamps cannot be treated as synchronized."
            if sync_gap not in evidence_gaps:
                evidence_gaps.append(sync_gap)

        overall_assessment = (
            llm_result.get("overall_assessment")
            or primary_raw.get("narrative")
            or "Investigative hypothesis synthesized from recorded evidence."
        )

        return {
            "case_id": case_id,
            "incident_type": primary_type,
            "status": "completed",
            "plausibility": primary_plausibility,
            "narrative": primary_raw.get("narrative", ""),
            "primary_hypothesis": primary_hypothesis,
            "alternative_hypotheses": alt_hypotheses,
            "supporting_evidence": supporting_evidence,
            "contradictions": contradictions,
            "evidence_gaps": evidence_gaps,
            "overall_assessment": overall_assessment,
            "overall_limitations": llm_result.get("overall_limitations", ""),
            "evidence_sufficiency": llm_result.get("evidence_sufficiency", "limited"),
            "generated_at": now_iso,
            "model": llm_result.get("model", "openai/gpt-oss-20b"),
            "forensic_disclaimer": FORENSIC_DISCLAIMER,
        }

    def _generate_heuristic_hypothesis(
        self,
        case_id: str,
        case_data: dict[str, Any],
        fusion_data: dict[str, Any],
        now_iso: str,
    ) -> dict[str, Any]:
        """
        Deterministic, evidence-grounded heuristic synthesis when LLM is offline or unavailable.
        Reasons strictly over recorded evidence, preserving provenance and separating witness claims
        from visual observations.
        """
        evidence_items = case_data.get("evidence", [])
        video_items = [e for e in evidence_items if e.get("media_type") == "video" and e.get("analysis")]
        audio_items = [e for e in evidence_items if e.get("media_type") == "audio" and e.get("analysis")]
        image_items = [e for e in evidence_items if e.get("media_type") == "image" and e.get("analysis")]

        supporting_evidence: list[str] = []
        contradictions: list[str] = []
        evidence_gaps: list[str] = []

        # 1. Audit Cross-Modal Conflicts from Fusion
        if fusion_data and "relationships" in fusion_data:
            for r in fusion_data["relationships"]:
                if r.get("relationship_type") == "conflict":
                    contradictions.append(
                        f"Cross-modal conflict ({r.get('source_a_evidence_id')} vs {r.get('source_b_evidence_id')}): {r.get('rationale')}"
                    )

        # 2. Check alignment status
        alignment = fusion_data.get("alignment_status", "independent") if fusion_data else "independent"
        if alignment != "aligned":
            evidence_gaps.append(
                "Recordings operate on independent, unsynchronized time bases. Events across files cannot be assumed simultaneous."
            )

        # 3. Audio / Witness Statement Examination
        witness_allegations: list[dict[str, Any]] = []
        theft_terms = {"steal", "stole", "theft", "snatch", "snatched", "rob", "robbed", "robbery", "take", "took", "shoplift"}
        weapon_terms = {"gun", "knife", "weapon", "blade", "pistol", "revolver", "firearm", "shot"}

        for a_ev in audio_items:
            fn = a_ev.get("original_filename", "audio")
            analysis = a_ev.get("analysis") or {}
            trans = (analysis.get("transcription") or {}).get("transcript", "")
            if not trans:
                continue

            trans_lower = trans.lower()
            has_theft = any(term in trans_lower for term in theft_terms)
            has_weapon = any(term in trans_lower for term in weapon_terms)

            if has_theft or has_weapon:
                witness_allegations.append({
                    "filename": fn,
                    "evidence_id": a_ev.get("evidence_id"),
                    "transcript": trans,
                    "has_theft": has_theft,
                    "has_weapon": has_weapon,
                })
                snippet = trans[:100] + ("..." if len(trans) > 100 else "")
                supporting_evidence.append(f"[Audio] Witness statement in '{fn}': \"{snippet}\"")

        # 4. Video Analysis Examination
        video_hypotheses: list[dict[str, Any]] = []
        detected_weapons: list[dict[str, Any]] = []
        person_count_max = 0

        for v_ev in video_items:
            fn = v_ev.get("original_filename", "video")
            analysis = v_ev.get("analysis") or {}
            v_hyp = analysis.get("incident_hypothesis") or {}
            if v_hyp:
                video_hypotheses.append({
                    "filename": fn,
                    "type": v_hyp.get("type") or v_hyp.get("incident_type_hypothesis", "unknown"),
                    "reasoning": v_hyp.get("reasoning") or v_hyp.get("incident_hypothesis_reasoning", ""),
                })

            frames = analysis.get("frame_observations", [])
            for f in frames:
                t = f.get("timestamp_seconds", 0.0)
                p_cnt = f.get("persons", 0)
                if p_cnt > person_count_max:
                    person_count_max = p_cnt
                for d in f.get("detections", []):
                    lbl = str(d.get("label", "")).lower()
                    if lbl in {"knife", "gun", "pistol", "rifle", "weapon", "baseball bat"}:
                        detected_weapons.append({
                            "filename": fn,
                            "label": lbl,
                            "time": t,
                            "confidence": d.get("confidence"),
                        })

        # 5. Image Analysis Examination
        for i_ev in image_items:
            fn = i_ev.get("original_filename", "image")
            analysis = i_ev.get("analysis") or {}
            obj_det = analysis.get("object_detection") or {}
            for d in obj_det.get("detections", []):
                lbl = str(d.get("label", "")).lower()
                if lbl in {"knife", "gun", "pistol", "rifle", "weapon"}:
                    detected_weapons.append({
                        "filename": fn,
                        "label": lbl,
                        "time": None,
                        "confidence": d.get("confidence"),
                    })

        # Add detected weapon observations with forensic safeguards
        if detected_weapons:
            for dw in detected_weapons[:4]:
                t_str = f" @ {dw['time']:.2f}s" if dw["time"] is not None else ""
                conf_str = f" (confidence: {round(float(dw['confidence']), 2)})" if dw.get("confidence") else ""
                supporting_evidence.append(
                    f"[{'Video' if dw['time'] is not None else 'Image'}] RT-DETR detected '{dw['label']}' in '{dw['filename']}'{t_str}{conf_str}"
                )
            evidence_gaps.append(
                "Weapon detection does not establish intent, brandishing, or discharge. Automated bounding boxes are unverified."
            )

        # Check direct audio claim vs visual detection conflicts
        for a_ev in audio_items:
            fn = a_ev.get("original_filename", "audio")
            analysis = a_ev.get("analysis") or {}
            trans = (analysis.get("transcription") or {}).get("transcript", "")
            if not trans:
                segs = (analysis.get("transcription") or {}).get("segments", [])
                trans = " ".join(s.get("text", "") for s in segs)
            trans_lower = trans.lower()
            unarmed_indicators = ["unarmed", "no weapon", "empty hands", "no gun", "no knife", "without a weapon"]
            if any(u in trans_lower for u in unarmed_indicators) and detected_weapons:
                w_labels = ", ".join(set(dw["label"] for dw in detected_weapons))
                conflict_msg = (
                    f"Cross-modal conflict: Audio witness in '{fn}' claimed subject was unarmed/had empty hands, "
                    f"but visual models detected weapon ({w_labels})."
                )
                if conflict_msg not in contradictions:
                    contradictions.append(conflict_msg)

            solitude_indicators = ["alone", "nobody", "no one", "empty room", "by myself"]
            if any(s in trans_lower for s in solitude_indicators) and person_count_max > 1:
                conflict_msg = (
                    f"Cross-modal conflict: Audio witness in '{fn}' claimed solitude/alone, "
                    f"but visual models detected {person_count_max} persons in scene."
                )
                if conflict_msg not in contradictions:
                    contradictions.append(conflict_msg)

        # 6. Formulate candidate incident type
        # Scenario A: Witness alleges theft, but video observations do not confirm concealment/removal
        if witness_allegations:
            has_theft = any(w["has_theft"] for w in witness_allegations)
            has_weapon = any(w["has_weapon"] for w in witness_allegations) or bool(detected_weapons)

            if has_theft and has_weapon:
                primary_type = "Suspected Robbery / Armed Theft (Witness-Reported)"
                plausibility = "Moderate"
                narrative = (
                    "Witness statements report an armed theft or robbery. Visual models detected weapon indicators, "
                    "but physical item removal has not been independently confirmed on video."
                )
            elif has_theft:
                primary_type = "Suspected Shoplifting / Theft (Witness-Reported)"
                plausibility = "Moderate" if len(video_items) > 0 else "Limited"
                narrative = (
                    "A witness statement alleges a theft occurred. While video surveillance confirms human presence "
                    "in the area, recorded frames do not independently verify concealment or removal of items."
                )
            else:
                primary_type = "Suspected Weapon Presence / Threat"
                plausibility = "Moderate"
                narrative = "Witness accounts and automated detections indicate potential weapon presence."

            contradictions.append(
                "Witness statement reports criminal action; however, surveillance frames do not independently establish the physical removal of merchandise or property."
            )
            evidence_gaps.append(
                "Visual blind spots or frame intervals may omit the critical moment of reported item removal."
            )

            alt_hypotheses = [
                {
                    "title": "Routine Presence / Unverified Allegation",
                    "incident_type": "Routine Activity",
                    "narrative": (
                        "Individuals were lawfully present in the recorded area. The reported theft or altercation "
                        "cannot be corroborated from available visual frames alone."
                    ),
                    "plausibility": "Moderate",
                    "supporting_evidence": [f"Persons observed in scene ({person_count_max} detected)."],
                    "contradicting_evidence": [w["transcript"][:80] + "..." for w in witness_allegations[:2]],
                    "assumptions": ["Witness account is inaccurate, mistaken, or misinterpreting routine interaction."],
                    "uncertainty_factors": ["Lack of continuous camera coverage of transaction counter."],
                }
            ]

        elif detected_weapons:
            primary_type = "Possible Weapon Presence / Armed Intrusion"
            plausibility = "Moderate"
            narrative = (
                "Automated object detection identified potential weapon-like objects in recorded evidence. "
                "No witness testimony or physical confrontation has been corroborated."
            )
            alt_hypotheses = [
                {
                    "title": "Routine Activity with False-Positive Detection",
                    "incident_type": "Routine Activity",
                    "narrative": "Detected weapon class represents an automated false positive or benign object.",
                    "plausibility": "Moderate",
                    "supporting_evidence": ["No aggressive action or threat reported in audio transcripts."],
                    "contradicting_evidence": [f"Weapon detection in {detected_weapons[0]['filename']}."],
                    "assumptions": ["Object detector misclassified a harmless personal item."],
                    "uncertainty_factors": ["RT-DETR COCO classes include knives but lack context calibration."],
                }
            ]

        elif video_hypotheses:
            first_v = video_hypotheses[0]
            v_type_clean = first_v["type"].replace("_", " ").title()
            primary_type = f"Candidate Incident: {v_type_clean}"
            plausibility = "Moderate" if "insufficient" not in first_v["type"] else "Limited"
            narrative = first_v["reasoning"] or "Video surveillance observations suggest candidate incident activity."
            alt_hypotheses = [
                {
                    "title": "Routine Activity",
                    "incident_type": "Routine Activity",
                    "narrative": "Observed activity represents authorized, routine presence without criminal intent.",
                    "plausibility": "Moderate",
                    "supporting_evidence": ["Absence of overt panic or struggle."],
                    "contradicting_evidence": [],
                    "assumptions": ["Subjects have lawful access to the premises."],
                    "uncertainty_factors": ["Absence of access-control logs."],
                }
            ]

        else:
            primary_type = "Routine Activity / Undetermined Incident"
            plausibility = "Moderate"
            narrative = (
                "Recorded evidence documents human presence or scene captures without verified criminal indicators. "
                "Available observations are consistent with routine presence."
            )
            alt_hypotheses = []

        overall_assessment = (
            f"Forensic synthesis of {len(evidence_items)} evidence item(s). "
            f"Primary candidate: '{primary_type}'. "
            "Direct observations confirm recorded presence; however, specific criminal acts, intent, "
            "and identities remain unverified AI hypotheses requiring investigator substantiation."
        )

        primary_hypothesis = {
            "title": primary_type,
            "incident_type": primary_type,
            "narrative": narrative,
            "plausibility": plausibility,
            "supporting_evidence": supporting_evidence or ["Human presence verified across recorded frames."],
            "contradicting_evidence": contradictions or ["None identified in available records."],
            "assumptions": ["Visual observations capture the primary period of interest."],
            "uncertainty_factors": evidence_gaps or ["Single-camera coverage; unverified subject identity."],
        }

        return {
            "case_id": case_id,
            "incident_type": primary_type,
            "status": "completed",
            "plausibility": plausibility,
            "narrative": narrative,
            "primary_hypothesis": primary_hypothesis,
            "alternative_hypotheses": alt_hypotheses,
            "supporting_evidence": supporting_evidence,
            "contradictions": contradictions,
            "evidence_gaps": evidence_gaps,
            "overall_assessment": overall_assessment,
            "overall_limitations": "Heuristic synthesis based on automated model observations. Requires human review.",
            "evidence_sufficiency": "limited" if plausibility == "Limited" else "moderate",
            "generated_at": now_iso,
            "model": "heuristic_synthesizer",
            "forensic_disclaimer": FORENSIC_DISCLAIMER,
        }
