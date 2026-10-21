
import json
import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from app.services.token_budget import (
    GroqRateLimitExceeded,
    TokenBudgetManager,
    default_token_budget,
)

load_dotenv()
logger = logging.getLogger(__name__)


class GroqService:
    """Generate cautious summaries from structured image findings."""

    MODEL_NAME = "openai/gpt-oss-20b"

    def __init__(self, token_budget: TokenBudgetManager | None = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing from .env")

        self.client = Groq(api_key=api_key)
        self.token_budget = token_budget or default_token_budget

    def analyze_image_findings(self, findings: dict) -> dict:
        prompt = f"""
You assist human reviewers examining forensic evidence.

Analyze ONLY the supplied model outputs. You have not independently
inspected or verified the original image.

Rules:
- Treat all model predictions as unverified.
- Never invent identities, objects, actions, motives, or relationships.
- Distinguish detections, scene labels, and generated captions.
- Preserve supplied labels and confidence scores accurately.
- Do not claim an object is absent just because it was not detected.
- Label interpretations as hypotheses, not conclusions.
- Explicitly state uncertainty and limitations.
- Do not claim to have verified the original image.

Return:
1. Summary of model outputs
2. Detection results and supplied confidence scores
3. Possible interpretations, labeled as unverified hypotheses
4. Limitations and recommended human verification

Keep the response concise but complete.

Structured model outputs:
{json.dumps(findings, ensure_ascii=False, indent=2, default=str)}
"""

        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide cautious, evidence-grounded analysis. "
                        "Never present model predictions as verified facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        choice = response.choices[0]
        analysis = choice.message.content or ""

        return {
            "model": self.MODEL_NAME,
            "analysis": analysis,
            "note": (
                "AI-generated interpretation of unverified model outputs. "
                "Human review of the original evidence is required."
            ),
            "truncated": choice.finish_reason == "length",
        }

    def extract_audio_events(self, transcription_data: dict) -> dict:
        """
        Extract structured anchor events from speech transcription segments.
        Activity 3.2: Extract Key Events and Generate Structured Timelines.
        """
        segments = transcription_data.get("segments", [])
        transcript = transcription_data.get("transcript", "")

        prompt = f"""
You assist forensic investigators analyzing audio evidence transcripts.

Analyze the supplied speech transcript segments to extract key anchor events.
Original recording transcript:
"{transcript}"

Timestamped segments:
{json.dumps(segments, ensure_ascii=False, indent=2)}

Rules for event extraction:
1. ONLY extract events that are directly and explicitly supported by the transcript text.
2. Categories include: threat, physical_struggle, arrival, suspect_mention, aggression, identification, spatial_movement, statement_or_allegation, distress_call, or other relevant forensic event types.
3. Distinguish between a speaker's allegation or subjective claim and a verified physical event. Set is_allegation to true for unverified claims.
4. DO NOT invent identities, actions, motives, locations, or precise times.
5. Extract who, what, and stated location ONLY if explicitly mentioned in the text; otherwise use null.
6. Map each event to the exact source_segment_ids (list of integers matching segment id) that support it.
7. Include exact supporting_quotes from the segments.
8. State clear uncertainties for each event (e.g., speaker identity unverified, unconfirmed claim).
9. Define confidence as model confidence (0.0 to 1.0), not factual certainty.
10. Return strictly a JSON object with an "events" list:
{{
  "events": [
    {{
      "event_type": "threat | physical_struggle | arrival | suspect_mention | aggression | identification | spatial_movement | statement_or_allegation | distress_call | other",
      "description": "Evidence-grounded description of the event or statement",
      "source_segment_ids": [0],
      "supporting_quotes": ["quote from segment"],
      "confidence": 0.85,
      "stated_location": "location or null",
      "entities_mentioned": ["entity 1"],
      "is_allegation": false,
      "uncertainties": ["Speaker identity unverified"]
    }}
  ]
}}
"""

        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide cautious, evidence-grounded forensic event extraction. "
                        "Never invent details or treat unverified allegations as facts. "
                        "Return strictly valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
        )

        choice = response.choices[0]
        content = choice.message.content or "{}"

        return {
            "model": self.MODEL_NAME,
            "raw_content": content,
            "truncated": choice.finish_reason == "length",
        }

    def analyze_video_events(self, video_summary: dict) -> dict:
        """
        Analyze sequential frame detections and descriptions to infer temporal visual
        events, key transitions, and a cautious incident hypothesis.
        Activity 4.2: Action Synthesis and Event Classification.
        """
        prompt = f"""
You assist forensic investigators analyzing sequential video surveillance frame observations.
The video was sampled at regular intervals (approx 4 frames/sec).

Sequential frame observations:
{json.dumps(video_summary, ensure_ascii=False, indent=2, default=str)}

Rules:
1. ONLY describe visual events supported by the sequential frame observations.
2. Note changes in model-predicted person presence, object detections, or scene descriptions.
3. DO NOT assume continuous identity tracking; RT-DETR detects per-frame objects, not continuous biometric identities. DO NOT infer or claim physical entry or exit solely from person detections or person count changes; qualify as model-predicted presence or absence.
4. DO NOT label model detections as verified facts; all detections are automated model predictions.
5. DO NOT convert object presence alone into a crime allegation.
6. DO NOT fabricate weapons, aggressive actions, or motives unless explicitly visible and supported by observations.
7. Classify incident_type_hypothesis cautiously. Valid values:
   "possible_theft_related_activity", "possible_physical_altercation", "property_damage",
   "loitering_or_trespassing", "suspicious_movement", "routine_activity", "unknown", "insufficient_evidence".
   Default to "insufficient_evidence" or "unknown" unless direct observations justify another hypothesis.
8. Return strictly a JSON object with this schema:
{{
  "incident_type_hypothesis": "insufficient_evidence | routine_activity | unknown | possible_theft_related_activity | ...",
  "incident_hypothesis_reasoning": "Cautious explanation based strictly on observed frames.",
  "events": [
    {{
      "event_id": "evt_v001",
      "timestamp_seconds": 0.0,
      "end_timestamp_seconds": 3.0,
      "event_type": "person_entered_frame | person_left_frame | object_interaction | movement | scene_transition | other",
      "description": "Evidence-grounded description of the observation",
      "observed_evidence": "Detections and descriptions supporting this event",
      "source_frame_indices": [0, 1, 2],
      "source_timestamps": [0.0, 1.0, 2.0],
      "confidence": 0.80,
      "reasoning_limitations": ["Observation derived from sampled frames; identity untracked"]
    }}
  ]
}}
"""

        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide cautious, evidence-grounded video surveillance analysis. "
                        "Never fabricate actions, weapons, or identities. "
                        "Return strictly valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1600,
        )

        choice = response.choices[0]
        content = choice.message.content or "{}"

        return {
            "model": self.MODEL_NAME,
            "raw_content": content,
            "truncated": choice.finish_reason == "length",
        }

    # ------------------------------------------------------------------
    # Activity 5.2: Intent classification patterns
    # ------------------------------------------------------------------
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

    @staticmethod
    def classify_query_intent(question: str) -> str:
        """Classify an investigator query as 'hypothesis', 'timestamp', or 'general'.

        Activity 5.2: Determines the processing path for a Q&A query.
        - 'hypothesis': Request for evidence-supported investigative hypotheses.
        - 'timestamp': Question about a specific timestamp or frame.
        - 'general': Any other evidence-grounded question.
        """
        if GroqService._HYPOTHESIS_PATTERNS.search(question):
            return "hypothesis"

        from app.services.context_builder import extract_query_target
        target = extract_query_target(question)
        if target:
            return "timestamp"

        return "general"

    def answer_case_question(self, case_context: dict, question: str) -> dict:
        """
        Answer investigator queries strictly grounded in stored case evidence findings.
        Epic 4 Section 7 + Activity 5.2: Investigator Q&A with strict evidence grounding
        and cross-modal fusion awareness.
        """
        prompt = f"""
You assist forensic investigators reviewing evidence findings for an active investigation.
Answer the investigator's question STRICTLY based on the provided case evidence context.

Case Evidence Context:
{json.dumps(case_context, ensure_ascii=False, separators=(',', ':'), default=str)}

Investigator Question:
"{question}"

FORENSIC GROUNDING & CITATION RULES:
1. PRIMARY SOURCE OF TRUTH (SAMPLED FRAME OBSERVATIONS):
   - For questions about visual content, detections, objects, persons, or specific timestamps, prioritize the `primary_frame_observations` (sampled-frame data) as your primary source of truth.
   - Do NOT rely solely on timeline events when underlying frame observations are available.
   - Do NOT claim an object was absent merely because it is not mentioned in an event summary; check the frame detections directly.

2. MODEL DETECTIONS VS. VERIFIED FACTS:
   - All object detections (RT-DETR), scene classifications (CLIP), captions (BLIP), and transcripts (Whisper) are AUTOMATED MODEL PREDICTIONS, NOT "verified facts", "proof", or "directly verified findings".
   - NEVER label or describe model-generated detections as "verified facts", "confirmed", "proven", or "directly verified".
   - Treat detected labels strictly as model predictions (e.g. "The object detection model predicted a 'person' with 0.89 confidence...").
   - NEVER claim or imply independent human verification unless the case record explicitly contains an investigator's verified finding.

3. ACCURATE AND DETAILED CITATIONS:
   - For every claim regarding visual content, cite the specific sampled frame: video filename, evidence ID, timestamp in seconds, and frame index (e.g. "[cctv_entrance.mp4 (Evidence ID: evi_001), Frame 25 at 1.0s]"). Include the detection confidence if available.
   - For audio claims: cite the audio filename, evidence ID, and segment timestamp.
   - Do NOT cite a timeline event as the sole support for a visual claim when the underlying frame observation is present.

4. AVOID UNSUPPORTED MOVEMENT / ENTRY / EXIT CLAIMS:
   - Do NOT infer or claim that a person "entered", "arrived", "exited", "escaped", or "fled" solely because an object detector detected a person or because person counts changed between frames.
   - Per-frame detectors do not track persistent identity over time. Explicitly state that while a person was detected at timestamp X and 0 persons at timestamp Y, physical entry or exit cannot be established without continuous tracking (the subject may have become occluded, moved off-camera, or been a different individual).

5. UNAVAILABLE FRAMES OR EVIDENCE:
   - If the requested timestamp, video, or frame observation is missing, outside the sampled interval, or marked unavailable in the context, explicitly state that limitation:
     e.g. "Sampled frame observations for timestamp [X]s are unavailable in the recorded evidence for [video]."
   - Do NOT guess or fabricate what might have happened between sampled frames.
   - If the case context does not contain evidence to answer the question, state: "Based on the recorded evidence in this case, there is no information regarding [topic]."

6. CROSS-MODAL FUSION CONTEXT:
   - When cross-modal fusion findings (`fusion_context`) are present, reference corroborations and conflicts between evidence sources.
   - A 'qualified_corroboration' is a potential link with stated caveats, not a verified match.
   - A 'conflict' means the sources disagree — report the disagreement honestly.
   - If alignment_status is 'unknown' or 'independent', explicitly state that temporal co-occurrence is unestablished.
"""

        # Enforce conservative token budget & check sliding-window rate limit
        prompt_tokens = self.token_budget.count_tokens(prompt)
        max_output = self.token_budget.max_output_tokens_qa
        est_total = prompt_tokens + max_output

        can_proceed, wait_time, reason = self.token_budget.can_proceed(est_total)
        if 0.0 < wait_time <= 3.0:
            logger.info(f"[GroqService] Pacing request under TPM limit, waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
        elif not can_proceed:
            logger.warning(f"[GroqService] Rate limit prevented request: {reason}")
            raise GroqRateLimitExceeded(reason, wait_seconds=wait_time)

        response = None
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an evidence-grounding forensic assistant for criminal investigations. "
                                "Base all answers strictly on the provided case data. "
                                "Treat all AI detections as unverified model predictions, never as verified facts or proof. "
                                "Never infer movement, entry, or exit without tracking. "
                                "Always cite the exact sampled frame (filename, evidence ID, timestamp, frame index, confidence). "
                                "Explicitly state when requested timestamps or frames are unavailable. "
                                "When cross-modal fusion context is provided, reference corroborations and conflicts "
                                "but never present them as independently verified conclusions."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.15,
                    max_tokens=max_output,
                )
                break
            except Exception as err:
                err_str = str(err).lower()
                is_rate = "rate limit" in err_str or "429" in err_str or "413" in err_str or "tokens per minute" in err_str
                if is_rate and attempt == 0:
                    retry_match = re.search(r"retry\s*after\s*(\d+(?:\.\d+)?)", err_str)
                    backoff = float(retry_match.group(1)) if retry_match else 2.0
                    if backoff <= 3.0:
                        logger.info(f"[GroqService] Rate limit encountered, retrying after {backoff}s...")
                        time.sleep(backoff)
                        continue
                if is_rate:
                    raise GroqRateLimitExceeded(f"Groq API rate limit reached: {err}")
                raise

        choice = response.choices[0]
        answer = choice.message.content or ""

        # Record usage in sliding-window rate tracker
        usage = getattr(response, "usage", None)
        total_used = getattr(usage, "total_tokens", None) if usage else None
        self.token_budget.record_usage(total_used or (prompt_tokens + self.token_budget.count_tokens(answer)))

        return {
            "model": self.MODEL_NAME,
            "answer": answer,
            "note": "AI-generated response grounded in recorded case findings. Model predictions are unverified by human investigators.",
            "truncated": choice.finish_reason == "length",
        }

    def generate_hypotheses(self, case_context: dict, question: str) -> dict:
        """
        Generate up to 3 evidence-supported investigative hypotheses.
        Activity 5.2: Hypothesis generation grounded in multimodal fusion results.

        Returns a dict with 'hypotheses' list, 'model', 'note', and 'truncated' keys.
        Each hypothesis includes: title, narrative, supporting_evidence,
        contradicting_evidence, assumptions, uncertainty_factors, and plausibility.
        """
        fusion = case_context.get("fusion_context") or {}
        evidence_summary = {
            "case_id": case_context.get("case_id"),
            "title": case_context.get("title"),
            "video_summaries": case_context.get("video_summaries", []),
            "audio_transcripts": case_context.get("audio_transcripts", []),
            "image_evidence": case_context.get("image_evidence", []),
            "timeline_events": case_context.get("derived_timeline_events", []),
            "fusion": fusion,
        }

        prompt = f"""
You assist forensic investigators generating evidence-supported investigative hypotheses.

Based STRICTLY on the provided case evidence, generate up to 3 plausible hypotheses about what
may have occurred. Each hypothesis must be directly grounded in the recorded evidence.

Case Evidence Summary:
{json.dumps(evidence_summary, ensure_ascii=False, separators=(',', ':'), default=str)}

Investigator Query:
"{question}"

HYPOTHESIS GENERATION RULES:
1. Generate between 1 and 3 hypotheses. Only generate additional hypotheses if the evidence
   genuinely supports alternative interpretations. Do NOT pad with speculative hypotheses.
2. Each hypothesis MUST cite specific evidence items (evidence IDs, filenames, timestamps,
   detection labels, transcript quotes) that support it.
3. For each hypothesis, identify:
   - supporting_evidence: Evidence items that support this hypothesis (cite specifically).
   - contradicting_evidence: Evidence items that weaken or contradict this hypothesis.
   - assumptions: What must be assumed beyond the recorded evidence for this hypothesis to hold.
   - uncertainty_factors: Limitations of the evidence and model predictions relevant to this hypothesis.
   - plausibility: "high", "moderate", or "low" based on the strength and quantity of supporting evidence.
4. All model detections (RT-DETR, CLIP, BLIP, Whisper) are UNVERIFIED AUTOMATED PREDICTIONS.
   Do NOT treat them as facts. Do NOT claim verification.
5. Cross-modal fusion relationships (corroborations, conflicts) are AUTOMATED CROSS-REFERENCES,
   not independently verified links. Acknowledge them but state their automated nature.
6. If the evidence is insufficient to support any meaningful hypothesis, state that explicitly
   and return an empty hypotheses list.
7. Do NOT fabricate evidence, identities, motives, timestamps, citations, or events not present in the data.
8. Separate direct visual observations, witness-reported claims, model-generated interpretations, and unverified hypotheses.
9. Do NOT infer theft merely because a person and a bag appear in the same scene.
10. Do NOT infer a weapon's use from its detection alone.
11. Do NOT treat generic matches across modalities as proof of the same person or event.
12. Do NOT treat source-relative timestamps from separate recordings as synchronized.
13. Preserve contradictions instead of forcing all evidence into one narrative.

Return strictly a JSON object:
{{
  "primary_incident_type": "Suspected Shoplifting | Suspected Robbery | Theft from Vehicle | Burglary | Routine Activity | Undetermined / Insufficient Evidence",
  "hypotheses": [
    {{
      "title": "Brief hypothesis title",
      "narrative": "Evidence-grounded narrative describing the hypothesized sequence of events",
      "supporting_evidence": [
        "Specific evidence citation with IDs, timestamps, and model predictions"
      ],
      "contradicting_evidence": [
        "Evidence that weakens this hypothesis, or 'None identified in available evidence'"
      ],
      "assumptions": [
        "What must be assumed for this hypothesis"
      ],
      "uncertainty_factors": [
        "Limitations of evidence and model predictions"
      ],
      "plausibility": "Strongly Supported | Moderate | Limited | Undetermined"
    }}
  ],
  "contradictory_evidence": [
    "Specific conflict between witness statement and video observation, or other finding"
  ],
  "evidence_gaps": [
    "Specific gap such as unsynchronized recordings, poor video quality, or unverified theft action"
  ],
  "overall_assessment": "Summary explaining what can reasonably be inferred vs what remains unverified.",
  "evidence_sufficiency": "sufficient | limited | insufficient",
  "overall_limitations": "Summary of key evidence gaps and model limitations"
}}
"""

        prompt_tokens = self.token_budget.count_tokens(prompt)
        max_output = self.token_budget.max_output_tokens_hypothesis
        est_total = prompt_tokens + max_output

        can_proceed, wait_time, reason = self.token_budget.can_proceed(est_total)
        if 0.0 < wait_time <= 3.0:
            logger.info(f"[GroqService] Pacing hypothesis request, waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
        elif not can_proceed:
            logger.warning(f"[GroqService] Rate limit prevented hypothesis request: {reason}")
            raise GroqRateLimitExceeded(reason, wait_seconds=wait_time)

        response = None
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a forensic hypothesis generator. "
                                "Generate evidence-supported investigative hypotheses strictly grounded in provided case data. "
                                "Treat all AI model predictions as unverified. Never fabricate evidence. "
                                "Return strictly valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.15,
                    max_tokens=max_output,
                )
                break
            except Exception as err:
                err_str = str(err).lower()
                is_rate = "rate limit" in err_str or "429" in err_str or "413" in err_str or "tokens per minute" in err_str
                if is_rate and attempt == 0:
                    retry_match = re.search(r"retry\s*after\s*(\d+(?:\.\d+)?)", err_str)
                    backoff = float(retry_match.group(1)) if retry_match else 2.0
                    if backoff <= 3.0:
                        logger.info(f"[GroqService] Rate limit encountered, retrying hypothesis after {backoff}s...")
                        time.sleep(backoff)
                        continue
                if is_rate:
                    raise GroqRateLimitExceeded(f"Groq API rate limit reached: {err}")
                raise

        choice = response.choices[0]
        raw_content = choice.message.content or "{}"

        usage = getattr(response, "usage", None)
        total_used = getattr(usage, "total_tokens", None) if usage else None
        self.token_budget.record_usage(total_used or (prompt_tokens + self.token_budget.count_tokens(raw_content)))

        # Parse the JSON response
        hypotheses_data: dict[str, Any] = {}
        try:
            # Strip markdown code fences if present
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```\s*$", "", cleaned)
            hypotheses_data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            hypotheses_data = {
                "hypotheses": [],
                "evidence_sufficiency": "insufficient",
                "overall_limitations": "Unable to parse LLM hypothesis response.",
                "raw_content": raw_content,
            }

        first_hyp = (hypotheses_data.get("hypotheses") or [{}])[0]
        derived_type = hypotheses_data.get("primary_incident_type") or first_hyp.get("title") or "Undetermined / Insufficient Evidence"
        return {
            "model": self.MODEL_NAME,
            "primary_incident_type": derived_type,
            "hypotheses": hypotheses_data.get("hypotheses", []),
            "contradictory_evidence": hypotheses_data.get("contradictory_evidence", []),
            "evidence_gaps": hypotheses_data.get("evidence_gaps", []),
            "overall_assessment": hypotheses_data.get("overall_assessment", ""),
            "evidence_sufficiency": hypotheses_data.get("evidence_sufficiency", "unknown"),
            "overall_limitations": hypotheses_data.get("overall_limitations", ""),
            "note": (
                "AI-generated hypotheses grounded in recorded case evidence. "
                "All model predictions are unverified. Hypotheses are investigative aids, "
                "not conclusions. Human review and independent verification are required."
            ),
            "truncated": choice.finish_reason == "length",
        }

