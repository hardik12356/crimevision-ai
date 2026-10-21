"""
API Routes for CrimeVisionAI Flask Application.
"""

import io
from pathlib import Path
import re
import uuid
from typing import Any
from flask import Blueprint, current_app, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from app.services.audio_service import AudioService
from app.services.audio_transcriber import AudioTranscriber
from app.services.case_service import CaseService
from app.services.context_builder import ContextBuilder, extract_query_target
from app.services.fusion_service import FusionService
from app.services.groq_service import GroqService
from app.services.hypothesis_service import HypothesisService
from app.services.pdf_service import PdfReportService
from app.services.token_budget import GroqRateLimitExceeded
from app.services.video_service import VideoService
from app.services.vision_service import VisionService

api_bp = Blueprint("api", __name__, url_prefix="/api")
main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET"])
def index():
    """Serve the CrimeVisionAI Dashboard MVP."""
    return render_template("index.html")


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Default service instances, can be overridden in app config for testing
_default_audio_service = None
_default_vision_service = None
_default_video_service = None
_default_case_service = None
_default_fusion_service = None


def get_case_service() -> CaseService:
    """Retrieve case service from app config or use default."""
    if current_app and "CASE_SERVICE" in current_app.config:
        return current_app.config["CASE_SERVICE"]
    if current_app and "DATABASE_PATH" in current_app.config:
        return CaseService(db_path=current_app.config["DATABASE_PATH"])
    global _default_case_service
    if _default_case_service is None:
        _default_case_service = CaseService()
    return _default_case_service


def get_video_service() -> VideoService:
    """Retrieve video service from app config or use default."""
    if current_app and "VIDEO_SERVICE" in current_app.config:
        return current_app.config["VIDEO_SERVICE"]
    global _default_video_service
    if _default_video_service is None:
        _default_video_service = VideoService()
    return _default_video_service


def get_audio_service() -> AudioService:
    """Retrieve audio service from app config or use default."""
    if current_app and "AUDIO_SERVICE" in current_app.config:
        return current_app.config["AUDIO_SERVICE"]
    global _default_audio_service
    if _default_audio_service is None:
        _default_audio_service = AudioService()
    return _default_audio_service


def get_vision_service() -> VisionService:
    """Retrieve vision service from app config or use default."""
    if current_app and "VISION_SERVICE" in current_app.config:
        return current_app.config["VISION_SERVICE"]
    global _default_vision_service
    if _default_vision_service is None:
        _default_vision_service = VisionService()
    return _default_vision_service


def get_fusion_service() -> FusionService:
    """Retrieve fusion service from app config or use default."""
    if current_app and "FUSION_SERVICE" in current_app.config:
        return current_app.config["FUSION_SERVICE"]
    global _default_fusion_service
    if _default_fusion_service is None:
        _default_fusion_service = FusionService()
    return _default_fusion_service


_default_pdf_service = None


def get_pdf_service() -> PdfReportService:
    """Retrieve PDF report service from app config or use default."""
    if current_app and "PDF_SERVICE" in current_app.config:
        return current_app.config["PDF_SERVICE"]
    global _default_pdf_service
    if _default_pdf_service is None:
        _default_pdf_service = PdfReportService()
    return _default_pdf_service


_default_hypothesis_service = None


def get_hypothesis_service() -> HypothesisService:
    """Retrieve incident hypothesis service from app config or use default."""
    if current_app and "HYPOTHESIS_SERVICE" in current_app.config:
        return current_app.config["HYPOTHESIS_SERVICE"]
    global _default_hypothesis_service
    if _default_hypothesis_service is None:
        case_svc = get_case_service()
        fusion_svc = get_fusion_service()
        groq_svc = current_app.config.get("GROQ_SERVICE") if current_app else None
        _default_hypothesis_service = HypothesisService(
            case_service=case_svc,
            fusion_service=fusion_svc,
            groq_service=groq_svc,
        )
    return _default_hypothesis_service


@api_bp.route("/audio/analyze", methods=["POST"])
def analyze_audio():
    """
    POST /api/audio/analyze
    Consolidated audio analysis endpoint:
    Accepts an uploaded audio file, executes AudioService (Whisper transcription +
    Groq timeline generation), and returns structured findings.
    """
    # 1. Validate presence of uploaded file
    file = request.files.get("file") or request.files.get("audio")
    if not file or not file.filename:
        return jsonify({
            "error": "Missing audio file",
            "message": "Please provide an audio file in the 'file' or 'audio' form field.",
        }), 400

    # 2. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in AudioTranscriber.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(AudioTranscriber.SUPPORTED_EXTENSIONS))
        return jsonify({
            "error": "Unsupported audio format",
            "message": f"Format '{ext}' is not supported. Supported formats: {supported}",
        }), 400

    # 3. Sanitize and save upload safely
    upload_dir = Path(current_app.config.get("UPLOAD_FOLDER", "evidence/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename)
    if not safe_name:
        safe_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"

    dest_path = upload_dir / safe_name
    try:
        file.save(str(dest_path))
    except Exception:
        return jsonify({
            "error": "Upload failed",
            "message": "Unable to save uploaded audio file to disk.",
        }), 500

    # 4. Optional parameters
    language = request.form.get("language") or request.args.get("language")

    # 5. Execute AudioService workflow
    audio_service = get_audio_service()
    try:
        result = audio_service.process_audio(
            audio_path=dest_path,
            language=language,
        )
        return jsonify(result), 200
    except (FileNotFoundError, ValueError) as err:
        return jsonify({
            "error": "Validation error",
            "message": str(err),
        }), 400
    except Exception as err:
        # Return generic error message without exposing stack traces or internal secrets
        return jsonify({
            "error": "Processing error",
            "message": f"Audio processing failed: {str(err)}",
        }), 500


@api_bp.route("/image/analyze", methods=["POST"])
def analyze_image():
    """
    POST /api/image/analyze
    Consolidated image analysis endpoint:
    Accepts an uploaded image file (.jpg, .jpeg, .png, .bmp, .webp, .tiff),
    executes VisionService (RT-DETR detection, CLIP scene classification,
    BLIP captioning, Groq interpretation), and returns structured findings.
    """
    # 1. Validate presence of uploaded file
    file = request.files.get("file") or request.files.get("image")
    if not file or not file.filename:
        return jsonify({
            "error": "Missing image file",
            "message": "Please provide an image file in the 'file' or 'image' form field.",
        }), 400

    # 2. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        return jsonify({
            "error": "Unsupported image format",
            "message": f"Format '{ext}' is not supported. Supported formats: {supported}",
        }), 400

    # 3. Sanitize and save upload safely
    upload_dir = Path(current_app.config.get("UPLOAD_FOLDER", "evidence/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename)
    if not safe_name:
        safe_name = f"image_{uuid.uuid4().hex[:8]}{ext}"

    dest_path = upload_dir / safe_name
    try:
        file.save(str(dest_path))
    except Exception:
        return jsonify({
            "error": "Upload failed",
            "message": "Unable to save uploaded image file to disk.",
        }), 500

    # 4. Check for empty file
    if dest_path.stat().st_size == 0:
        dest_path.unlink(missing_ok=True)
        return jsonify({
            "error": "Empty file",
            "message": "Uploaded image file is empty (0 bytes).",
        }), 400

    # 5. Optional confidence threshold parameter
    confidence_threshold = 0.25
    conf_param = request.form.get("confidence_threshold") or request.args.get("confidence_threshold")
    if conf_param is not None:
        try:
            val = float(conf_param)
            if not (0.0 <= val <= 1.0):
                return jsonify({
                    "error": "Invalid parameter",
                    "message": "confidence_threshold must be between 0.0 and 1.0.",
                }), 400
            confidence_threshold = val
        except ValueError:
            return jsonify({
                "error": "Invalid parameter",
                "message": "confidence_threshold must be a valid float.",
            }), 400

    # 6. Execute VisionService workflow
    vision_service = get_vision_service()
    try:
        result = vision_service.analyze_image(
            image_path=str(dest_path),
            confidence_threshold=confidence_threshold,
        )
        return jsonify(result), 200
    except (FileNotFoundError, ValueError) as err:
        return jsonify({
            "error": "Validation error",
            "message": str(err),
        }), 400
    except Exception as err:
        return jsonify({
            "error": "Processing error",
            "message": f"Image processing failed: {str(err)}",
        }), 500


@api_bp.route("/video/analyze", methods=["POST"])
def analyze_video():
    """
    POST /api/video/analyze
    Consolidated video analysis endpoint:
    Accepts an uploaded video file (.mp4, .mov, .avi, .mkv, .webm),
    executes VideoService (OpenCV 1 FPS frame sampling, RT-DETR detection,
    BLIP captioning, Groq temporal event synthesis), and returns structured findings.
    """
    # 1. Validate file presence
    file = request.files.get("file") or request.files.get("video")
    if not file or not file.filename:
        return jsonify({
            "error": "Missing video file",
            "message": "Please provide a video file in the 'file' or 'video' form field.",
        }), 400

    # 2. Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        return jsonify({
            "error": "Unsupported video format",
            "message": f"Format '{ext}' is not supported. Supported formats: {supported}",
        }), 400

    # 3. Sanitize and save upload safely
    upload_dir = Path(current_app.config.get("UPLOAD_FOLDER", "evidence/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename)
    if not safe_name:
        safe_name = f"video_{uuid.uuid4().hex[:8]}{ext}"

    dest_path = upload_dir / safe_name
    try:
        file.save(str(dest_path))
    except Exception:
        return jsonify({
            "error": "Upload failed",
            "message": "Unable to save uploaded video file to disk.",
        }), 500

    # 4. Check for empty file
    if dest_path.stat().st_size == 0:
        dest_path.unlink(missing_ok=True)
        return jsonify({
            "error": "Empty file",
            "message": "Uploaded video file is empty (0 bytes).",
        }), 400

    # 5. Optional confidence threshold parameter
    confidence_threshold = 0.25
    conf_param = request.form.get("confidence_threshold") or request.args.get("confidence_threshold")
    if conf_param is not None:
        try:
            val = float(conf_param)
            if not (0.0 <= val <= 1.0):
                return jsonify({
                    "error": "Invalid parameter",
                    "message": "confidence_threshold must be between 0.0 and 1.0.",
                }), 400
            confidence_threshold = val
        except ValueError:
            return jsonify({
                "error": "Invalid parameter",
                "message": "confidence_threshold must be a valid float.",
            }), 400

    # 6. Execute VideoService workflow
    video_service = get_video_service()
    try:
        result = video_service.analyze_video(
            video_path=dest_path,
            confidence_threshold=confidence_threshold,
        )
        return jsonify(result), 200
    except (FileNotFoundError, ValueError) as err:
        return jsonify({
            "error": "Validation error",
            "message": str(err),
        }), 400
    except Exception as err:
        return jsonify({
            "error": "Processing error",
            "message": f"Video processing failed: {str(err)}",
        }), 500


# ===========================================================================
# Case & Evidence Management Routes
# ===========================================================================

@api_bp.route("/cases", methods=["GET"])
def list_cases():
    """
    GET /api/cases
    List all recorded cases with summary evidence statistics.
    """
    case_service = get_case_service()
    try:
        cases = case_service.list_cases()
        return jsonify({"status": "success", "cases": cases}), 200
    except Exception as err:
        return jsonify({
            "error": "Database error",
            "message": f"Failed to list cases: {str(err)}",
        }), 500


@api_bp.route("/cases", methods=["POST"])
def create_case():
    """
    POST /api/cases
    Create a new investigation case.
    Payload: {"title": "...", "description": "...", "case_id": "..." (optional)}
    """
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    case_id = payload.get("case_id")
    if case_id is not None:
        case_id = str(case_id).strip()

    if not title:
        return jsonify({
            "error": "Validation error",
            "message": "Case title is required and cannot be empty.",
        }), 400

    case_service = get_case_service()
    try:
        case_data = case_service.create_case(
            title=title,
            description=description,
            case_id=case_id if case_id else None,
        )
        return jsonify({"status": "success", "case": case_data}), 201
    except ValueError as err:
        return jsonify({
            "error": "Validation error",
            "message": str(err),
        }), 400
    except Exception as err:
        return jsonify({
            "error": "Creation failed",
            "message": f"Failed to create case: {str(err)}",
        }), 500


@api_bp.route("/cases/<case_id>", methods=["GET"])
def get_case(case_id: str):
    """
    GET /api/cases/<case_id>
    Retrieve case details, all associated evidence, and consolidated timeline events.
    """
    case_service = get_case_service()
    try:
        case_data = case_service.get_case(case_id)
        if not case_data:
            return jsonify({
                "error": "Not found",
                "message": f"Case '{case_id}' does not exist.",
            }), 404
        return jsonify({"status": "success", "case": case_data}), 200
    except ValueError as err:
        return jsonify({
            "error": "Validation error",
            "message": str(err),
        }), 400
    except Exception as err:
        return jsonify({
            "error": "Retrieval error",
            "message": f"Failed to retrieve case: {str(err)}",
        }), 500


@api_bp.route("/cases/<case_id>/export", methods=["GET"])
def export_case_json(case_id: str):
    """
    GET /api/cases/<case_id>/export
    Export comprehensive verified case metadata, modality findings, consolidated timeline,
    and multimodal fusion relationships for forensic audit and record preservation.
    """
    case_service = get_case_service()
    try:
        case_data = case_service.get_case(case_id)
        if not case_data:
            return jsonify({
                "error": "Not found",
                "message": f"Case '{case_id}' does not exist.",
            }), 404

        fusion = case_data.get("fusion")
        if not fusion and case_data.get("evidence"):
            try:
                fusion_service = get_fusion_service()
                fusion = fusion_service.fuse_case(case_data)
                case_service.save_case_fusion(case_id, fusion)
                case_data["fusion"] = fusion
            except Exception:
                pass

        from datetime import datetime, timezone
        export_payload = {
            "case_id": case_data.get("case_id"),
            "title": case_data.get("title"),
            "description": case_data.get("description"),
            "created_at": case_data.get("created_at"),
            "updated_at": case_data.get("updated_at"),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "evidence_count": len(case_data.get("evidence", [])),
            "evidence": case_data.get("evidence", []),
            "timeline_events": case_data.get("timeline_events", []),
            "fusion": case_data.get("fusion"),
            "forensic_metadata": {
                "system": "CrimeVisionAI Multimodal Forensic Analysis Platform",
                "version": "1.0-epic6",
                "audit_notice": (
                    "Automated model predictions (speech recognition, object detection, "
                    "scene classification, automated cross-referencing, and AI hypotheses) "
                    "are probabilistic interpretations and do not constitute verified facts. "
                    "Independent human verification against original raw evidence is strictly required."
                ),
            },
        }
        return jsonify({"status": "success", "export": export_payload}), 200
    except ValueError as err:
        return jsonify({
            "error": "Validation error",
            "message": str(err),
        }), 400
    except Exception as err:
        return jsonify({
            "error": "Export error",
            "message": f"Failed to export case: {str(err)}",
        }), 500


@api_bp.route("/cases/<case_id>/export/pdf", methods=["GET"])
@api_bp.route("/cases/<case_id>/report.pdf", methods=["GET"])
def export_case_pdf(case_id: str):
    """
    GET /api/cases/<case_id>/export/pdf
    Compile and download an official, professional A4 forensic investigative PDF report for the case.
    """
    case_service = get_case_service()
    try:
        case_data = case_service.get_case(case_id)
        if not case_data:
            return jsonify({
                "error": "Not found",
                "message": f"Case '{case_id}' does not exist.",
            }), 404

        # If fusion hasn't been generated yet, attempt to fuse if evidence exists
        fusion = case_data.get("fusion")
        if not fusion and case_data.get("evidence"):
            try:
                fusion_service = get_fusion_service()
                fusion = fusion_service.fuse_case(case_data)
                case_service.save_case_fusion(case_id, fusion)
                case_data["fusion"] = fusion
            except Exception:
                pass

        pdf_service = get_pdf_service()
        pdf_bytes = pdf_service.generate_report(case_data)

        clean_id = secure_filename(case_id).replace(" ", "_") or "case"
        download_name = f"crimevision_{clean_id}_investigative_report.pdf"

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=download_name,
        )
    except ValueError as err:
        return jsonify({
            "error": "Validation error",
            "message": str(err),
        }), 400
    except Exception as err:
        current_app.logger.error(f"Failed to generate PDF report for case {case_id}: {err}", exc_info=True)
        return jsonify({
            "error": "PDF generation error",
            "message": f"Unable to generate forensic PDF report: {str(err)}",
        }), 500


@api_bp.route("/cases/<case_id>/fusion", methods=["GET"])
def get_case_fusion_endpoint(case_id: str):
    """
    GET /api/cases/<case_id>/fusion
    Retrieve structured multimodal fusion results for a case.
    If not yet generated, executes FusionService and persists the result.
    """
    case_service = get_case_service()
    case_data = case_service.get_case(case_id)
    if not case_data:
        return jsonify({
            "error": "Not found",
            "message": f"Case '{case_id}' does not exist.",
        }), 404

    fusion = case_data.get("fusion")
    if not fusion:
        fusion_service = get_fusion_service()
        fusion = fusion_service.fuse_case(case_data)
        case_service.save_case_fusion(case_id, fusion)

    return jsonify({"status": "success", "case_id": case_id, "fusion": fusion}), 200


@api_bp.route("/cases/<case_id>/fusion", methods=["POST"])
def run_case_fusion_endpoint(case_id: str):
    """
    POST /api/cases/<case_id>/fusion
    Execute or re-run multimodal fusion on case evidence.
    Accepts optional sync_groups in JSON payload:
    {"sync_groups": {"EVD-1": "common", "EVD-2": "common"}}
    """
    case_service = get_case_service()
    case_data = case_service.get_case(case_id)
    if not case_data:
        return jsonify({
            "error": "Not found",
            "message": f"Case '{case_id}' does not exist.",
        }), 404

    payload = request.get_json(silent=True) or {}
    sync_groups = payload.get("sync_groups")

    fusion_service = get_fusion_service()
    try:
        fusion = fusion_service.fuse_case(case_data, sync_groups=sync_groups)
        case_service.save_case_fusion(case_id, fusion)
        return jsonify({"status": "success", "case_id": case_id, "fusion": fusion}), 200
    except Exception as err:
        return jsonify({
            "error": "Fusion error",
            "message": f"Failed to execute cross-modal fusion: {str(err)}",
        }), 500


@api_bp.route("/cases/<case_id>/hypothesis", methods=["GET"])
def get_case_hypothesis_endpoint(case_id: str):
    """
    GET /api/cases/<case_id>/hypothesis
    Retrieve structured incident hypothesis for a case.
    If not yet generated, evaluates and persists it.
    Optional query parameter: ?refresh=true or ?generate=true
    """
    case_service = get_case_service()
    case_data = case_service.get_case(case_id)
    if not case_data:
        return jsonify({
            "error": "Not found",
            "message": f"Case '{case_id}' does not exist.",
        }), 404

    force_refresh = request.args.get("refresh", "false").lower() in ("true", "1")
    hyp = case_data.get("hypothesis")
    if not hyp or force_refresh:
        hyp_service = get_hypothesis_service()
        hyp = hyp_service.generate_case_hypothesis(case_data, force_refresh=force_refresh)

    return jsonify({"status": "success", "case_id": case_id, "hypothesis": hyp}), 200


@api_bp.route("/cases/<case_id>/hypothesis", methods=["POST"])
def run_case_hypothesis_endpoint(case_id: str):
    """
    POST /api/cases/<case_id>/hypothesis
    Explicitly synthesize case evidence and generate evidence-grounded candidate incident hypotheses.
    """
    case_service = get_case_service()
    case_data = case_service.get_case(case_id)
    if not case_data:
        return jsonify({
            "error": "Not found",
            "message": f"Case '{case_id}' does not exist.",
        }), 404

    hyp_service = get_hypothesis_service()
    try:
        hyp = hyp_service.generate_case_hypothesis(case_data, force_refresh=True)
        return jsonify({"status": "success", "case_id": case_id, "hypothesis": hyp}), 200
    except Exception as err:
        return jsonify({
            "error": "Hypothesis evaluation error",
            "message": f"Failed to generate incident hypothesis: {str(err)}",
        }), 500


@api_bp.route("/cases/<case_id>/evidence", methods=["POST"])
def upload_case_evidence(case_id: str):
    """
    POST /api/cases/<case_id>/evidence
    Upload an audio or image evidence file, associate it with the case,
    and optionally execute multimodal analysis immediately.
    """
    case_service = get_case_service()

    # 1. Verify case exists
    try:
        case_data = case_service.get_case(case_id)
        if not case_data:
            return jsonify({
                "error": "Not found",
                "message": f"Case '{case_id}' does not exist.",
            }), 404
    except ValueError as err:
        return jsonify({
            "error": "Validation error",
            "message": str(err),
        }), 400

    # 2. Validate file presence
    file = request.files.get("file") or request.files.get("evidence")
    if not file or not file.filename:
        return jsonify({
            "error": "Missing file",
            "message": "Please provide an evidence file in the 'file' or 'evidence' form field.",
        }), 400

    # 3. Detect and validate media type from extension
    ext = Path(file.filename).suffix.lower()
    media_type = request.form.get("media_type")

    is_audio = ext in AudioTranscriber.SUPPORTED_EXTENSIONS
    is_image = ext in SUPPORTED_IMAGE_EXTENSIONS
    is_video = ext in SUPPORTED_VIDEO_EXTENSIONS

    if not is_audio and not is_image and not is_video:
        supported_audio = ", ".join(sorted(AudioTranscriber.SUPPORTED_EXTENSIONS))
        supported_images = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        supported_video = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        return jsonify({
            "error": "Unsupported file format",
            "message": (
                f"Format '{ext}' is unsupported. Supported audio: {supported_audio}. "
                f"Supported images: {supported_images}. Supported video: {supported_video}."
            ),
        }), 400

    if is_audio:
        detected_media_type = "audio"
    elif is_image:
        detected_media_type = "image"
    else:
        detected_media_type = "video"

    if media_type and media_type not in ("audio", "image", "video"):
        return jsonify({
            "error": "Invalid parameter",
            "message": "media_type must be 'audio', 'image', or 'video'.",
        }), 400
    actual_media_type = media_type or detected_media_type

    # 4. Save upload safely into case folder
    upload_root = Path(current_app.config.get("UPLOAD_FOLDER", "evidence/uploads"))
    safe_case_dir = upload_root / "cases" / secure_filename(case_id)
    safe_case_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename)
    if not safe_name:
        safe_name = f"evidence_{uuid.uuid4().hex[:8]}{ext}"

    dest_path = safe_case_dir / safe_name
    try:
        file.save(str(dest_path))
    except Exception:
        return jsonify({
            "error": "Upload failed",
            "message": "Unable to save uploaded evidence file to disk.",
        }), 500

    # 5. Check for empty file
    file_size = dest_path.stat().st_size
    if file_size == 0:
        dest_path.unlink(missing_ok=True)
        return jsonify({
            "error": "Empty file",
            "message": "Uploaded evidence file is empty (0 bytes).",
        }), 400

    # 6. Record evidence in database
    try:
        evidence_record = case_service.add_evidence(
            case_id=case_id,
            original_filename=file.filename,
            media_type=actual_media_type,
            file_path=str(dest_path),
            file_size=file_size,
        )
    except Exception as err:
        return jsonify({
            "error": "Database error",
            "message": f"Failed to record evidence: {str(err)}",
        }), 500

    # 7. Optionally trigger analysis immediately (default: true if analyze param != 'false')
    should_analyze = request.form.get("analyze", "true").lower() in ("true", "1", "yes")
    analysis_result = None

    if should_analyze:
        try:
            if actual_media_type == "audio":
                audio_service = get_audio_service()
                language = request.form.get("language") or request.args.get("language")
                analysis_result = audio_service.process_audio(
                    audio_path=dest_path,
                    language=language,
                )
                case_service.save_analysis_result(
                    case_id=case_id,
                    evidence_id=evidence_record["evidence_id"],
                    analysis_type="audio",
                    result_data=analysis_result,
                )
            elif actual_media_type == "image":
                vision_service = get_vision_service()
                confidence_threshold = 0.25
                conf_param = request.form.get("confidence_threshold")
                if conf_param is not None:
                    try:
                        val = float(conf_param)
                        if 0.0 <= val <= 1.0:
                            confidence_threshold = val
                    except ValueError:
                        pass
                analysis_result = vision_service.analyze_image(
                    image_path=str(dest_path),
                    confidence_threshold=confidence_threshold,
                )
                case_service.save_analysis_result(
                    case_id=case_id,
                    evidence_id=evidence_record["evidence_id"],
                    analysis_type="image",
                    result_data=analysis_result,
                )
            elif actual_media_type == "video":
                video_service = get_video_service()
                confidence_threshold = 0.25
                conf_param = request.form.get("confidence_threshold")
                if conf_param is not None:
                    try:
                        val = float(conf_param)
                        if 0.0 <= val <= 1.0:
                            confidence_threshold = val
                    except ValueError:
                        pass
                analysis_result = video_service.analyze_video(
                    video_path=dest_path,
                    confidence_threshold=confidence_threshold,
                )
                case_service.save_analysis_result(
                    case_id=case_id,
                    evidence_id=evidence_record["evidence_id"],
                    analysis_type="video",
                    result_data=analysis_result,
                )
            evidence_record["status"] = "completed"
        except Exception as err:
            case_service.update_evidence_status(evidence_record["evidence_id"], "error")
            evidence_record["status"] = "error"
            # Return partial response with recorded evidence and error notice
            return jsonify({
                "status": "partial_success",
                "evidence": evidence_record,
                "analysis_error": f"Analysis failed: {str(err)}",
            }), 201

    return jsonify({
        "status": "success",
        "evidence": evidence_record,
        "analysis": analysis_result,
    }), 201


@api_bp.route("/cases/<case_id>/evidence/<evidence_id>/analyze", methods=["POST"])
def analyze_case_evidence(case_id: str, evidence_id: str):
    """
    POST /api/cases/<case_id>/evidence/<evidence_id>/analyze
    Execute or re-run multimodal analysis for an existing evidence record.
    """
    case_service = get_case_service()

    # 1. Lookup evidence
    evidence = case_service.get_evidence(evidence_id)
    if not evidence or evidence["case_id"] != case_id:
        return jsonify({
            "error": "Not found",
            "message": f"Evidence '{evidence_id}' not found for case '{case_id}'.",
        }), 404

    file_path = Path(evidence["file_path"])
    if not file_path.is_file():
        return jsonify({
            "error": "File missing",
            "message": "Evidence file not found on disk.",
        }), 404

    # 2. Execute analysis according to media type
    media_type = evidence["media_type"]
    try:
        if media_type == "audio":
            audio_service = get_audio_service()
            language = request.form.get("language") or request.args.get("language")
            result = audio_service.process_audio(
                audio_path=file_path,
                language=language,
            )
            case_service.save_analysis_result(
                case_id=case_id,
                evidence_id=evidence_id,
                analysis_type="audio",
                result_data=result,
            )
        elif media_type == "image":
            vision_service = get_vision_service()
            confidence_threshold = 0.25
            conf_param = request.form.get("confidence_threshold")
            if conf_param is not None:
                try:
                    val = float(conf_param)
                    if 0.0 <= val <= 1.0:
                        confidence_threshold = val
                except ValueError:
                    pass
            result = vision_service.analyze_image(
                image_path=str(file_path),
                confidence_threshold=confidence_threshold,
            )
            case_service.save_analysis_result(
                case_id=case_id,
                evidence_id=evidence_id,
                analysis_type="image",
                result_data=result,
            )
        elif media_type == "video":
            video_service = get_video_service()
            confidence_threshold = 0.25
            conf_param = request.form.get("confidence_threshold")
            if conf_param is not None:
                try:
                    val = float(conf_param)
                    if 0.0 <= val <= 1.0:
                        confidence_threshold = val
                except ValueError:
                    pass
            result = video_service.analyze_video(
                video_path=file_path,
                confidence_threshold=confidence_threshold,
            )
            case_service.save_analysis_result(
                case_id=case_id,
                evidence_id=evidence_id,
                analysis_type="video",
                result_data=result,
            )
        else:
            return jsonify({
                "error": "Unsupported media type",
                "message": f"Media type '{media_type}' is not supported for analysis.",
            }), 400

        return jsonify({
            "status": "success",
            "evidence_id": evidence_id,
            "analysis": result,
        }), 200
    except Exception as err:
        case_service.update_evidence_status(evidence_id, "error")
        return jsonify({
            "error": "Processing error",
            "message": f"Analysis failed: {str(err)}",
        }), 500


@api_bp.route("/evidence/<evidence_id>/media", methods=["GET"])
def get_evidence_media(evidence_id: str):
    """
    GET /api/evidence/<evidence_id>/media
    Safely stream or serve media for in-browser playback without exposing filesystem paths.
    """
    case_service = get_case_service()
    evidence = case_service.get_evidence(evidence_id)
    if not evidence:
        return jsonify({"error": "Not found", "message": "Evidence not found."}), 404

    file_path = Path(evidence["file_path"])
    if not file_path.is_file():
        return jsonify({"error": "File missing", "message": "Media file not found on disk."}), 404

    mimetype = None
    ext = file_path.suffix.lower()
    if ext == ".mp4":
        mimetype = "video/mp4"
    elif ext == ".webm":
        mimetype = "video/webm"
    elif ext == ".mov":
        mimetype = "video/quicktime"
    elif ext in (".wav", ".mp3", ".ogg"):
        mimetype = f"audio/{ext[1:]}"
    elif ext in (".jpg", ".jpeg"):
        mimetype = "image/jpeg"
    elif ext == ".png":
        mimetype = "image/png"

    return send_file(str(file_path), mimetype=mimetype)


def _extract_query_target(question: str) -> dict[str, Any]:
    """Backward-compatible wrapper delegating to ContextBuilder.extract_query_target."""
    return extract_query_target(question)


@api_bp.route("/cases/<case_id>/chat", methods=["POST"])
def case_chat_query(case_id: str):
    """
    POST /api/cases/<case_id>/chat
    Investigator Q&A grounded strictly in stored case findings.
    Uses token-budgeted ContextBuilder and sliding-window rate limit protection.
    """
    case_service = get_case_service()
    case_data = case_service.get_case(case_id)
    if not case_data:
        return jsonify({"error": "Not found", "message": f"Case '{case_id}' does not exist."}), 404

    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Validation error", "message": "Question cannot be empty."}), 400

    # Build token-efficient, question-targeted context (reserving room for output tokens)
    case_context = ContextBuilder.build_compact_context(case_data, question, max_input_tokens=4000)
    query_type = GroqService.classify_query_intent(question)
    timestamp_matches = len(case_context.get("primary_frame_observations", [])) or len(case_context.get("direct_timestamp_timeline_events", []))

    try:
        if current_app and "GROQ_SERVICE" in current_app.config:
            groq = current_app.config["GROQ_SERVICE"]
        else:
            groq = GroqService()

        if query_type == "hypothesis":
            result = groq.generate_hypotheses(case_context=case_context, question=question)
            return jsonify({
                "status": "success",
                "question": question,
                "query_type": "hypothesis",
                "hypotheses": result.get("hypotheses", []),
                "evidence_sufficiency": result.get("evidence_sufficiency", "unknown"),
                "overall_limitations": result.get("overall_limitations", ""),
                "note": result.get("note", ""),
                "model": result.get("model", ""),
                "truncated": result.get("truncated", False),
            }), 200
        else:
            result = groq.answer_case_question(case_context=case_context, question=question)
            return jsonify({
                "status": "success",
                "question": question,
                "query_type": query_type,
                "answer": result.get("answer", ""),
                "note": result.get(
                    "note",
                    "AI-generated response grounded in case records. Model predictions are unverified by human investigators.",
                ),
                "timestamp_matches": timestamp_matches,
            }), 200

    except GroqRateLimitExceeded as rle:
        current_app.logger.warning(
            f"Groq rate limit exceeded for case {case_id}: {rle}. Serving grounded deterministic answer."
        )
        fallback = ContextBuilder.build_deterministic_answer(
            case_data,
            question,
            focus=case_context.get("query_focus"),
            advisory_note=str(rle),
        )
        return jsonify({
            "status": "success",
            "question": question,
            "query_type": query_type,
            "answer": fallback.get("answer", ""),
            "note": fallback.get("note", ""),
            "rate_limited": True,
            "timestamp_matches": timestamp_matches,
        }), 200

    except Exception as err:
        err_msg = str(err)
        current_app.logger.error(f"Chat processing error for case {case_id}: {err_msg}", exc_info=True)
        # If rate limit or token limit was reported by provider or client
        if any(term in err_msg.lower() for term in ("rate limit", "429", "413", "tokens per minute", "tpm")):
            fallback = ContextBuilder.build_deterministic_answer(
                case_data,
                question,
                focus=case_context.get("query_focus"),
                advisory_note="Cloud LLM rate budget threshold reached. Grounded findings provided directly from verified case records.",
            )
            return jsonify({
                "status": "success",
                "question": question,
                "query_type": query_type,
                "answer": fallback.get("answer", ""),
                "note": fallback.get("note", ""),
                "rate_limited": True,
                "timestamp_matches": timestamp_matches,
            }), 200

        return jsonify({
            "error": "Chat processing error",
            "message": f"Unable to generate grounded answer: {err_msg}",
        }), 500



