"""
CrimeVisionAI - Forensic PDF Investigative Report Generation Service.
Implements Activity 6.3: Professional A4 forensic investigative PDF report.

Generates a concise, readable, publication-quality A4 PDF report with:
- Case overview & executive summary (concise, clear evidence counts & time distinctions)
- Evidence inventory table
- Modality-specific findings (Image, Audio, Video)
- Chronological forensic timeline (timestamped events only; untimed records summarized separately)
- Cross-modal evidence fusion & conflict audit
- Investigative hypotheses & uncertainty analysis
- Comprehensive forensic limitations & investigator disclaimer
"""

from collections import Counter
from datetime import datetime, timezone
import html
import io
import math
import re
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ===========================================================================
# 1. Numbered Canvas with Running Header & Footer
# ===========================================================================
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass ReportLab canvas to compute total page count and draw
    consistent running headers and footers with 'Page X of Y'.
    Uses neutral, objective, non-official investigative header/footer text.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []
        self.case_id: str = "CASE"

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 810, 559, 810)
            self.drawString(36, 814, "CrimeVisionAI — Investigative Evidence Report")
            case_text = f"Case ID: {self.case_id}"
            self.drawRightString(559, 814, case_text)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 38, 559, 38)
        self.drawString(36, 28, "CrimeVisionAI | AI-Assisted Investigative Report | Human Review Required")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 28, page_str)

        self.restoreState()


# ===========================================================================
# 2. PDF Report Generator Class
# ===========================================================================
class PdfReportService:
    """Service to compile and generate concise forensic case PDF investigative reports."""

    def __init__(self) -> None:
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        """Define consistent typography and palette matching forensic theme."""
        # Palette: Dark Navy (#0f172a, #1e293b), Slate (#475569), Charcoal (#1e293b)
        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                fontName="Helvetica-Bold",
                fontSize=17,
                leading=21,
                textColor=colors.HexColor("#0f172a"),
                spaceAfter=3,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ReportSubtitle",
                fontName="Helvetica",
                fontSize=9.0,
                leading=12,
                textColor=colors.HexColor("#475569"),
                spaceAfter=10,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                fontName="Helvetica-Bold",
                fontSize=11.0,
                leading=14,
                textColor=colors.HexColor("#0f172a"),
                spaceBefore=10,
                spaceAfter=4,
                keepWithNext=True,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SubSectionHeading",
                fontName="Helvetica-Bold",
                fontSize=9.0,
                leading=12,
                textColor=colors.HexColor("#1e293b"),
                spaceBefore=7,
                spaceAfter=3,
                keepWithNext=True,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ForensicBody",
                fontName="Helvetica",
                fontSize=8.0,
                leading=11,
                textColor=colors.HexColor("#1e293b"),
                spaceAfter=3,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ForensicBodyMuted",
                fontName="Helvetica",
                fontSize=7.5,
                leading=10,
                textColor=colors.HexColor("#64748b"),
                spaceAfter=3,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CalloutNotice",
                fontName="Helvetica",
                fontSize=7.0,
                leading=10,
                textColor=colors.HexColor("#854d0e"),
                spaceBefore=3,
                spaceAfter=3,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="TableHeader",
                fontName="Helvetica-Bold",
                fontSize=7.5,
                leading=10,
                textColor=colors.white,
                alignment=0,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="TableCell",
                fontName="Helvetica",
                fontSize=7.0,
                leading=9.5,
                textColor=colors.HexColor("#1e293b"),
                wordWrap="CJK",
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="TableCellBold",
                fontName="Helvetica-Bold",
                fontSize=7.0,
                leading=9.5,
                textColor=colors.HexColor("#0f172a"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="TableCellMuted",
                fontName="Helvetica",
                fontSize=6.5,
                leading=8.5,
                textColor=colors.HexColor("#64748b"),
            )
        )

    @staticmethod
    def _esc(text: Any) -> str:
        """Sanitize text for safe ReportLab XML Paragraph embedding."""
        if text is None:
            return "--"
        s = str(text).strip()
        if not s:
            return "--"
        return html.escape(s)

    @staticmethod
    def _clean_markdown_to_xml(raw_text: Any) -> str:
        """
        Convert markdown text into ReportLab Paragraph safe HTML/XML.
        Handles bold (**text**), bullet items (- or *), backticks, and unicode dashes.
        """
        if not raw_text:
            return ""
        text = str(raw_text).replace("\r\n", "\n").replace("\r", "\n")
        # Replace problematic unicode characters
        text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u202f", " ").replace("\u00a0", " ")
        text = text.replace("\u2248", "~")

        lines = text.split("\n")
        formatted = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Skip markdown table separators
            if re.match(r"^\|?[\s\-:]+\|[\s\-:|]+$", line_str):
                continue

            is_bullet = False
            if line_str.startswith(("- ", "* ", "• ")):
                is_bullet = True
                line_str = line_str[2:].strip()
            elif re.match(r"^\d+\.\s+", line_str):
                m = re.match(r"^(\d+\.)\s+(.*)$", line_str)
                if m:
                    is_bullet = True
                    line_str = f"<b>{m.group(1)}</b> {m.group(2)}"

            escaped = html.escape(line_str)
            # **bold** -> <b>bold</b>
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
            # *italic* -> <i>italic</i>
            escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
            # `code` -> <code>code</code>
            escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)

            if is_bullet:
                formatted.append(f"&bull; {escaped}")
            else:
                formatted.append(escaped)

        return "<br/>".join(formatted)

    @staticmethod
    def _format_bytes(size: Any) -> str:
        """Format byte counts into human-readable strings."""
        if not isinstance(size, (int, float)) or size <= 0:
            return "--"
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.2f} MB"

    def generate_report(self, case_data: dict[str, Any]) -> bytes:
        """
        Compile full forensic case data into a clean, concise multi-page A4 PDF document.

        Parameters
        ----------
        case_data : dict[str, Any]
            Full case dictionary including metadata, evidence items, timeline_events,
            and fusion data.

        Returns
        -------
        bytes
            Compiled binary PDF content.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=40,
            bottomMargin=42,
        )

        case_id = str(case_data.get("case_id", "UNKNOWN-CASE"))

        def canvas_maker(*args: Any, **kwargs: Any) -> NumberedCanvas:
            c = NumberedCanvas(*args, **kwargs)
            c.case_id = case_id
            return c

        story: list[Any] = []

        # Build each required section
        self._build_header(story, case_data)
        self._build_section_a_overview(story, case_data)
        self._build_section_b_evidence_inventory(story, case_data)
        self._build_section_c_image_findings(story, case_data)
        self._build_section_d_audio_findings(story, case_data)
        self._build_section_e_video_findings(story, case_data)
        self._build_section_f_unified_timeline(story, case_data)
        self._build_section_g_fusion_findings(story, case_data)
        self._build_section_h_hypotheses(story, case_data)
        self._build_section_i_limitations(story)

        doc.build(story, canvasmaker=canvas_maker)
        return buffer.getvalue()

    # -----------------------------------------------------------------------
    # Document Header & Title
    # -----------------------------------------------------------------------
    def _build_header(self, story: list[Any], case_data: dict[str, Any]) -> None:
        title = self._esc(case_data.get("title", "Forensic Investigation"))
        story.append(Paragraph("CrimeVisionAI — Investigative Evidence Report", self.styles["ReportTitle"]))
        story.append(
            Paragraph(
                f"Automated Multimodal Forensic Analysis &amp; Chronological Audit &nbsp;|&nbsp; <b>{title}</b>",
                self.styles["ReportSubtitle"],
            )
        )
        story.append(
            HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0f172a"), spaceBefore=0, spaceAfter=6)
        )

    # -----------------------------------------------------------------------
    # Section A: Case Overview & Executive Summary
    # -----------------------------------------------------------------------
    def _build_section_a_overview(self, story: list[Any], case_data: dict[str, Any]) -> None:
        story.append(Paragraph("A. Case Overview &amp; Executive Summary", self.styles["SectionHeading"]))

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        case_id = self._esc(case_data.get("case_id"))
        title = self._esc(case_data.get("title"))
        desc = self._esc(case_data.get("description", "No description recorded."))
        created = self._esc(case_data.get("created_at"))
        updated = self._esc(case_data.get("updated_at"))

        evidence_items = case_data.get("evidence", []) or []
        evidence_count = len(evidence_items)

        img_count = sum(1 for e in evidence_items if e.get("media_type") == "image")
        aud_count = sum(1 for e in evidence_items if e.get("media_type") == "audio")
        vid_count = sum(1 for e in evidence_items if e.get("media_type") == "video")

        fusion = case_data.get("fusion") or {}
        fusion_status = "Available" if fusion else "Not executed"
        all_events = fusion.get("events", []) if fusion.get("events") else (case_data.get("timeline_events", []) or [])
        total_events_count = len(all_events)

        timestamped_events = [
            e for e in all_events
            if (e.get("timestamp_normalized") is not None or e.get("start_raw") is not None)
        ]
        ts_count = len(timestamped_events)
        untimed_count = total_events_count - ts_count

        anchors_list = fusion.get("anchors", []) if fusion else [e for e in all_events if e.get("is_anchor")]
        anchor_count = len(anchors_list)
        rel_count = len(fusion.get("relationships", [])) if fusion else 0

        # Metadata Table (width: 523pt total)
        meta_data = [
            [
                Paragraph("<b>Case ID:</b>", self.styles["TableCellBold"]),
                Paragraph(case_id, self.styles["TableCell"]),
                Paragraph("<b>Report Generated:</b>", self.styles["TableCellBold"]),
                Paragraph(now_utc, self.styles["TableCell"]),
            ],
            [
                Paragraph("<b>Case Title:</b>", self.styles["TableCellBold"]),
                Paragraph(title, self.styles["TableCell"]),
                Paragraph("<b>Record Created:</b>", self.styles["TableCellBold"]),
                Paragraph(created, self.styles["TableCell"]),
            ],
            [
                Paragraph("<b>Evidence Recorded:</b>", self.styles["TableCellBold"]),
                Paragraph(f"{evidence_count} file(s) ({vid_count} video, {img_count} image, {aud_count} audio)", self.styles["TableCell"]),
                Paragraph("<b>Record Last Updated:</b>", self.styles["TableCellBold"]),
                Paragraph(updated, self.styles["TableCell"]),
            ],
            [
                Paragraph("<b>Timeline / Fusion:</b>", self.styles["TableCellBold"]),
                Paragraph(
                    f"{total_events_count} total entries ({ts_count} timestamped, {untimed_count} untimed; {anchor_count} anchors)",
                    self.styles["TableCell"],
                ),
                Paragraph("<b>Cross-Modal Status:</b>", self.styles["TableCellBold"]),
                Paragraph(f"{fusion_status} ({rel_count} relationships)", self.styles["TableCell"]),
            ],
        ]

        t_meta = Table(meta_data, colWidths=[95, 166, 95, 167])
        t_meta.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(t_meta)
        story.append(Spacer(1, 4))

        # Concise Executive Summary
        story.append(Paragraph("<b>Case Scope &amp; Executive Summary:</b>", self.styles["SubSectionHeading"]))
        story.append(Paragraph(desc, self.styles["ForensicBody"]))

        # Build concise bulleted findings summary
        modality_bullets = []
        if vid_count > 0:
            modality_bullets.append(
                f"<b>Video Evidence ({vid_count} file):</b> Extracted {ts_count} time-stamped video observation(s) "
                f"indexed at video-relative offsets. Key person transitions were identified as candidate chronological anchors."
            )
        if img_count > 0:
            modality_bullets.append(
                f"<b>Image Evidence ({img_count} file(s)):</b> Identified {untimed_count} object detection observation(s). "
                f"Image evidence operates with <i>unknown incident times</i> and is analyzed independently from video timestamps."
            )
        if aud_count > 0:
            modality_bullets.append(
                f"<b>Audio Evidence ({aud_count} file(s)):</b> Processed acoustic events and transcripts."
            )
        else:
            modality_bullets.append(
                "<b>Audio Evidence:</b> No audio evidence was submitted for this case."
            )

        summary_html = "<br/>&bull; ".join([""] + modality_bullets)
        story.append(Paragraph(f"<b>Modality Summary:</b>{summary_html}", self.styles["ForensicBody"]))

        # Concise AI Advisory
        advisory_text = (
            "<b>NOTICE OF AI-ASSISTED FORENSIC DECISION SUPPORT:</b> This report compiles automated model predictions "
            "(RT-DETR object detection, CLIP scene classification, BLIP captioning, and LLM reasoning). "
            "Detections and hypotheses are probabilistic interpretations, not certified forensic facts. "
            "Independent human review of raw evidence is strictly required prior to investigative or legal decisions."
        )
        t_advisory = Table([[Paragraph(advisory_text, self.styles["CalloutNotice"])]], colWidths=[523])
        t_advisory.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fefce8")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#fde047")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(Spacer(1, 3))
        story.append(t_advisory)
        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section B: Evidence Inventory
    # -----------------------------------------------------------------------
    def _build_section_b_evidence_inventory(self, story: list[Any], case_data: dict[str, Any]) -> None:
        story.append(Paragraph("B. Evidence Inventory", self.styles["SectionHeading"]))

        evidence_items = case_data.get("evidence", []) or []
        if not evidence_items:
            story.append(Paragraph("<i>No evidence files have been recorded for this case yet.</i>", self.styles["ForensicBody"]))
            story.append(Spacer(1, 6))
            return

        table_data = [
            [
                Paragraph("Evidence ID", self.styles["TableHeader"]),
                Paragraph("Filename / Source", self.styles["TableHeader"]),
                Paragraph("Modality", self.styles["TableHeader"]),
                Paragraph("File Size", self.styles["TableHeader"]),
                Paragraph("Ingested At (UTC)", self.styles["TableHeader"]),
                Paragraph("Status", self.styles["TableHeader"]),
            ]
        ]

        for ev in evidence_items:
            ev_id = self._esc(ev.get("evidence_id"))
            fname = self._esc(ev.get("original_filename"))
            mtype = (ev.get("media_type") or "unknown").upper()
            fsize = self._format_bytes(ev.get("file_size"))
            uploaded = self._esc(ev.get("uploaded_at"))
            status = (ev.get("status") or "pending").capitalize()

            table_data.append([
                Paragraph(f"<b>{ev_id}</b>", self.styles["TableCellBold"]),
                Paragraph(fname, self.styles["TableCell"]),
                Paragraph(mtype, self.styles["TableCell"]),
                Paragraph(fsize, self.styles["TableCell"]),
                Paragraph(uploaded, self.styles["TableCell"]),
                Paragraph(status, self.styles["TableCell"]),
            ])

        t_inv = Table(table_data, colWidths=[75, 158, 50, 50, 120, 70], repeatRows=1)
        t_inv.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(t_inv)
        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section C: Image Analysis Findings (Concise, Formatted & Summarized)
    # -----------------------------------------------------------------------
    def _build_section_c_image_findings(self, story: list[Any], case_data: dict[str, Any]) -> None:
        story.append(Paragraph("C. Image Analysis Findings", self.styles["SectionHeading"]))
        images = [e for e in case_data.get("evidence", []) if e.get("media_type") == "image"]

        if not images:
            story.append(Paragraph("<i>No image evidence provided or recorded for this case.</i>", self.styles["ForensicBodyMuted"]))
            story.append(Spacer(1, 6))
            return

        for img in images:
            ev_id = self._esc(img.get("evidence_id"))
            fname = self._esc(img.get("original_filename"))
            analysis = img.get("analysis") or {}

            story.append(Paragraph(f"<b>Image Source:</b> {fname} (Evidence ID: <code>{ev_id}</code>)", self.styles["SubSectionHeading"]))

            if not analysis:
                story.append(Paragraph("<i>Analysis pending or not available for this image.</i>", self.styles["ForensicBodyMuted"]))
                continue

            obj_det = analysis.get("object_detection") or {}
            detections = obj_det.get("detections", []) or []
            caption_data = analysis.get("image_caption") or {}
            scene_data = analysis.get("scene_classification") or {}
            llm_analysis = analysis.get("llm_analysis") or {}

            # Scene and Caption Summary
            top_scene = self._esc(scene_data.get("top_scene"))
            caption = self._esc(caption_data.get("caption"))
            meta_p = f"<b>Scene Classification (CLIP):</b> {top_scene} &nbsp;|&nbsp; <b>AI Caption (BLIP):</b> &ldquo;{caption}&rdquo;"
            story.append(Paragraph(meta_p, self.styles["ForensicBody"]))

            # Summarized Object Detections (Grouping by label for concise, non-repetitive reading)
            if detections:
                counts = Counter(d.get("label", "object") for d in detections)
                det_rows = [
                    [
                        Paragraph("Detected Label", self.styles["TableHeader"]),
                        Paragraph("Count", self.styles["TableHeader"]),
                        Paragraph("Confidence Range", self.styles["TableHeader"]),
                        Paragraph("Sample Bounding Box [x1, y1, x2, y2]", self.styles["TableHeader"]),
                    ]
                ]
                for label, cnt in counts.most_common():
                    label_dets = [d for d in detections if d.get("label") == label]
                    confs = [float(d.get("confidence", 0)) for d in label_dets if d.get("confidence") is not None]
                    conf_str = f"{min(confs):.2f} - {max(confs):.2f}" if confs else "--"

                    # Map bounding box correctly from bbox_xyxy or box
                    first_det = label_dets[0]
                    bbox = first_det.get("bbox_xyxy") or first_det.get("box") or first_det.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        box_str = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
                    else:
                        box_str = "--"

                    det_rows.append([
                        Paragraph(f"<b>{self._esc(label)}</b>", self.styles["TableCellBold"]),
                        Paragraph(str(cnt), self.styles["TableCell"]),
                        Paragraph(conf_str, self.styles["TableCell"]),
                        Paragraph(box_str, self.styles["TableCellMuted"]),
                    ])

                t_det = Table(det_rows, colWidths=[130, 45, 110, 238], repeatRows=1)
                t_det.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ])
                )
                story.append(t_det)
                story.append(Spacer(1, 3))
            else:
                story.append(Paragraph("<i>No discrete objects detected above threshold.</i>", self.styles["ForensicBodyMuted"]))

            # Cleanly rendered LLM interpretation (no raw markdown **)
            raw_llm = llm_analysis.get("analysis") if isinstance(llm_analysis, dict) else None
            if raw_llm:
                formatted_llm = self._clean_markdown_to_xml(raw_llm)
                story.append(
                    Paragraph(
                        f"<b>AI Interpretation &amp; Context:</b><br/>{formatted_llm}",
                        self.styles["ForensicBodyMuted"],
                    )
                )

            story.append(Spacer(1, 5))

    # -----------------------------------------------------------------------
    # Section D: Audio Analysis Findings
    # -----------------------------------------------------------------------
    def _build_section_d_audio_findings(self, story: list[Any], case_data: dict[str, Any]) -> None:
        audio_items = [e for e in case_data.get("evidence", []) if e.get("media_type") == "audio"]

        if not audio_items:
            # If no audio evidence, render a clean, brief note without empty placeholder sections
            story.append(Paragraph("D. Audio Analysis Findings", self.styles["SectionHeading"]))
            story.append(Paragraph("<i>No audio evidence was submitted for this case.</i>", self.styles["ForensicBodyMuted"]))
            story.append(Spacer(1, 6))
            return

        story.append(Paragraph("D. Audio Analysis Findings", self.styles["SectionHeading"]))
        for aud in audio_items:
            ev_id = self._esc(aud.get("evidence_id"))
            fname = self._esc(aud.get("original_filename"))
            analysis = aud.get("analysis") or {}

            story.append(Paragraph(f"<b>Audio Source:</b> {fname} (Evidence ID: <code>{ev_id}</code>)", self.styles["SubSectionHeading"]))

            if not analysis:
                story.append(Paragraph("<i>Audio analysis pending or not available.</i>", self.styles["ForensicBodyMuted"]))
                continue

            trans = analysis.get("transcription") or {}
            transcript_text = self._esc(trans.get("transcript", "(No speech detected)"))
            lang = self._esc(trans.get("detected_language") or trans.get("language_hint") or "Auto/Unknown")
            dur = f"{float(analysis.get('recording_duration_seconds', 0)):.1f}s" if analysis.get("recording_duration_seconds") else "--"

            story.append(
                Paragraph(
                    f"<b>Duration:</b> {dur} &nbsp;|&nbsp; <b>Detected Language:</b> {lang} &nbsp;|&nbsp; "
                    f"<b>Time Basis:</b> Audio-relative seconds",
                    self.styles["ForensicBody"],
                )
            )
            story.append(Paragraph(f"<b>Full Spoken Transcript:</b> &ldquo;{transcript_text}&rdquo;", self.styles["ForensicBody"]))

            # Acoustic Segments
            segments = trans.get("segments", []) or []
            if segments:
                story.append(Paragraph("<b>Timestamped Acoustic Segments:</b>", self.styles["ForensicBodyMuted"]))
                seg_rows = [
                    [
                        Paragraph("Span (s)", self.styles["TableHeader"]),
                        Paragraph("Transcribed Utterance", self.styles["TableHeader"]),
                    ]
                ]
                for seg in segments[:10]:
                    s = f"{float(seg.get('start', 0)):.1f}s" if seg.get("start") is not None else "--"
                    e = f"{float(seg.get('end', 0)):.1f}s" if seg.get("end") is not None else "--"
                    text = self._esc(seg.get("text", ""))
                    seg_rows.append([
                        Paragraph(f"{s} - {e}", self.styles["TableCellBold"]),
                        Paragraph(text, self.styles["TableCell"]),
                    ])
                t_seg = Table(seg_rows, colWidths=[80, 443], repeatRows=1)
                t_seg.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ])
                )
                story.append(t_seg)

            story.append(Spacer(1, 5))

    # -----------------------------------------------------------------------
    # Section E: Video Surveillance Findings
    # -----------------------------------------------------------------------
    def _build_section_e_video_findings(self, story: list[Any], case_data: dict[str, Any]) -> None:
        video_items = [e for e in case_data.get("evidence", []) if e.get("media_type") == "video"]

        if not video_items:
            story.append(Paragraph("E. Video Surveillance Findings", self.styles["SectionHeading"]))
            story.append(Paragraph("<i>No video surveillance evidence recorded for this case.</i>", self.styles["ForensicBodyMuted"]))
            story.append(Spacer(1, 6))
            return

        story.append(Paragraph("E. Video Surveillance Findings", self.styles["SectionHeading"]))
        for vid in video_items:
            ev_id = self._esc(vid.get("evidence_id"))
            fname = self._esc(vid.get("original_filename"))
            analysis = vid.get("analysis") or {}

            story.append(Paragraph(f"<b>Video Source:</b> {fname} (Evidence ID: <code>{ev_id}</code>)", self.styles["SubSectionHeading"]))

            if not analysis:
                story.append(Paragraph("<i>Video analysis pending or not available.</i>", self.styles["ForensicBodyMuted"]))
                continue

            dur = f"{float(analysis.get('duration_seconds', 0)):.1f}s" if analysis.get("duration_seconds") else "--"
            fps = f"{float(analysis.get('sampled_fps', 1.0)):.1f}" if analysis.get("sampled_fps") else "1.0"

            # Check actual video incident hypothesis correctly without stale placeholders
            hyp = analysis.get("incident_hypothesis") or {}
            hyp_type = hyp.get("type") or hyp.get("incident_type_hypothesis")
            hyp_reason = hyp.get("reasoning") or hyp.get("incident_hypothesis_reasoning")

            if hyp_type and hyp_type != "unknown":
                hyp_type_display = self._esc(hyp_type)
            else:
                hyp_type_display = "No specific incident pattern classified"

            if hyp_reason and "awaiting" not in hyp_reason.lower():
                hyp_reason_display = self._esc(hyp_reason)
            elif analysis.get("frame_observations"):
                hyp_reason_display = "Observations processed across sampled frames; no high-confidence incident signatures detected."
            else:
                hyp_reason_display = "Incident classification unavailable."

            story.append(
                Paragraph(
                    f"<b>Duration:</b> {dur} &nbsp;|&nbsp; <b>Sampling Rate:</b> {fps} FPS &nbsp;|&nbsp; "
                    f"<b>Time Basis:</b> Video-relative seconds",
                    self.styles["ForensicBody"],
                )
            )

            # Hypothesis banner
            story.append(
                Paragraph(
                    f"<b>Automated Classification:</b> <code>{hyp_type_display}</code><br/>"
                    f"<b>Reasoning / Status:</b> {hyp_reason_display}",
                    self.styles["ForensicBody"],
                )
            )

            # Sampled frame observations table (compact)
            frames = analysis.get("frame_observations", []) or []
            if frames:
                story.append(Paragraph("<b>Sampled Frame Observations (1 FPS):</b>", self.styles["ForensicBodyMuted"]))
                f_rows = [
                    [
                        Paragraph("Time", self.styles["TableHeader"]),
                        Paragraph("Frame", self.styles["TableHeader"]),
                        Paragraph("Persons", self.styles["TableHeader"]),
                        Paragraph("Detected Objects (RT-DETR)", self.styles["TableHeader"]),
                        Paragraph("Scene / Caption (BLIP)", self.styles["TableHeader"]),
                    ]
                ]
                for f in frames[:10]:
                    ts = f"{float(f.get('timestamp_seconds', 0)):.1f}s" if f.get("timestamp_seconds") is not None else "--"
                    idx = str(f.get("frame_index", "--"))
                    pcount = str(f.get("person_count", 0))

                    # Format detections compactly
                    dets = f.get("detections", [])
                    if dets:
                        # Count labels
                        lbl_counts = Counter(d.get("label", "obj") for d in dets)
                        det_summary = ", ".join(f"{cnt} {lbl}" if cnt > 1 else lbl for lbl, cnt in lbl_counts.most_common(4))
                    else:
                        det_summary = "None"

                    cap = self._esc(f.get("caption") or f.get("scene") or "--")
                    f_rows.append([
                        Paragraph(ts, self.styles["TableCellBold"]),
                        Paragraph(idx, self.styles["TableCellMuted"]),
                        Paragraph(pcount, self.styles["TableCell"]),
                        Paragraph(self._esc(det_summary), self.styles["TableCell"]),
                        Paragraph(cap, self.styles["TableCell"]),
                    ])

                t_frames = Table(f_rows, colWidths=[45, 38, 42, 180, 218], repeatRows=1)
                t_frames.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ])
                )
                story.append(t_frames)
                if len(frames) > 10:
                    story.append(
                        Paragraph(
                            f"<i>... ({len(frames) - 10} additional sampled frames omitted for report brevity).</i>",
                            self.styles["ForensicBodyMuted"],
                        )
                    )

            # Video Caveat
            story.append(
                Paragraph(
                    "<i>Limitation: RT-DETR detects objects per frame. It does not perform continuous biometric tracking "
                    "or verify physical entry/exit over time.</i>",
                    self.styles["ForensicBodyMuted"],
                )
            )
            story.append(Spacer(1, 5))

    # -----------------------------------------------------------------------
    # Section F: Unified Event Timeline (Timestamped Events Only + Untimed Summary)
    # -----------------------------------------------------------------------
    def _build_section_f_unified_timeline(self, story: list[Any], case_data: dict[str, Any]) -> None:
        story.append(Paragraph("F. Unified Forensic Timeline", self.styles["SectionHeading"]))
        story.append(
            Paragraph(
                "Chronological compilation of time-stamped anchor events derived from video surveillance and audio analysis. "
                "<b>Notice:</b> Timestamps reflect recording-relative offsets unless multi-camera synchronization is established.",
                self.styles["ForensicBodyMuted"],
            )
        )

        fusion = case_data.get("fusion") or {}
        all_events = (fusion.get("events") if fusion.get("events") else None) or case_data.get("timeline_events", []) or []

        # Filter strictly for timestamped events
        timestamped_events = []
        untimed_events = []

        for ev in all_events:
            raw_s = ev.get("timestamp_normalized")
            if raw_s is None:
                raw_s = ev.get("start_raw")
            if raw_s is None:
                raw_s = ev.get("timestamp_seconds")

            if raw_s is not None and isinstance(raw_s, (int, float)):
                timestamped_events.append(ev)
            else:
                untimed_events.append(ev)

        if not timestamped_events:
            story.append(Paragraph("<i>No timestamped timeline events currently recorded for this case.</i>", self.styles["ForensicBodyMuted"]))
        else:
            table_data = [
                [
                    Paragraph("Event ID", self.styles["TableHeader"]),
                    Paragraph("Time Span", self.styles["TableHeader"]),
                    Paragraph("Time Basis", self.styles["TableHeader"]),
                    Paragraph("Modality / Source", self.styles["TableHeader"]),
                    Paragraph("Event Description", self.styles["TableHeader"]),
                    Paragraph("Score / Anchor", self.styles["TableHeader"]),
                ]
            ]

            for ev in timestamped_events:
                eid = self._esc(ev.get("event_id") or "EVT")

                raw_s = ev.get("timestamp_normalized")
                if raw_s is None:
                    raw_s = ev.get("start_raw")
                if raw_s is None:
                    raw_s = ev.get("timestamp_seconds")

                raw_e = ev.get("end_timestamp_normalized")
                if raw_e is None:
                    raw_e = ev.get("end_raw")
                if raw_e is None:
                    raw_e = ev.get("end_timestamp_seconds")

                s_str = f"{float(raw_s):.1f}s" if raw_s is not None and isinstance(raw_s, (int, float)) else "Unknown"
                e_str = f"{float(raw_e):.1f}s" if raw_e is not None and isinstance(raw_e, (int, float)) else s_str
                span = f"{s_str} - {e_str}" if s_str != e_str else s_str

                # Modality and Source mapping with proper fallbacks
                modality = (ev.get("modality") or ev.get("media_type") or "event").upper()
                source_file = ev.get("original_filename") or ev.get("source_filename") or ev.get("source_evidence_id") or "--"
                modality_source = f"{modality}<br/>{self._esc(source_file)}"

                tbasis = ev.get("original_time_basis")
                if not tbasis:
                    tbasis = f"{modality.lower()}-relative" if s_str != "Unknown" else "unknown"

                desc = self._esc(ev.get("description", ""))

                # Score mapping with proper fallbacks
                score_val = ev.get("evidence_consistency_score")
                if score_val is None:
                    score_val = ev.get("certainty_score")

                score_str = f"{float(score_val):.2f}" if score_val is not None and isinstance(score_val, (int, float)) else "--"
                if ev.get("is_anchor") or ev.get("anchors"):
                    score_str += "<br/><b>(Anchor)</b>"

                table_data.append([
                    Paragraph(f"<b>{eid}</b>", self.styles["TableCellBold"]),
                    Paragraph(span, self.styles["TableCellBold"]),
                    Paragraph(self._esc(tbasis), self.styles["TableCellMuted"]),
                    Paragraph(modality_source, self.styles["TableCellMuted"]),
                    Paragraph(desc, self.styles["TableCell"]),
                    Paragraph(score_str, self.styles["TableCell"]),
                ])

            t_time = Table(table_data, colWidths=[55, 65, 80, 100, 163, 60], repeatRows=1)
            t_time.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ])
            )
            story.append(t_time)

        # Separate concise summary of untimed observations (prevents 40-row image dumps)
        if untimed_events:
            story.append(Spacer(1, 3))
            untimed_by_file: dict[str, int] = Counter(
                e.get("original_filename") or e.get("source_filename") or "Unknown Image"
                for e in untimed_events
            )
            summary_parts = [f"{cnt} detections from {fname}" for fname, cnt in untimed_by_file.items()]
            story.append(
                Paragraph(
                    f"<b>Untimed Evidence Observations ({len(untimed_events)} total):</b> "
                    f"{'; '.join(summary_parts)}. "
                    f"<i>These image-derived records lack timestamp baselines and are detailed by file in Section C.</i>",
                    self.styles["ForensicBodyMuted"],
                )
            )

        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section G: Evidence Fusion Findings
    # -----------------------------------------------------------------------
    def _build_section_g_fusion_findings(self, story: list[Any], case_data: dict[str, Any]) -> None:
        story.append(Paragraph("G. Cross-Modal Evidence Fusion Findings", self.styles["SectionHeading"]))
        fusion = case_data.get("fusion") or {}

        if not fusion:
            story.append(Paragraph("<i>Cross-modal evidence fusion has not been executed for this case.</i>", self.styles["ForensicBodyMuted"]))
            story.append(Spacer(1, 6))
            return

        align = fusion.get("alignment_status", "unknown")
        align_label = "Aligned (Synchronized Common Timeline)" if align == "aligned" else "Independent Time Bases (Unsynchronized)"
        anchors_count = len(fusion.get("anchors", []))
        raw_narrative = fusion.get("grounded_narrative", "Synthesis narrative unavailable.")

        # Clean narrative text if present
        clean_narrative = self._clean_markdown_to_xml(raw_narrative)

        story.append(
            Paragraph(
                f"<b>Temporal Alignment Status:</b> {align_label} &nbsp;|&nbsp; <b>Chronological Anchors:</b> {anchors_count}",
                self.styles["ForensicBody"],
            )
        )
        story.append(Paragraph(f"<b>Grounded Multimodal Narrative:</b><br/>{clean_narrative}", self.styles["ForensicBody"]))

        # Cross-modal relationships & conflicts
        relationships = fusion.get("relationships", []) or []
        if relationships:
            story.append(Paragraph("<b>Cross-Modal Relationships &amp; Conflict Audit:</b>", self.styles["SubSectionHeading"]))
            rel_rows = [
                [
                    Paragraph("Link ID", self.styles["TableHeader"]),
                    Paragraph("Type", self.styles["TableHeader"]),
                    Paragraph("Sources &amp; Events", self.styles["TableHeader"]),
                    Paragraph("Description &amp; Observations", self.styles["TableHeader"]),
                    Paragraph("Caveats / Limitations", self.styles["TableHeader"]),
                ]
            ]
            for r in relationships:
                rid = self._esc(r.get("rel_id") or r.get("relationship_id") or "REL")
                rtype = (r.get("relationship_type") or "relationship").replace("_", " ").upper()
                sources = ", ".join(r.get("evidence_sources") or r.get("source_event_ids") or [])
                desc = self._esc(r.get("description") or r.get("explanation") or "--")
                caveats = r.get("caveats") or r.get("uncertainties") or []
                if isinstance(caveats, list):
                    cav_str = "; ".join(caveats) if caveats else "None specified"
                else:
                    cav_str = str(caveats)

                rel_rows.append([
                    Paragraph(f"<b>{rid}</b>", self.styles["TableCellBold"]),
                    Paragraph(rtype, self.styles["TableCellBold"]),
                    Paragraph(self._esc(sources), self.styles["TableCellMuted"]),
                    Paragraph(desc, self.styles["TableCell"]),
                    Paragraph(self._esc(cav_str), self.styles["TableCellMuted"]),
                ])

            t_rel = Table(rel_rows, colWidths=[55, 75, 85, 178, 130], repeatRows=1)
            t_rel.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ])
            )
            story.append(t_rel)

        story.append(
            Paragraph(
                "<i>Forensic Notice: Cross-modal consistency scores and corroboration links are automated heuristics. "
                "Coincident detections or semantic overlaps do not constitute definitive proof that two observations "
                "concern the same entity, person, or physical action.</i>",
                self.styles["ForensicBodyMuted"],
            )
        )
        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section H: Case Hypotheses & Uncertainty
    # -----------------------------------------------------------------------
    def _build_section_h_hypotheses(self, story: list[Any], case_data: dict[str, Any]) -> None:
        story.append(Paragraph("H. Investigative Hypotheses &amp; Uncertainty", self.styles["SectionHeading"]))

        fusion = case_data.get("fusion") or {}
        hypotheses = fusion.get("hypotheses", []) or []

        # If not in fusion, check video incident hypotheses
        if not hypotheses:
            for ev in case_data.get("evidence", []):
                if ev.get("media_type") == "video":
                    v_hyp = (ev.get("analysis") or {}).get("incident_hypothesis")
                    if v_hyp and v_hyp.get("incident_type_hypothesis") and v_hyp.get("incident_type_hypothesis") != "unknown":
                        hypotheses.append({
                            "title": f"Video Event Hypothesis ({ev.get('original_filename')})",
                            "narrative": v_hyp.get("incident_hypothesis_reasoning", ""),
                            "plausibility": "moderate",
                            "supporting_evidence": [f"Incident classification: {v_hyp.get('incident_type_hypothesis')}"],
                            "uncertainty_factors": ["Derived from video surveillance sampling"],
                        })

        if not hypotheses:
            story.append(
                Paragraph(
                    "<i>No structured case-level hypotheses are currently stored for this case. "
                    "Unverified image interpretations (detailed in Section C) operate at the single-image level "
                    "and do not constitute case-wide hypotheses. Investigators may formulate grounded hypotheses "
                    "via the interactive Q&amp;A Assistant.</i>",
                    self.styles["ForensicBodyMuted"],
                )
            )
            story.append(Spacer(1, 6))
            return

        for idx, h in enumerate(hypotheses[:3], 1):
            title = self._esc(h.get("title", f"Hypothesis {idx}"))
            plaus = (h.get("plausibility") or "unrated").upper()
            narr = self._clean_markdown_to_xml(h.get("narrative", ""))

            story.append(Paragraph(f"<b>{idx}. {title}</b> (Plausibility: <code>{plaus}</code>)", self.styles["SubSectionHeading"]))
            if narr:
                story.append(Paragraph(narr, self.styles["ForensicBody"]))

            supp = h.get("supporting_evidence") or []
            if supp:
                supp_txt = " &bull; " + "<br/>&bull; ".join(self._esc(s) for s in supp)
                story.append(Paragraph(f"<b>Supporting Evidence:</b><br/>{supp_txt}", self.styles["ForensicBodyMuted"]))

            contra = h.get("contradicting_evidence") or []
            if contra and contra != ["None identified in available evidence"]:
                contra_txt = " &bull; " + "<br/>&bull; ".join(self._esc(c) for c in contra)
                story.append(Paragraph(f"<b>Contradicting / Missing Evidence:</b><br/>{contra_txt}", self.styles["ForensicBodyMuted"]))

            assump = h.get("assumptions") or []
            if assump:
                assump_txt = " &bull; " + "<br/>&bull; ".join(self._esc(a) for a in assump)
                story.append(Paragraph(f"<b>Key Assumptions:</b><br/>{assump_txt}", self.styles["ForensicBodyMuted"]))

            uncert = h.get("uncertainty_factors") or []
            if uncert:
                uncert_txt = " &bull; " + "<br/>&bull; ".join(self._esc(u) for u in uncert)
                story.append(Paragraph(f"<b>Uncertainty Factors:</b><br/>{uncert_txt}", self.styles["ForensicBodyMuted"]))

            story.append(Spacer(1, 4))

        story.append(
            Paragraph(
                "<i>Forensic Disclaimer: Hypotheses are probabilistic interpretations to assist human investigators "
                "in pursuing investigative leads. They do not constitute established legal facts or conclusions.</i>",
                self.styles["ForensicBodyMuted"],
            )
        )
        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # Section I: Forensic Limitations & Review Disclaimer
    # -----------------------------------------------------------------------
    def _build_section_i_limitations(self, story: list[Any]) -> None:
        story.append(KeepTogether([
            Paragraph("I. Limitations &amp; Investigator-Review Disclaimer", self.styles["SectionHeading"]),
            Paragraph(
                "This report was compiled by CrimeVisionAI as an automated decision-support utility. "
                "The findings contained herein are subject to the following explicit operational boundaries:",
                self.styles["ForensicBody"],
            ),
            Paragraph(
                "&bull; <b>Model Prediction Uncertainty:</b> Automated detections (RT-DETR, CLIP, BLIP, Whisper) "
                "contain inherent false-positive and false-negative error rates and must be verified against original source recordings.<br/>"
                "&bull; <b>Non-Establishment of Identity or Intent:</b> Object detectors detect visual class categories; "
                "they do not identify specific individuals, continuous identities, or criminal intent.<br/>"
                "&bull; <b>Temporal Boundary Assumptions:</b> Recordings without explicit common timecode or synchronization "
                "operate on independent time bases. Co-occurrence cannot be assumed solely from timestamp values.<br/>"
                "&bull; <b>Evidence Completeness:</b> Findings are strictly bounded by the evidence files ingested. "
                "Unrecorded angles, unanalyzed segments, or occluded visual views limit the scope of inferences.<br/>"
                "&bull; <b>No Legal Certification:</b> This document is an investigative aid, not a certified forensic finding, "
                "sworn testimony, or official law enforcement record. Certified expert human witness review is required before courtroom submission.",
                self.styles["ForensicBodyMuted"],
            ),
            Spacer(1, 4),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceBefore=2, spaceAfter=3),
            Paragraph(
                "CrimeVisionAI Multi-Modal Forensic Analysis Platform &nbsp;|&nbsp; Human Review Required Prior to Actionable Use",
                self.styles["TableCellMuted"],
            ),
        ]))
