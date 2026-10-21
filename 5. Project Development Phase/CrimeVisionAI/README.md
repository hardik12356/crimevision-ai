# CrimeVisionAI — Forensic Multimodal Investigation & Evidence Synthesis System

CrimeVisionAI is an AI-assisted digital forensics and investigative decision-support platform. It ingests, analyzes, aligns, and cross-corroborates multi-source surveillance video, forensic audio recordings, and still imagery into a unified, source-traceable timeline.

The system is designed to assist human investigators by automating low-level media parsing, surfacing cross-modal consistencies and contradictions, generating evidence-grounded incident hypotheses, and providing an interactive, rate-governed Q&A assistant grounded in recorded findings.

---

## Table of Contents

- [Core Capabilities](#core-capabilities)
- [System Architecture & Evidence Flow](#system-architecture--evidence-flow)
- [Technology Stack & AI Models](#technology-stack--ai-models)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Setup (Windows)](#installation--setup-windows)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Running the Test Suite](#running-the-test-suite)
- [Supported Evidence Formats & Constraints](#supported-evidence-formats--constraints)
- [API Endpoints Reference](#api-endpoints-reference)
- [Forensic Integrity & Known Limitations](#forensic-integrity--known-limitations)
- [Responsible Use & Disclaimer](#responsible-use--disclaimer)

---

## Core Capabilities

### 1. Case Management & Evidence Locker
- **Case Tracking:** Create and manage distinct investigative cases with full metadata, timestamps, and persistent state in SQLite (`data/crimevision.db`).
- **Multimodal Evidence Ingestion:** Ingest image, video, and audio evidence with automatic format validation, sanitization, and secure local disk persistence.
- **Evidence Locker Interface:** Direct inspection of ingested media files, processing status tracking, and central primary canvas viewing.

### 2. Deep Visual & Audio Analysis
- **Object Detection (RT-DETR-L):** General-purpose Transformer-based object detection using COCO-pretrained weights (80 classes). Person counting is derived from per-frame detection counts. Weapon-related labels (e.g., `knife`, `baseball bat`) are surfaced via heuristic label matching in the fusion layer — **not** a dedicated or reliable weapon detector.
- **Image Captioning (BLIP):** Generates descriptive natural-language captions for analyzed image frames using `Salesforce/blip-image-captioning-base` (standard captioning, not dense captioning).
- **Zero-Shot Scene Classification (CLIP):** Classifies operational surveillance environments (e.g., retail shop, street, office, warehouse, vehicle interior) using `openai/clip-vit-base-patch32`.
- **Video Sampling & Analysis:** Samples video frames at a configurable target of ~4 FPS (default `DEFAULT_TARGET_FPS = 4.0`). Frame timestamps are derived from frame index and source file FPS via float division — precision depends on the source video metadata. Heuristic events (person presence changes, scene transitions) are generated from frame-to-frame detection count differences; **no person tracking, re-identification, entry/exit confirmation, or physical interaction detection is implemented.**
- **Speech-to-Text & Keyword Extraction (Whisper):** Transcribes audio recordings using `openai/whisper-base` with word- and segment-level timestamps. Entity extraction (persons, vehicles, locations) uses regex and keyword heuristics applied to transcription text.

### 3. Cross-Modal Evidence Fusion & Audit
- **Temporal Normalization:** Float-normalized time indexing relative to individual recording baselines. Timestamps derived from media playback do not establish real-world timing accuracy or synchronize independent recordings without common external timecodes.
- **Deterministic Hash Index:** Uses SHA-256 hashing to generate reproducible dictionary lookup keys for normalized temporal intervals and extracted entities. This is an internal lookup and indexing mechanism, not a cryptographic proof of evidence authenticity or chain of custody.
- **Pairwise Heuristic Correlation & Discrepancy Auditing:**
  - *Weapon presence correlation:* Matches visual object detections with spoken weapon mentions based on label dictionaries; surfaces investigative leads rather than establishing physical corroboration.
  - *Explicit contradiction flagging:* Highlights discrepancies between audio claims (e.g., statement asserting subject was unarmed) and visual object detections at proximate playback timestamps.
  - *Alibi / solitude contradiction flagging:* Highlights discrepancies between statements claiming a subject was alone and visual detections of multiple persons.
  - *Aggression & altercation consistency:* Heuristically flags acoustic aggression keywords alongside visual altercation labels.
  - *Deduplicated thematic matching:* Correlates witness person mentions with visual person observations, consolidated per evidence pair.
- **Anchor Event Highlighting:** Automatically flags high-priority investigative anchors (potential weapon mentions/detections, person observations, distress keywords, and active contradiction flags).

### 4. Investigator Evidence Synthesis Workspace
- **Executive Overview & Dynamic Metrics:** Accurately calculated metrics for evidence links, flagged conflicts, items requiring review, and total events, paired with prominent temporal alignment status (`⏱ Aligned Common Timeline` vs. `⚠️ Independent Time Bases`).
- **Evidence Gaps Banner:** Highlights unaligned media time bases, missing modalities, and low-confidence detections requiring verification.
- **Key Findings Hierarchy:** Categorized view summarizing witness statements, visual detections, cross-modal alignments, and contradictions.
- **Readable Relationship Cards:** Source comparison chips (`Source A [Icon/Filename/Time]` vs. `Source B [Icon/Filename/Time]`) with one-click canvas inspection and video seek synchronization.
- **Collapsible Technical Audit Trail:** `<details>` accordion preserving the complete grounded narrative, SHA-256 hash index records, and raw forensic event ledger for investigator inspection.

### 5. Incident Hypothesis Engine
- Synthesizes case evidence to generate candidate incident types using Groq LLM prompting.
- Outlines supporting evidence, contradicting evidence, investigative assumptions, uncertainty factors, and overall evidence sufficiency.
- **Probabilistic Guidance:** Hypotheses are automated interpretations intended to assist human investigators, not certified determinations or conclusive facts.

### 6. Investigator AI Q&A Assistant
- Retrieval-augmented query answering conditioned on case timeline events, frame observations, audio transcripts, and fusion links. Case context bounds the model's inputs, but LLM responses remain probabilistic interpretations and may contain inaccuracies.
- Query intent classification (`person`, `timestamp`, `conflict`, `hypothesis`, `general`).
- **Token Budget Protection:** Tracks rolling TPM and RPM usage against Groq API limits to mitigate rate-limit errors, though upstream provider throttling (HTTP 429) can still occur.
- **Responsive Table Rendering:** Custom Markdown renderer with dedicated styling and horizontal scroll containment, ensuring multi-column tables fit neatly inside chat bubbles.

### 7. Forensic Reporting & Case Export
- Unified JSON export of case metadata, evidence summaries, and forensic event records.
- PDF forensic report generation using ReportLab, compiling executive summaries, evidence registries, fusion matrices, and forensic limitation disclaimers (not legally certified or courtroom-admissible without certified expert human review).

---

## System Architecture & Evidence Flow

```
                                  EVIDENCE INGESTION
                         (Images, CCTV Video, Audio Recordings)
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │        CrimeVisionAI Flask App         │
                      │         (routes.py / API layer)        │
                      └───────────────────┬────────────────────┘
                                          │
               ┌──────────────────────────┼──────────────────────────┐
               ▼                          ▼                          ▼
     ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
     │   VisionService   │      │   VideoService    │      │   AudioService    │
     │ ───────────────── │      │ ───────────────── │      │ ───────────────── │
     │ • RT-DETR-L       │      │ • ~4 FPS Sampling │      │ • Whisper ASR     │
     │ • BLIP Captioner  │      │ • RT-DETR Detections│    │ • Segment Timers  │
     │ • CLIP Classifier │      │ • Timeline Events │      │ • Entity Parsing  │
     └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
               │                          │                          │
               └──────────────────────────┼──────────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │             FusionService              │
                      │  • Playback Time Normalization         │
                      │  • Typed ForensicEvent Standardization │
                      │  • Deterministic SHA-256 Hash Index    │
                      │  • Pairwise Heuristic Rules (1 to 5)   │
                      │  • Anchor Event Highlighting           │
                      │  • Evidence Consistency Scoring        │
                      └───────────────────┬────────────────────┘
                                          │
               ┌──────────────────────────┼──────────────────────────┐
               ▼                          ▼                          ▼
     ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
     │Incident Hypothesis│      │ Investigator Q&A  │      │  Forensic Export  │
     │      Engine       │      │   (Groq + Token   │      │    (JSON & PDF    │
     │ (HypothesisService│      │   Budget Manager) │      │  Summary Report)  │
     └───────────────────┘      └───────────────────┘      └───────────────────┘
```

---

## Technology Stack & AI Models

| Component | Technology / Library | Specific Model / Version |
|---|---|---|
| **Web Framework** | Python 3.12, Flask 3.0, Flask-CORS | Werkzeug, Jinja2 |
| **Database** | SQLite 3 (Standard Library `sqlite3`) | `data/crimevision.db` |
| **Object Detection** | PyTorch, Ultralytics | **RT-DETR-L** (`rtdetr-l.pt`, 63.4 MB, COCO 80 classes) |
| **Image Captioning** | Hugging Face Transformers | **Salesforce/blip-image-captioning-base** |
| **Scene Classification** | Hugging Face Transformers | **openai/clip-vit-base-patch32** |
| **Audio Transcription** | OpenAI Whisper, Librosa, SoundFile | **openai/whisper-base** |
| **LLM Reasoning & Chat** | Groq Python SDK | **openai/gpt-oss-20b** (or configured Groq model) |
| **Document Export** | ReportLab 4.0, PyPDF2 | Forensic PDF generator |
| **Frontend UI** | Vanilla JavaScript (ES6+), HTML5, CSS3 | Dual Light/Dark Forensic Theme |
| **Testing Framework** | pytest 9.1, pytest-flask | 139 automated tests |

---

## Project Directory Structure

```
CrimeVisionAI/
├── app/
│   ├── __init__.py                 # Flask application factory, config & error handlers
│   ├── routes.py                   # REST API routes & dashboard view endpoints
│   ├── services/
│   │   ├── audio_service.py        # Audio pipeline coordinator & timeline builder
│   │   ├── audio_transcriber.py    # Whisper transcription wrapper & entity parser
│   │   ├── case_service.py         # Direct sqlite3 persistence for cases, evidence & fusion
│   │   ├── context_builder.py      # Compact grounded context assembler for LLM
│   │   ├── fusion_service.py       # Cross-modal timeline normalization & audit engine
│   │   ├── groq_service.py         # Groq LLM client for hypothesis & grounded Q&A
│   │   ├── hypothesis_service.py   # Incident hypothesis evaluator
│   │   ├── image_captioner.py      # BLIP image captioning service
│   │   ├── object_detector.py      # RT-DETR-L inference & label heuristic extractor
│   │   ├── pdf_service.py          # Forensic investigative ReportLab PDF generator
│   │   ├── scene_classifier.py     # CLIP zero-shot scene classification service
│   │   ├── timeline_service.py     # Temporal sorting & observation aggregation
│   │   ├── token_budget.py         # Token & request rate-limit governor (TPM/RPM)
│   │   ├── video_service.py        # Video frame extraction, sampling & analysis
│   │   └── vision_service.py       # Combined vision analysis coordinator
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css           # Design tokens, responsive layout & table styling
│   │   └── js/
│   │       ├── api.js              # Client-side API fetch client
│   │       └── app.js              # State manager, Markdown parser & UI controller
│   └── templates/
│       └── index.html              # Investigation workspace single-page dashboard
├── data/
│   └── crimevision.db              # SQLite case database (auto-created if absent)
├── evidence/
│   └── uploads/                    # Local storage directory for uploaded case media
├── tests/                          # Automated unit and integration test suite (139 tests)
│   ├── test_activity_5_2.py
│   ├── test_case_management.py
│   ├── test_dashboard_routes.py
│   ├── test_epic_6_1.py
│   ├── test_fusion_redesign.py     # Verification tests for fusion deduplication & UI
│   ├── test_fusion_service.py
│   ├── test_groq_service.py
│   ├── test_incident_hypothesis.py
│   ├── test_pdf_report.py
│   ├── test_token_budget_and_qa.py
│   └── ...
├── requirements.txt                # Python package dependencies
├── rtdetr-l.pt                     # Local PyTorch weights for RT-DETR-L
├── run.py                          # Application entry point script
└── .env.example                    # Sample environment variable template
```

---

## Installation & Setup (Windows)

### Prerequisites
- **Operating System:** Windows 10 or 11 (64-bit)
- **Python:** Python 3.12 (64-bit) recommended
- **Hardware:** Minimum 8 GB RAM (16 GB recommended). A CUDA-capable NVIDIA GPU is optional; all models operate on CPU by default.
- **Model Weights:** Ensure `rtdetr-l.pt` is located in the project root directory.

### Step 1: Navigate to Project Directory
Open PowerShell or Windows Terminal:
```powershell
cd C:\path\to\CrimeVisionAI
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Package Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Environment Configuration

Configure environment variables via PowerShell or a local `.env` file:

```powershell
$env:GROQ_API_KEY="your_actual_groq_api_key_here"
```

Alternatively, create a `.env` file in the project root:

```ini
# Groq API Key for Investigator AI Q&A and Incident Hypothesis Generation
GROQ_API_KEY=your_actual_groq_api_key_here

# Flask Application Settings
SECRET_KEY=crimevision-local-dev-key
DATABASE_PATH=data/crimevision.db
UPLOAD_FOLDER=evidence/uploads
```

> [!NOTE]
> The Flask application reads settings directly from process environment variables (`os.getenv`). If configuring via a `.env` file, ensure variables are exported into your active shell environment before running the application.

> [!IMPORTANT]
> Never commit actual API keys or credentials to version control. Keep `.env` strictly local.

---

## Running the Application

Ensure your virtual environment is active:

```powershell
.\.venv\Scripts\Activate.ps1
python run.py
```

The application will start on:
```
http://127.0.0.1:5000
```

Open this address in any modern web browser to access the CrimeVisionAI Investigation Workspace.

---

## Running the Test Suite

The test suite covers route handlers, mock AI pipelines, multimodal fusion, token governance, rate limiting, and PDF exports.

Run all 139 tests:
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\pytest -q
```

Run a specific test module (e.g., Cross-Modal Fusion tests):
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\pytest tests/test_fusion_redesign.py -v
```

---

## Supported Evidence Formats & Constraints

| Modality | Supported Formats | Constraints & Parameters |
|---|---|---|
| **Images** | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff` | Max 100 MB per file. Processed with RT-DETR-L (detection), BLIP (captions), CLIP (scenes). |
| **Videos** | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` | Max 100 MB per file. Adaptive frame sampling target: ~4 FPS. Subsecond timestamp tracking. |
| **Audio** | `.wav`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.flac`, `.mp3` | Max 100 MB per file. Whisper-base transcription with word/segment timestamps. |

---

## API Endpoints Reference

### Dashboard & Cases
- `GET /` — Investigation Dashboard single-page application.
- `GET /api/cases` — List all registered cases.
- `POST /api/cases` — Create a new investigative case (`title`, `description`).
- `GET /api/cases/<case_id>` — Fetch case details, evidence items, timeline, and fusion findings.

### Evidence Ingestion & Analysis
- `POST /api/cases/<case_id>/evidence` — Upload evidence file (`file`, `notes`).
- `POST /api/cases/<case_id>/evidence/<evidence_id>/analyze` — Run full model analysis on a specific evidence item.
- `GET /api/evidence/<evidence_id>/media` — Stream raw uploaded media file.

### Fusion & Reasoning
- `GET /api/cases/<case_id>/fusion` — Retrieve existing multimodal fusion results.
- `POST /api/cases/<case_id>/fusion` — Execute deterministic cross-modal fusion and contradiction auditing.
- `GET /api/cases/<case_id>/hypothesis` — Retrieve evaluated incident hypotheses.
- `POST /api/cases/<case_id>/hypothesis` — Trigger LLM incident hypothesis evaluation.
- `POST /api/cases/<case_id>/chat` — Query Investigator AI Q&A (`query`, grounded in case records).

### Forensic Export
- `GET /api/cases/<case_id>/export` — Download complete unified forensic case ledger in JSON format.
- `GET /api/cases/<case_id>/export/pdf` — Generate and stream official forensic case summary PDF.

---

## Forensic Integrity & Known Limitations

1. **Automated Decision Support Only:**
   CrimeVisionAI outputs are heuristic decision-support markers designed to accelerate evidence discovery. They do **not** constitute legal conclusions or forensic proof. Human investigator oversight and verification of raw recordings are strictly required.

2. **Independent Time Bases:**
   Unless evidence files share an authenticated common UTC timecode, media timestamps are relative to individual file playback (`0.0s` = file start). Temporal co-occurrence indicates intra-recording alignment or proximity, not proven real-world simultaneity.

3. **Detector & Captioner Uncertainty:**
   Object detection is subject to model confidence thresholds (default 0.25). Low-light conditions, occlusion, and motion blur can produce missed detections or misclassifications (e.g., confusing handheld items with weapons). Captions generated by BLIP represent probabilistic image-level summaries and must be independently verified.

4. **Speech Transcription Constraints:**
   Audio transcriptions generated by Whisper may misinterpret low-clarity speech, heavy background noise, or overlapping dialogue. Audio timestamps represent acoustic speech activity boundaries.

5. **LLM Reasoning & Rate Governance:**
   Incident hypotheses and Q&A responses are generated by Large Language Models via Groq. While inputs are grounded in extracted case records, LLM reasoning remains probabilistic and can introduce errors, misinterpretations, or hallucinations. The in-memory token budget manager mitigates local request spikes, but cannot prevent upstream provider-side rate limits (HTTP 429). All AI-generated insights must be verified against primary evidence before investigative action.

6. **Legal & Courtroom Admissibility Notice:**
   CrimeVisionAI reports, JSON ledgers, and visual alignment cards are automated investigative aids. They do not constitute certified forensic records, chain-of-custody certificates, or sworn expert testimony. Courtroom presentation requires independent human expert analysis and jurisdictional forensic validation.

---

## Responsible Use & Disclaimer

CrimeVisionAI is designed for authorized digital forensics, academic research, and investigative case management. Users are responsible for ensuring evidence handling complies with jurisdictional standards of digital evidence custody, data protection laws, and institutional review policies.
