"""
Case and Evidence Service for CrimeVisionAI.
Provides SQLite-backed persistence for investigations, evidence items,
and associated multimodal analysis results.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
import uuid


class CaseService:
    """
    Manages investigations (cases), associated evidence files,
    and persisted forensic analysis findings using SQLite.
    """

    DEFAULT_CASE_ID = "CASE-2026-MVP-001"
    DEFAULT_CASE_TITLE = "MVP Forensic Investigation"
    DEFAULT_CASE_DESC = "Default investigation workspace for session evidence."

    def __init__(self, db_path: str | Path | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.db_path = project_root / "data" / "crimevision.db"

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema and seed default case if empty."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER DEFAULT 0,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    analysis_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL UNIQUE,
                    analysis_type TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE,
                    FOREIGN KEY (evidence_id) REFERENCES evidence (evidence_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS case_fusion (
                    case_id TEXT PRIMARY KEY,
                    fusion_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS case_hypotheses (
                    case_id TEXT PRIMARY KEY,
                    hypothesis_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
                );
            """)

            # Seed default case if no cases exist
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cases;")
            count = cursor.fetchone()[0]
            if count == 0:
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    """
                    INSERT INTO cases (case_id, title, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self.DEFAULT_CASE_ID,
                        self.DEFAULT_CASE_TITLE,
                        self.DEFAULT_CASE_DESC,
                        now,
                        now,
                    ),
                )
                conn.commit()

    def _validate_case_id(self, case_id: str) -> str:
        """Ensure case ID contains only safe alphanumeric/hyphen/underscore characters."""
        clean_id = case_id.strip()
        if not clean_id:
            raise ValueError("Case ID cannot be empty.")
        if not re.match(r"^[A-Za-z0-9_\-]+$", clean_id):
            raise ValueError(
                "Invalid case ID: must contain only letters, numbers, hyphens, and underscores."
            )
        if len(clean_id) > 64:
            raise ValueError("Case ID exceeds maximum length of 64 characters.")
        return clean_id

    def create_case(
        self,
        title: str,
        description: str = "",
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create and persist a new investigation case.

        Parameters
        ----------
        title : str
            Human-readable case title.
        description : str, optional
            Detailed description of the investigation.
        case_id : str | None, optional
            Optional identifier; generated automatically if not provided.

        Returns
        -------
        dict[str, Any]
            The created case record.
        """
        clean_title = (title or "").strip()
        if not clean_title:
            raise ValueError("Case title is required and cannot be empty.")

        now = datetime.now(timezone.utc).isoformat()

        if case_id:
            cid = self._validate_case_id(case_id)
        else:
            year = datetime.now(timezone.utc).year
            random_suffix = uuid.uuid4().hex[:6].upper()
            cid = f"CASE-{year}-{random_suffix}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT case_id FROM cases WHERE case_id = ?", (cid,))
            if cursor.fetchone():
                raise ValueError(f"Case with ID '{cid}' already exists.")

            cursor.execute(
                """
                INSERT INTO cases (case_id, title, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, clean_title, description.strip(), now, now),
            )
            conn.commit()

        return {
            "case_id": cid,
            "title": clean_title,
            "description": description.strip(),
            "created_at": now,
            "updated_at": now,
        }

    def list_cases(self) -> list[dict[str, Any]]:
        """
        List all cases with evidence counts and analysis statistics.

        Returns
        -------
        list[dict[str, Any]]
            Chronologically ordered list of cases (newest first).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    c.case_id,
                    c.title,
                    c.description,
                    c.created_at,
                    c.updated_at,
                    COUNT(e.evidence_id) AS evidence_count,
                    SUM(CASE WHEN e.status = 'completed' THEN 1 ELSE 0 END) AS analyzed_count
                FROM cases c
                LEFT JOIN evidence e ON c.case_id = e.case_id
                GROUP BY c.case_id
                ORDER BY c.created_at DESC;
            """)
            rows = cursor.fetchall()

        cases = []
        for row in rows:
            cases.append({
                "case_id": row["case_id"],
                "title": row["title"],
                "description": row["description"] or "",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "evidence_count": int(row["evidence_count"] or 0),
                "analyzed_count": int(row["analyzed_count"] or 0),
            })
        return cases

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """
        Retrieve case details, all linked evidence items, their analysis results,
        and consolidated chronological timeline events.

        Parameters
        ----------
        case_id : str
            Target case identifier.

        Returns
        -------
        dict[str, Any] | None
            Case details or None if not found.
        """
        clean_id = self._validate_case_id(case_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT case_id, title, description, created_at, updated_at
                FROM cases WHERE case_id = ?
                """,
                (clean_id,),
            )
            case_row = cursor.fetchone()
            if not case_row:
                return None

            cursor.execute(
                """
                SELECT
                    e.evidence_id,
                    e.case_id,
                    e.original_filename,
                    e.media_type,
                    e.file_size,
                    e.uploaded_at,
                    e.status,
                    a.analysis_id,
                    a.analysis_type,
                    a.result_json,
                    a.created_at AS analyzed_at
                FROM evidence e
                LEFT JOIN analysis_results a ON e.evidence_id = a.evidence_id
                WHERE e.case_id = ?
                ORDER BY e.uploaded_at ASC;
                """,
                (clean_id,),
            )
            evidence_rows = cursor.fetchall()

        evidence_list = []
        consolidated_timeline_events = []

        for row in evidence_rows:
            analysis_data = None
            if row["result_json"]:
                try:
                    analysis_data = json.loads(row["result_json"])
                except Exception:
                    analysis_data = None

            item = {
                "evidence_id": row["evidence_id"],
                "case_id": row["case_id"],
                "original_filename": row["original_filename"],
                "media_type": row["media_type"],
                "file_size": row["file_size"],
                "uploaded_at": row["uploaded_at"],
                "status": row["status"],
                "analysis": analysis_data,
                "analyzed_at": row["analyzed_at"],
            }
            evidence_list.append(item)

            # Extract timeline events if audio or video analysis exists
            if (
                row["media_type"] in ("audio", "video")
                and analysis_data
                and isinstance(analysis_data.get("timeline"), dict)
            ):
                events = analysis_data["timeline"].get("events", [])
                ev_id_prefix = row["evidence_id"]
                for ev in events:
                    if isinstance(ev, dict):
                        # Attach source reference without mutating existing fields
                        enriched_event = dict(ev)
                        enriched_event["source_evidence_id"] = row["evidence_id"]
                        enriched_event["source_filename"] = row["original_filename"]
                        enriched_event["media_type"] = row["media_type"]
                        # Namespace event_id to ensure global uniqueness across multiple evidence items
                        raw_eid = str(ev.get("event_id", "evt"))
                        if not raw_eid.startswith(f"{ev_id_prefix}_"):
                            enriched_event["event_id"] = f"{ev_id_prefix}_{raw_eid}"
                        else:
                            enriched_event["event_id"] = raw_eid
                        # Clarify that timestamps are media-relative
                        enriched_event["time_basis"] = f"media_relative:{ev_id_prefix}"
                        consolidated_timeline_events.append(enriched_event)

        # Sort consolidated timeline events chronologically
        def timeline_sort_key(ev: dict[str, Any]) -> tuple[int, float]:
            s = ev.get("start_raw")
            if s is not None and isinstance(s, (int, float)):
                return (0, float(s))
            return (1, 0.0)

        consolidated_timeline_events.sort(key=timeline_sort_key)

        # Retrieve any persisted cross-modal fusion result
        fusion_data = self.get_case_fusion(clean_id)

        # Retrieve any persisted incident hypothesis
        hypothesis_data = self.get_case_hypothesis(clean_id)

        return {
            "case_id": case_row["case_id"],
            "title": case_row["title"],
            "description": case_row["description"] or "",
            "created_at": case_row["created_at"],
            "updated_at": case_row["updated_at"],
            "evidence": evidence_list,
            "evidence_count": len(evidence_list),
            "timeline_events": consolidated_timeline_events,
            "fusion": fusion_data,
            "hypothesis": hypothesis_data,
        }

    def add_evidence(
        self,
        case_id: str,
        original_filename: str,
        media_type: str,
        file_path: str,
        file_size: int = 0,
    ) -> dict[str, Any]:
        """
        Record a new uploaded evidence file associated with a case.

        Parameters
        ----------
        case_id : str
            Case identifier.
        original_filename : str
            Client-supplied filename.
        media_type : str
            'audio', 'image', or 'video'.
        file_path : str
            Server filesystem location where the file is securely saved.
        file_size : int, optional
            Size of file in bytes.

        Returns
        -------
        dict[str, Any]
            Created evidence record metadata (omits server filesystem path).
        """
        clean_id = self._validate_case_id(case_id)
        if media_type not in ("audio", "image", "video"):
            raise ValueError(f"Invalid media_type '{media_type}'; must be 'audio', 'image', or 'video'.")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT case_id FROM cases WHERE case_id = ?", (clean_id,))
            if not cursor.fetchone():
                raise FileNotFoundError(f"Case '{clean_id}' does not exist.")

            evidence_id = f"EVD-{uuid.uuid4().hex[:8].upper()}"
            now = datetime.now(timezone.utc).isoformat()

            cursor.execute(
                """
                INSERT INTO evidence (
                    evidence_id, case_id, original_filename, media_type,
                    file_path, file_size, uploaded_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    evidence_id,
                    clean_id,
                    original_filename,
                    media_type,
                    str(file_path),
                    int(file_size or 0),
                    now,
                ),
            )
            # Update case timestamp
            cursor.execute("UPDATE cases SET updated_at = ? WHERE case_id = ?", (now, clean_id))
            conn.commit()

        return {
            "evidence_id": evidence_id,
            "case_id": clean_id,
            "original_filename": original_filename,
            "media_type": media_type,
            "file_size": file_size,
            "uploaded_at": now,
            "status": "pending",
        }

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        """
        Retrieve evidence by evidence ID, including the stored internal file path.

        Parameters
        ----------
        evidence_id : str
            Evidence record identifier.

        Returns
        -------
        dict[str, Any] | None
            Evidence record with file_path, or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    e.evidence_id, e.case_id, e.original_filename,
                    e.media_type, e.file_path, e.file_size,
                    e.uploaded_at, e.status,
                    a.result_json, a.created_at as analyzed_at
                FROM evidence e
                LEFT JOIN analysis_results a ON e.evidence_id = a.evidence_id
                WHERE e.evidence_id = ?
                """,
                (evidence_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            analysis_data = None
            if row["result_json"]:
                try:
                    analysis_data = json.loads(row["result_json"])
                except Exception:
                    analysis_data = None

            return {
                "evidence_id": row["evidence_id"],
                "case_id": row["case_id"],
                "original_filename": row["original_filename"],
                "media_type": row["media_type"],
                "file_path": row["file_path"],
                "file_size": row["file_size"],
                "uploaded_at": row["uploaded_at"],
                "status": row["status"],
                "analysis": analysis_data,
                "analyzed_at": row["analyzed_at"],
            }

    def save_analysis_result(
        self,
        case_id: str,
        evidence_id: str,
        analysis_type: str,
        result_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Persist analysis results for an evidence record.

        Parameters
        ----------
        case_id : str
            Case identifier.
        evidence_id : str
            Evidence identifier.
        analysis_type : str
            'audio' or 'image'.
        result_data : dict[str, Any]
            The analysis findings returned by AudioService or VisionService.

        Returns
        -------
        dict[str, Any]
            The persisted analysis metadata.
        """
        now = datetime.now(timezone.utc).isoformat()
        analysis_id = f"ANL-{uuid.uuid4().hex[:8].upper()}"
        result_json = json.dumps(result_data, ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Upsert into analysis_results
            cursor.execute(
                """
                INSERT INTO analysis_results (
                    analysis_id, case_id, evidence_id, analysis_type, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    analysis_id = excluded.analysis_id,
                    analysis_type = excluded.analysis_type,
                    result_json = excluded.result_json,
                    created_at = excluded.created_at;
                """,
                (analysis_id, case_id, evidence_id, analysis_type, result_json, now),
            )
            cursor.execute(
                "UPDATE evidence SET status = 'completed' WHERE evidence_id = ?",
                (evidence_id,),
            )
            cursor.execute(
                "UPDATE cases SET updated_at = ? WHERE case_id = ?",
                (now, case_id),
            )
            cursor.execute("DELETE FROM case_hypotheses WHERE case_id = ?", (case_id,))
            conn.commit()

        return {
            "analysis_id": analysis_id,
            "case_id": case_id,
            "evidence_id": evidence_id,
            "analysis_type": analysis_type,
            "created_at": now,
        }

    def update_evidence_status(self, evidence_id: str, status: str) -> None:
        """Update processing status for an evidence record (e.g., 'error')."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE evidence SET status = ? WHERE evidence_id = ?",
                (status, evidence_id),
            )
            conn.commit()

    def save_case_fusion(self, case_id: str, fusion_data: dict[str, Any]) -> dict[str, Any]:
        """Persist structured cross-modal fusion results for a case."""
        clean_id = self._validate_case_id(case_id)
        now = datetime.now(timezone.utc).isoformat()
        fusion_json = json.dumps(fusion_data, ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO case_fusion (case_id, fusion_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    fusion_json = excluded.fusion_json,
                    updated_at = excluded.updated_at;
                """,
                (clean_id, fusion_json, now),
            )
            cursor.execute("UPDATE cases SET updated_at = ? WHERE case_id = ?", (now, clean_id))
            conn.commit()

        return {
            "case_id": clean_id,
            "updated_at": now,
            "status": "success",
        }

    def get_case_fusion(self, case_id: str) -> dict[str, Any] | None:
        """Retrieve persisted fusion results for a case if available."""
        clean_id = self._validate_case_id(case_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fusion_json, updated_at FROM case_fusion WHERE case_id = ?",
                (clean_id,),
            )
            row = cursor.fetchone()
            if not row or not row["fusion_json"]:
                return None

            try:
                data = json.loads(row["fusion_json"])
                return data
            except Exception:
                return None

    def save_case_hypothesis(self, case_id: str, hypothesis_data: dict[str, Any]) -> dict[str, Any]:
        """Persist structured incident hypothesis results for a case."""
        clean_id = self._validate_case_id(case_id)
        now = datetime.now(timezone.utc).isoformat()
        hyp_json = json.dumps(hypothesis_data, ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO case_hypotheses (case_id, hypothesis_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    hypothesis_json = excluded.hypothesis_json,
                    updated_at = excluded.updated_at;
                """,
                (clean_id, hyp_json, now),
            )
            cursor.execute("UPDATE cases SET updated_at = ? WHERE case_id = ?", (now, clean_id))
            conn.commit()

        return {
            "case_id": clean_id,
            "updated_at": now,
            "status": "success",
        }

    def get_case_hypothesis(self, case_id: str) -> dict[str, Any] | None:
        """Retrieve persisted incident hypothesis for a case if available."""
        clean_id = self._validate_case_id(case_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT hypothesis_json, updated_at FROM case_hypotheses WHERE case_id = ?",
                (clean_id,),
            )
            row = cursor.fetchone()
            if not row or not row["hypothesis_json"]:
                return None

            try:
                data = json.loads(row["hypothesis_json"])
                return data
            except Exception:
                return None

    def clear_case_hypothesis(self, case_id: str) -> None:
        """Clear/invalidate stored incident hypothesis for a case (e.g. when evidence changes)."""
        clean_id = self._validate_case_id(case_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM case_hypotheses WHERE case_id = ?", (clean_id,))
            conn.commit()

