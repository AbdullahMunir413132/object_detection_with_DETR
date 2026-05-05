# =============================================================================
# core/logger.py — SQLite-based detection event logger
# =============================================================================

from __future__ import annotations

import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'unknown',
    ts          TEXT    NOT NULL,
    class_id    INTEGER NOT NULL,
    class_name  TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    x1          REAL, y1 REAL, x2 REAL, y2 REAL,
    model_name  TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    source      TEXT,
    model_name  TEXT,
    total_frames INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0
);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    """Create tables if they don't exist."""
    with _conn() as con:
        con.execute(_CREATE_DETECTIONS)
        con.execute(_CREATE_SESSIONS)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def start_session(session_id: str, source: str, model_name: str):
    with _conn() as con:
        con.execute(
            "INSERT OR IGNORE INTO sessions(session_id, started_at, source, model_name) VALUES (?,?,?,?)",
            (session_id, datetime.utcnow().isoformat(), source, model_name),
        )


def end_session(session_id: str):
    with _conn() as con:
        con.execute(
            "UPDATE sessions SET ended_at=? WHERE session_id=?",
            (datetime.utcnow().isoformat(), session_id),
        )


def update_session_stats(session_id: str, frames: int, detections: int):
    with _conn() as con:
        con.execute(
            "UPDATE sessions SET total_frames=?, total_detections=? WHERE session_id=?",
            (frames, detections, session_id),
        )


# ---------------------------------------------------------------------------
# Detection logging
# ---------------------------------------------------------------------------

def log_detections(
    session_id: str,
    records: list[dict],
    source: str = "unknown",
    model_name: str = "",
):
    """Batch-insert detection records."""
    if not records:
        return
    ts = datetime.utcnow().isoformat()
    rows = [
        (
            session_id, source, ts,
            r["class_id"], r["class_name"], r["confidence"],
            r.get("x1"), r.get("y1"), r.get("x2"), r.get("y2"),
            model_name,
        )
        for r in records
    ]
    with _conn() as con:
        con.executemany(
            """INSERT INTO detections
               (session_id, source, ts, class_id, class_name, confidence,
                x1, y1, x2, y2, model_name)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def fetch_all_detections(limit: int = 10_000) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_sessions() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_class_summary() -> list[dict]:
    """Return (class_name, count, avg_conf) grouped stats."""
    with _conn() as con:
        rows = con.execute(
            """SELECT class_name,
                      COUNT(*) as count,
                      AVG(confidence) as avg_conf
               FROM detections
               GROUP BY class_name
               ORDER BY count DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_detections_over_time(bucket: str = "minute") -> list[dict]:
    """
    Returns (bucket_ts, count) bucketed by minute or hour.
    *bucket* can be 'minute' or 'hour'.
    """
    fmt = "%Y-%m-%dT%H:%M" if bucket == "minute" else "%Y-%m-%dT%H"
    with _conn() as con:
        rows = con.execute(
            f"""SELECT SUBSTR(ts, 1, {len('YYYY-MM-DDTHH:MM')}) as bucket,
                       COUNT(*) as count
                FROM detections
                GROUP BY bucket
                ORDER BY bucket ASC"""
        ).fetchall()
    return [dict(r) for r in rows]


def clear_all_data():
    with _conn() as con:
        con.execute("DELETE FROM detections")
        con.execute("DELETE FROM sessions")


# Initialise on import
init_db()
