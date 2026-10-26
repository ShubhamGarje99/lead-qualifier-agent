"""Run registry: one row per lead/thread with status, tier and exact cost.

Graph state itself lives in the LangGraph checkpointer; this table is the
queryable index (pending list, `leadq runs`, idempotency for the webhook).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATUSES = ("queued", "running", "awaiting_review", "approved", "rejected", "disqualified", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    thread_id TEXT PRIMARY KEY,
                    email TEXT, name TEXT, company TEXT, source TEXT,
                    mode TEXT, status TEXT NOT NULL,
                    tier TEXT, score REAL,
                    input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0, llm_calls INTEGER DEFAULT 0,
                    usage_json TEXT, error TEXT,
                    created_at TEXT, updated_at TEXT
                )
                """
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def claim(self, thread_id: str, *, email: str, name: str, company: str, source: str, mode: str) -> bool:
        """Insert a queued run. Returns False if this lead (thread) already exists: idempotency."""
        now = _now()
        with self._conn() as c:
            cur = c.execute(
                """INSERT OR IGNORE INTO runs (thread_id, email, name, company, source, mode, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
                (thread_id, email, name, company, source, mode, now, now),
            )
            return cur.rowcount == 1

    def update(self, thread_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE runs SET {cols} WHERE thread_id = ?", (*fields.values(), thread_id))

    def sync_usage(self, thread_id: str, usage: list[dict[str, Any]]) -> None:
        self.update(
            thread_id,
            input_tokens=sum(u.get("input_tokens", 0) for u in usage),
            output_tokens=sum(u.get("output_tokens", 0) for u in usage),
            cost_usd=round(sum(u.get("cost_usd", 0.0) for u in usage), 8),
            llm_calls=len(usage),
            usage_json=json.dumps(usage),
        )

    def get(self, thread_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE thread_id = ?", (thread_id,)).fetchone()
        return dict(row) if row else None

    def list(self, status: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM runs", []
        if status:
            q += " WHERE status = ?"
            args.append(status)
        q += " ORDER BY created_at DESC, thread_id LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    def totals(self) -> dict[str, Any]:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS runs, COALESCE(SUM(input_tokens),0) AS input_tokens, "
                "COALESCE(SUM(output_tokens),0) AS output_tokens, COALESCE(SUM(cost_usd),0) AS cost_usd FROM runs"
            ).fetchone()
        return dict(row)
