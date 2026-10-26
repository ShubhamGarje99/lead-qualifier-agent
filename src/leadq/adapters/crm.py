"""CRM adapters.

`CRM` is the port; `SQLiteCRM` is the default local mock (contacts, deals,
notes) and `HubSpotCRM` maps the same calls onto HubSpot's CRM v3 objects API.
HubSpot stays in dry-run mode (payloads recorded, nothing sent) unless
HUBSPOT_TOKEN is set. Tests never call HubSpot.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ContactPayload(BaseModel):
    email: str
    first_name: str
    last_name: str = ""
    company: str = ""
    title: str = ""
    website: str = ""
    lead_source: str = ""
    tier: str = ""
    score: float = 0.0
    industry: str = ""
    employee_band: str = ""
    region: str = ""


class DealPayload(BaseModel):
    thread_id: str
    name: str
    stage: str
    tier: str
    score: float


class NotePayload(BaseModel):
    thread_id: str
    kind: str  # "qualification" | "outbound_email"
    body: str


class CRM(ABC):
    name: str = "crm"

    @abstractmethod
    def upsert_contact(self, contact: ContactPayload) -> str: ...

    @abstractmethod
    def create_deal(self, contact_id: str, deal: DealPayload) -> str: ...

    @abstractmethod
    def add_note(self, contact_id: str, note: NotePayload) -> str: ...


# --------------------------------------------------------------------------
class SQLiteCRM(CRM):
    """Local mock CRM. Idempotent: contacts keyed by email, deals/notes by thread."""

    name = "sqlite"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    first_name TEXT, last_name TEXT, company TEXT, title TEXT, website TEXT,
                    lead_source TEXT, tier TEXT, score REAL, industry TEXT, employee_band TEXT, region TEXT,
                    created_at TEXT, updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS deals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contact_id INTEGER NOT NULL REFERENCES contacts(id),
                    thread_id TEXT UNIQUE NOT NULL,
                    name TEXT, stage TEXT, tier TEXT, score REAL, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contact_id INTEGER NOT NULL REFERENCES contacts(id),
                    thread_id TEXT NOT NULL, kind TEXT NOT NULL, body TEXT, created_at TEXT,
                    UNIQUE(thread_id, kind)
                );
                """
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_contact(self, contact: ContactPayload) -> str:
        d = contact.model_dump()
        now = _now()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO contacts (email, first_name, last_name, company, title, website, lead_source,
                                      tier, score, industry, employee_band, region, created_at, updated_at)
                VALUES (:email, :first_name, :last_name, :company, :title, :website, :lead_source,
                        :tier, :score, :industry, :employee_band, :region, :now, :now)
                ON CONFLICT(email) DO UPDATE SET
                    first_name=excluded.first_name, last_name=excluded.last_name, company=excluded.company,
                    title=excluded.title, website=excluded.website, tier=excluded.tier, score=excluded.score,
                    industry=excluded.industry, employee_band=excluded.employee_band, region=excluded.region,
                    updated_at=excluded.updated_at
                """,
                {**d, "now": now},
            )
            row = c.execute("SELECT id FROM contacts WHERE email = ?", (contact.email,)).fetchone()
        return str(row["id"])

    def create_deal(self, contact_id: str, deal: DealPayload) -> str:
        with self._conn() as c:
            c.execute(
                """INSERT INTO deals (contact_id, thread_id, name, stage, tier, score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO NOTHING""",
                (int(contact_id), deal.thread_id, deal.name, deal.stage, deal.tier, deal.score, _now()),
            )
            row = c.execute("SELECT id FROM deals WHERE thread_id = ?", (deal.thread_id,)).fetchone()
        return str(row["id"])

    def add_note(self, contact_id: str, note: NotePayload) -> str:
        with self._conn() as c:
            c.execute(
                """INSERT INTO notes (contact_id, thread_id, kind, body, created_at) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(thread_id, kind) DO NOTHING""",
                (int(contact_id), note.thread_id, note.kind, note.body, _now()),
            )
            row = c.execute(
                "SELECT id FROM notes WHERE thread_id = ? AND kind = ?", (note.thread_id, note.kind)
            ).fetchone()
        return str(row["id"])

    # Read helpers for the CLI / UI / tests
    def contacts(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM contacts ORDER BY id")]

    def deals(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM deals ORDER BY id")]

    def notes(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM notes ORDER BY id")]


# --------------------------------------------------------------------------
class HubSpotCRM(CRM):
    """Maps CRM calls onto HubSpot CRM v3 (contacts batch upsert, deals, notes).

    Dry-run unless a token is provided: payloads are appended to `dry_run_log`
    (and optionally a JSONL file) and synthetic ids are returned.
    """

    name = "hubspot"
    BASE_URL = "https://api.hubapi.com"
    NOTE_TO_CONTACT = 202  # HUBSPOT_DEFINED association type ids
    DEAL_TO_CONTACT = 3

    def __init__(self, token: str | None = None, *, log_path: Path | None = None, client: httpx.Client | None = None):
        self.token = token
        self.dry_run = not token
        self.log_path = log_path
        self.dry_run_log: list[dict[str, Any]] = []
        self._client = client

    # ---- payload mapping (pure functions, unit-testable) ------------------
    @staticmethod
    def contact_body(contact: ContactPayload) -> dict[str, Any]:
        return {
            "inputs": [
                {
                    "idProperty": "email",
                    "id": contact.email,
                    "properties": {
                        "email": contact.email,
                        "firstname": contact.first_name,
                        "lastname": contact.last_name,
                        "company": contact.company,
                        "jobtitle": contact.title,
                        "website": contact.website,
                        "lifecyclestage": "salesqualifiedlead" if contact.tier in {"A", "B"} else "lead",
                        "hs_lead_status": "OPEN",
                        "industry": contact.industry,
                    },
                }
            ]
        }

    def deal_body(self, contact_id: str, deal: DealPayload) -> dict[str, Any]:
        return {
            "properties": {"dealname": deal.name, "pipeline": "default", "dealstage": deal.stage},
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": self.DEAL_TO_CONTACT}],
                }
            ],
        }

    def note_body(self, contact_id: str, note: NotePayload) -> dict[str, Any]:
        return {
            "properties": {"hs_timestamp": _now(), "hs_note_body": note.body},
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": self.NOTE_TO_CONTACT}],
                }
            ],
        }

    # ---- transport ---------------------------------------------------------
    def _send(self, path: str, body: dict[str, Any], synthetic_id: str) -> str:
        if self.dry_run:
            entry = {"at": _now(), "method": "POST", "url": f"{self.BASE_URL}{path}", "body": body}
            self.dry_run_log.append(entry)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
            return synthetic_id
        client = self._client or httpx.Client(timeout=15.0)
        resp = client.post(
            f"{self.BASE_URL}{path}", json=body, headers={"Authorization": f"Bearer {self.token}"}
        )
        resp.raise_for_status()
        data = resp.json()
        if "results" in data:  # batch upsert
            return str(data["results"][0]["id"])
        return str(data["id"])

    @staticmethod
    def _sid(prefix: str, key: str) -> str:
        return f"dryrun-{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"

    def upsert_contact(self, contact: ContactPayload) -> str:
        return self._send("/crm/v3/objects/contacts/batch/upsert", self.contact_body(contact), self._sid("contact", contact.email))

    def create_deal(self, contact_id: str, deal: DealPayload) -> str:
        return self._send("/crm/v3/objects/deals", self.deal_body(contact_id, deal), self._sid("deal", deal.thread_id))

    def add_note(self, contact_id: str, note: NotePayload) -> str:
        return self._send("/crm/v3/objects/notes", self.note_body(contact_id, note), self._sid("note", note.thread_id + note.kind))
