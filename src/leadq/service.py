"""Application service: wires settings, adapters, checkpointer and graph.

Both the CLI and the FastAPI app talk to `LeadQualifier`; nothing else
touches the graph directly.
"""

from __future__ import annotations

import shutil
import sqlite3
import traceback
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .adapters.crm import CRM, HubSpotCRM, SQLiteCRM
from .adapters.notify import Notifier, OutboxNotifier, SlackWebhookNotifier
from .enrich import Fetcher, FixtureFetcher, LiveFetcher
from .graph import Deps, build_graph, review_card
from .llm import ClaudeClient
from .models import LeadIn, ReviewDecision
from .rubric import load_rubric
from .settings import RequestedMode, Settings
from .store import RunStore


class NotPendingError(RuntimeError):
    """The thread is not waiting for human review."""


class UnknownRunError(KeyError):
    pass


def build_crm(settings: Settings) -> CRM:
    if settings.crm_backend == "hubspot":
        return HubSpotCRM(settings.hubspot_token, log_path=settings.outbox_dir / "hubspot_dryrun.jsonl")
    return SQLiteCRM(settings.crm_db)


def build_notifier(settings: Settings) -> Notifier:
    if settings.slack_webhook_url:
        return SlackWebhookNotifier(settings.slack_webhook_url)
    return OutboxNotifier(settings.outbox_dir / "slack.jsonl")


class LeadQualifier:
    def __init__(
        self,
        settings: Settings,
        *,
        fetcher: Fetcher | None = None,
        llm: ClaudeClient | None = None,
        crm: CRM | None = None,
        notifier: Notifier | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.rubric = load_rubric(settings.rubric_path)
        self.store = RunStore(settings.app_db)
        self.crm = crm or build_crm(settings)
        self.notifier = notifier or build_notifier(settings)
        self.fetcher = fetcher or (
            LiveFetcher(settings.fetch_timeout) if settings.live_fetch else FixtureFetcher(settings.sites_dir)
        )
        if llm is None and settings.anthropic_api_key:
            llm = ClaudeClient(settings.anthropic_api_key, fallbacks=settings.fallbacks)
        self.llm = llm
        self._conn: sqlite3.Connection | None = None
        if checkpointer is None:
            # check_same_thread=False: FastAPI runs sync handlers/background tasks in a threadpool;
            # SqliteSaver serialises access with its own lock.
            self._conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
            checkpointer = SqliteSaver(self._conn)
        self.checkpointer = checkpointer
        self.graph = build_graph(
            Deps(
                rubric=self.rubric,
                fetcher=self.fetcher,
                crm=self.crm,
                notifier=self.notifier,
                llm=self.llm,
                model=settings.model,
                retry_initial_interval=settings.retry_initial_interval,
                retry_max_attempts=settings.retry_max_attempts,
            ),
            self.checkpointer,
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "LeadQualifier":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- intake
    def claim(self, raw: dict[str, Any], *, mode: RequestedMode | None = None) -> tuple[str, bool, dict]:
        """Validate + register a lead. Returns (thread_id, is_new, graph_input)."""
        lead = LeadIn.model_validate(raw)
        resolved = self.settings.resolve_mode(mode)
        thread_id = lead.thread_id()
        is_new = self.store.claim(
            thread_id, email=lead.email, name=lead.name, company=lead.company, source=lead.source, mode=resolved
        )
        graph_input = {"thread_id": thread_id, "mode": resolved, "lead_input": lead.model_dump()}
        return thread_id, is_new, graph_input

    def submit(self, raw: dict[str, Any], *, mode: RequestedMode | None = None) -> dict[str, Any]:
        """Idempotent: a lead with the same email+company is processed once."""
        thread_id, is_new, graph_input = self.claim(raw, mode=mode)
        if not is_new:
            return {**self.summary(thread_id), "duplicate": True}
        return {**self.run(thread_id, graph_input), "duplicate": False}

    def run(self, thread_id: str, graph_input: dict[str, Any] | None) -> dict[str, Any]:
        self.store.update(thread_id, status="running", error=None)
        try:
            self.graph.invoke(graph_input, self._config(thread_id))
        except Exception as exc:  # noqa: BLE001 - recorded, run can be retried from its checkpoint
            self.store.update(thread_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
            self._sync(thread_id, keep_status=True)
            if self.settings.extra.get("raise_errors"):
                traceback.print_exc()
            return self.summary(thread_id)
        return self._sync(thread_id)

    def retry(self, thread_id: str) -> dict[str, Any]:
        """Continue a failed run from its last checkpoint (completed nodes are not re-run)."""
        row = self.store.get(thread_id)
        if not row:
            raise UnknownRunError(thread_id)
        return self.run(thread_id, None)

    # ---------------------------------------------------------------- review
    def snapshot(self, thread_id: str):
        return self.graph.get_state(self._config(thread_id))

    def is_pending(self, thread_id: str) -> bool:
        snap = self.snapshot(thread_id)
        return "human_review" in (snap.next or ()) and bool(snap.interrupts)

    def resume(self, thread_id: str, decision: ReviewDecision | dict[str, Any]) -> dict[str, Any]:
        decision = ReviewDecision.model_validate(decision)  # validate before touching the graph
        if not self.store.get(thread_id):
            raise UnknownRunError(thread_id)
        if not self.is_pending(thread_id):
            raise NotPendingError(f"{thread_id} is not awaiting review")
        return self.run(thread_id, Command(resume=decision.model_dump()))

    def card(self, thread_id: str) -> dict[str, Any]:
        snap = self.snapshot(thread_id)
        if not snap.values:
            raise UnknownRunError(thread_id)
        values = dict(snap.values)
        card = review_card(values)
        card.update(
            status=(self.store.get(thread_id) or {}).get("status"),
            events=values.get("events", []),
            outcome=values.get("outcome"),
            decision=values.get("decision"),
            usage=values.get("usage", []),
            pending=self.is_pending(thread_id),
        )
        return card

    def pending(self, tier: str | None = None) -> list[dict[str, Any]]:
        cards = []
        for row in self.store.list(status="awaiting_review"):
            card = self.card(row["thread_id"])
            card["created_at"] = row["created_at"]
            if tier is None or card["score"].get("tier") == tier:
                cards.append(card)
        cards.sort(key=lambda c: (-(c["score"].get("score") or 0), c["thread_id"]))
        return cards

    # --------------------------------------------------------------- status
    def _sync(self, thread_id: str, keep_status: bool = False) -> dict[str, Any]:
        snap = self.snapshot(thread_id)
        values = snap.values or {}
        fields: dict[str, Any] = {}
        if not keep_status:
            if "human_review" in (snap.next or ()) and snap.interrupts:
                fields["status"] = "awaiting_review"
            elif values.get("status") in {"approved", "rejected", "disqualified"}:
                fields["status"] = values["status"]
                fields["error"] = None
        score = values.get("score") or {}
        if score:
            fields.update(tier=score.get("tier"), score=score.get("score"))
        if fields:
            self.store.update(thread_id, **fields)
        self.store.sync_usage(thread_id, values.get("usage", []))
        return self.summary(thread_id)

    def summary(self, thread_id: str) -> dict[str, Any]:
        row = self.store.get(thread_id)
        if not row:
            raise UnknownRunError(thread_id)
        return {
            "thread_id": thread_id,
            "status": row["status"],
            "tier": row["tier"],
            "score": row["score"],
            "mode": row["mode"],
            "name": row["name"],
            "company": row["company"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "cost_usd": row["cost_usd"],
            "error": row["error"],
        }

    def runs(self) -> list[dict[str, Any]]:
        return self.store.list()


def reset_state(settings: Settings) -> None:
    """Delete runtime state (checkpoints, run store, CRM mock, outbox)."""
    for path in (settings.var_dir, settings.outbox_dir):
        if path.exists():
            shutil.rmtree(path)
