"""Notifier adapters: Slack incoming webhook (Block Kit) or a local JSONL outbox."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ..enrich import TransientFetchError


class Notifier(ABC):
    name: str = "notifier"

    @abstractmethod
    def send(self, payload: dict[str, Any]) -> str:
        """Deliver a Slack-style payload ({"text", "blocks"}); return a delivery id."""


def build_slack_payload(
    *, lead: dict, score: dict, subject: str, body: str, action: str, crm_ref: str, thread_id: str, reviewer: str
) -> dict[str, Any]:
    tier = score.get("tier", "?")
    emoji = {"A": ":large_green_circle:", "B": ":large_yellow_circle:", "C": ":white_circle:"}.get(tier, ":red_circle:")
    top = sorted(score.get("criteria", []), key=lambda c: c["points"], reverse=True)[:3]
    reasons = "\n".join(f"• *{c['name']}* {c['points']:g}/{c['weight']:g}: {c['reason']}" for c in top)
    text = f"Tier {tier} lead approved: {lead.get('name')} ({lead.get('company')})"
    preview = body if len(body) <= 600 else body[:600] + "…"
    return {
        "text": text,
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"Tier {tier} lead: {lead.get('company') or lead.get('name')}"}},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Contact*\n{lead.get('name')} <mailto:{lead.get('email')}|{lead.get('email')}>"},
                    {"type": "mrkdwn", "text": f"*Score*\n{emoji} {score.get('score', 0):.0f}/100 (tier {tier})"},
                    {"type": "mrkdwn", "text": f"*Source*\n{lead.get('source', '')}"},
                    {"type": "mrkdwn", "text": f"*CRM*\n{crm_ref}"},
                ],
            },
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Why*\n{reasons}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Approved reply* ({action} by {reviewer})\n*{subject}*\n```{preview}```"}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": f"thread `{thread_id}` · leadq"}]},
        ],
    }


class OutboxNotifier(Notifier):
    """Default mock: appends each payload to outbox/slack.jsonl."""

    name = "outbox"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def send(self, payload: dict[str, Any]) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "channel": "mock-slack", "payload": payload}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with self.path.open(encoding="utf-8") as fh:
            line_no = sum(1 for _ in fh)
        return f"outbox:{self.path.name}#{line_no}"

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SlackWebhookNotifier(Notifier):
    name = "slack"

    def __init__(self, webhook_url: str, client: httpx.Client | None = None) -> None:
        self.webhook_url = webhook_url
        self.client = client or httpx.Client(timeout=10.0)

    def send(self, payload: dict[str, Any]) -> str:
        try:
            resp = self.client.post(self.webhook_url, json=payload)
        except httpx.TransportError as exc:
            raise TransientFetchError(f"slack webhook: {exc}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientFetchError(f"slack webhook HTTP {resp.status_code}")
        resp.raise_for_status()
        return "slack:ok"
