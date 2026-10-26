"""`leadq` command-line interface."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .adapters.crm import SQLiteCRM
from .adapters.notify import OutboxNotifier
from .models import ReviewDecision
from .service import LeadQualifier, NotPendingError, UnknownRunError, reset_state
from .settings import MissingAPIKeyError, Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Inbound lead qualifier agent (LangGraph + human-in-the-loop).")
console = Console()

TIER_STYLE = {"A": "bold green", "B": "yellow", "C": "cyan", "disqualified": "red"}
STATUS_STYLE = {"awaiting_review": "bold magenta", "approved": "green", "rejected": "red", "disqualified": "dim red", "failed": "bold red"}

ModeOpt = typer.Option(None, "--mode", "-m", help="auto (default: LLM if ANTHROPIC_API_KEY is set) | offline | llm")


def _settings(mode: str | None = None, live: bool = False) -> Settings:
    s = Settings.from_env()
    overrides: dict = {}
    if mode:
        if mode not in {"auto", "offline", "llm"}:
            raise typer.BadParameter("mode must be auto, offline or llm")
        overrides["requested_mode"] = mode
    if live:
        overrides["live_fetch"] = True
    return s.with_overrides(**overrides) if overrides else s


def _qualifier(settings: Settings) -> LeadQualifier:
    return LeadQualifier(settings)


def _fail(msg: str, code: int = 2) -> None:
    console.print(f"[bold red]error:[/] {msg}")
    raise typer.Exit(code)


def _styled(text: str, style: str | None) -> str:
    return f"[{style}]{escape(text)}[/]" if style else escape(text)


def _tier(t: str | None) -> str:
    return _styled("DQ" if t == "disqualified" else (t or "-"), TIER_STYLE.get(t or ""))


def _status(s: str | None) -> str:
    return _styled((s or "-").replace("_", " "), STATUS_STYLE.get(s or ""))


def _money(x: float | None) -> str:
    return f"${(x or 0):.4f}"


def _mode_banner(settings: Settings, mode: str) -> None:
    if mode == "offline":
        console.print(
            Panel.fit(
                "[bold]OFFLINE mode[/]: heuristic signal extraction + template drafts (no LLM calls, $0).\n"
                "Add ANTHROPIC_API_KEY to .env for Claude extraction/drafting.",
                border_style="yellow",
            )
        )
    else:
        fb = "on" if settings.fallbacks else "off"
        console.print(Panel.fit(f"[bold]LLM mode[/]: {settings.model} (structured outputs, refusal fallback {fb})", border_style="green"))


# --------------------------------------------------------------------------
@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="CSV with name,email,company,title,website,message,source"),
    mode: Optional[str] = ModeOpt,
    live: bool = typer.Option(False, "--live", help="Fetch real homepages over HTTP instead of offline fixtures"),
) -> None:
    """Qualify every lead in a CSV. Idempotent: re-running skips leads already seen."""
    settings = _settings(mode, live)
    try:
        resolved = settings.resolve_mode()
    except MissingAPIKeyError as exc:
        _fail(str(exc))
    _mode_banner(settings, resolved)
    table = Table(title=f"Ingest: {path.name}", box=box.SIMPLE_HEAD)
    table.add_column("thread", no_wrap=True)
    table.add_column("lead", min_width=12)
    table.add_column("tier", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("status")
    table.add_column("tokens", justify="right", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)
    totals = {"n": 0, "tok": 0, "cost": 0.0, "dup": 0, "bad": 0}
    with _qualifier(settings) as q, open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                r = q.submit(row)
            except ValidationError as exc:
                totals["bad"] += 1
                console.print(f"[red]row {i} skipped:[/] {exc.errors()[0]['loc'][0]}: {exc.errors()[0]['msg']}")
                continue
            totals["n"] += 1
            totals["dup"] += int(r["duplicate"])
            tok = (r["input_tokens"] or 0) + (r["output_tokens"] or 0)
            if not r["duplicate"]:
                totals["tok"] += tok
                totals["cost"] += r["cost_usd"] or 0
            status = "duplicate (skipped)" if r["duplicate"] else r["status"]
            table.add_row(
                r["thread_id"], escape(f"{r['name']} · {r['company'] or '-'}"), _tier(r["tier"]), f"{r['score'] or 0:.1f}",
                _status(status), str(tok), _money(r["cost_usd"]),
            )
            if r.get("error"):
                console.print(f"[red]{r['thread_id']} failed:[/] {r['error']}")
    console.print(table)
    console.print("tier DQ = disqualified (archived, no draft)")
    console.print(
        f"Processed {totals['n']} leads ({totals['dup']} duplicates, {totals['bad']} invalid rows). "
        f"Tokens: {totals['tok']:,}. LLM cost this run: [bold]{_money(totals['cost'])}[/]."
    )
    console.print("Next: [bold]leadq pending[/], [bold]leadq review <thread>[/], or [bold]leadq serve[/] -> http://127.0.0.1:8000/approvals")


@app.command()
def pending(tier: Optional[str] = typer.Option(None, help="Filter by tier (A/B/C)")) -> None:
    """List runs paused at human review."""
    with _qualifier(_settings()) as q:
        cards = q.pending(tier)
    if not cards:
        console.print("No runs awaiting review.")
        return
    table = Table(title="Awaiting human review", box=box.SIMPLE_HEAD)
    table.add_column("thread", no_wrap=True)
    table.add_column("lead")
    table.add_column("tier", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("draft subject")
    for c in cards:
        table.add_row(c["thread_id"], escape(f"{c['lead']['name']} · {c['lead']['company'] or '-'}"), _tier(c["score"]["tier"]),
                      f"{c['score']['score']:.1f}", escape(c["draft"]["subject"]))
    console.print(table)


def _print_card(card: dict) -> None:
    lead, score, sig, draft = card["lead"], card["score"], card["signals"], card["draft"]
    e = {k: escape(str(v or "")) for k, v in lead.items()}  # lead data is untrusted: never parse it as markup
    head = (
        f"[bold]{e['name']}[/] <{e['email']}>  {e.get('title', '')}\n"
        f"[bold]{e['company']}[/]  {e.get('website', '')}   source: {e.get('source', '')}\n\n"
        f"[italic]\"{e.get('message') or '(no message)'}\"[/]"
    )
    console.print(Panel(head, title=f"{card['thread_id']}  ·  tier {_tier(score['tier'])}  ·  {score['score']:.1f}/100", border_style="blue"))
    t = Table(title="Score breakdown", show_header=True)
    for col in ("criterion", "points", "value", "reason"):
        t.add_column(col)
    for c in score["criteria"]:
        t.add_row(c["name"], f"{c['points']:g}/{c['weight']:g}", escape(c["value"]), escape(c["reason"]))
    console.print(t)
    for line in score.get("disqualifiers", []) + score.get("caps", []):
        console.print(f"[red]• {escape(line)}[/]")
    console.print(escape(
        f"signals ({sig['extracted_by']}): industry={sig['industry']} size={sig['employee_band']} region={sig['region']} "
        f"model={sig['business_model']} intent={sig['buying_intent']} budget={sig['budget']} tech={', '.join(sig['tech_stack']) or '-'}"
    ))
    if draft:
        console.print(Panel(f"[bold]Subject:[/] {escape(draft['subject'])}\n\n{escape(draft['body'])}",
                            title=f"Draft ({draft['generated_by']})", border_style="green"))
    console.print(f"Run cost so far: {_money(card.get('cost_usd'))}")


def _edit_draft(draft: dict) -> tuple[str, str]:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor:
        text = typer.edit(f"Subject: {draft['subject']}\n\n{draft['body']}\n", extension=".txt")
        if text:
            first, _, rest = text.strip().partition("\n")
            subject = first.removeprefix("Subject:").strip() or draft["subject"]
            return subject, rest.strip() or draft["body"]
        return draft["subject"], draft["body"]
    subject = typer.prompt("Subject", default=draft["subject"])
    console.print("New body: type lines, finish with a single '.' line (just '.' keeps the current body).")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == ".":
            break
        lines.append(line)
    return subject, ("\n".join(lines).strip() or draft["body"])


@app.command()
def review(
    thread_id: str,
    reviewer: str = typer.Option(os.environ.get("USER", "reviewer"), help="Name recorded on the decision"),
    approve: bool = typer.Option(False, "--approve", help="Approve without prompting"),
    reject: Optional[str] = typer.Option(None, "--reject", help="Reject without prompting, with this reason"),
) -> None:
    """Show a paused run and approve / edit / reject it (resumes the graph)."""
    with _qualifier(_settings()) as q:
        try:
            card = q.card(thread_id)
        except UnknownRunError:
            _fail(f"unknown thread {thread_id}")
        _print_card(card)
        if not card["pending"]:
            _fail(f"{thread_id} is not awaiting review (status: {card['status']})", code=1)
        if approve:
            decision = ReviewDecision(action="approve", reviewer=reviewer)
        elif reject is not None:
            decision = ReviewDecision(action="reject", reason=reject or None, reviewer=reviewer)
        else:
            choice = typer.prompt("[a]pprove / [e]dit / [r]eject / [s]kip", default="a").strip().lower()[:1]
            if choice == "s":
                console.print("Skipped; the run stays paused.")
                return
            if choice == "e":
                subject, body = _edit_draft(card["draft"])
                decision = ReviewDecision(action="edit", subject=subject, body=body, reviewer=reviewer)
            elif choice == "r":
                reason = typer.prompt("Reason", default="not a fit")
                decision = ReviewDecision(action="reject", reason=reason, reviewer=reviewer)
            else:
                decision = ReviewDecision(action="approve", reviewer=reviewer)
        try:
            result = q.resume(thread_id, decision)
        except NotPendingError as exc:
            _fail(str(exc), code=1)
        outcome = q.card(thread_id).get("outcome") or {}
    console.print(f"{thread_id}: {_status(result['status'])}  " + escape(json.dumps({k: v for k, v in outcome.items() if k != "final_body"})))


@app.command("approve-all")
def approve_all(
    tier: str = typer.Option("A", help="Approve every pending run in this tier"),
    reviewer: str = typer.Option(os.environ.get("USER", "reviewer")),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Bulk-approve pending drafts for one tier (drafts sent as written)."""
    with _qualifier(_settings()) as q:
        cards = q.pending(tier)
        if not cards:
            console.print(f"No pending tier {tier} runs.")
            return
        if not yes and not typer.confirm(f"Approve {len(cards)} tier {tier} drafts as written?"):
            raise typer.Abort()
        for c in cards:
            r = q.resume(c["thread_id"], ReviewDecision(action="approve", reviewer=reviewer))
            console.print(f"{c['thread_id']} {escape(c['lead']['company'])}: {_status(r['status'])}")


@app.command()
def runs() -> None:
    """All runs with status, tier and exact token/cost totals."""
    with _qualifier(_settings()) as q:
        rows = q.runs()
        totals = q.store.totals()
    table = Table(title="Runs", box=box.SIMPLE_HEAD)
    table.add_column("thread", no_wrap=True)
    table.add_column("lead")
    for col in ("mode", "status", "tier"):
        table.add_column(col, no_wrap=True)
    for col in ("score", "in tok", "out tok", "calls", "cost"):
        table.add_column(col, justify="right", no_wrap=True)
    for r in rows:
        table.add_row(r["thread_id"], escape(f"{r['name']} · {r['company'] or '-'}"), r["mode"], _status(r["status"]), _tier(r["tier"]),
                      f"{r['score'] or 0:.1f}", str(r["input_tokens"]), str(r["output_tokens"]), str(r["llm_calls"]), _money(r["cost_usd"]))
    console.print(table)
    n = max(totals["runs"], 1)
    console.print(f"{totals['runs']} runs · {totals['input_tokens']:,} in / {totals['output_tokens']:,} out tokens · "
                  f"total {_money(totals['cost_usd'])} · avg {_money(totals['cost_usd'] / n)}/lead")


@app.command()
def show(thread_id: str) -> None:
    """Full state of one run: card, timeline, outcome."""
    with _qualifier(_settings()) as q:
        try:
            card = q.card(thread_id)
        except UnknownRunError:
            _fail(f"unknown thread {thread_id}")
    _print_card(card)
    t = Table(title="Timeline")
    for col in ("at", "node", "event"):
        t.add_column(col)
    for e in card["events"]:
        t.add_row(e["at"], e["node"], escape(e["message"]))
    console.print(t)
    if card.get("outcome"):
        console.print_json(json.dumps({k: v for k, v in card["outcome"].items() if k != "final_body"}))


@app.command()
def retry(thread_id: str) -> None:
    """Resume a failed run from its last checkpoint."""
    with _qualifier(_settings()) as q:
        try:
            r = q.retry(thread_id)
        except UnknownRunError:
            _fail(f"unknown thread {thread_id}")
    console.print(f"{thread_id}: {_status(r['status'])} {r.get('error') or ''}")


@app.command()
def crm() -> None:
    """Show the local mock CRM (contacts, deals, notes)."""
    settings = _settings()
    db = SQLiteCRM(settings.crm_db)
    t = Table(title=f"CRM contacts ({settings.crm_db})")
    for col in ("id", "email", "company", "tier", "score", "industry", "size", "region", "updated"):
        t.add_column(col)
    for c in db.contacts():
        t.add_row(str(c["id"]), escape(c["email"]), escape(c["company"] or ""), _tier(c["tier"]), f"{c['score']:.1f}", c["industry"],
                  c["employee_band"], c["region"], c["updated_at"])
    console.print(t)
    d = Table(title="Deals")
    for col in ("id", "contact", "name", "stage", "tier", "thread"):
        d.add_column(col)
    for x in db.deals():
        d.add_row(str(x["id"]), str(x["contact_id"]), escape(x["name"]), x["stage"], _tier(x["tier"]), x["thread_id"])
    console.print(d)
    console.print(f"{len(db.notes())} notes (qualification + approved outbound email per lead)")


@app.command()
def outbox(n: int = typer.Option(5, help="Show the last N notifications")) -> None:
    """Show the mock Slack outbox (outbox/slack.jsonl)."""
    settings = _settings()
    entries = OutboxNotifier(settings.outbox_dir / "slack.jsonl").entries()
    if not entries:
        console.print("Outbox is empty.")
        return
    for e in entries[-n:]:
        blocks = e["payload"].get("blocks", [])
        lines = [e["payload"]["text"]]
        for b in blocks:
            if b.get("type") == "section" and "text" in b:
                lines.append(b["text"]["text"])
        console.print(Panel(escape("\n\n".join(lines)), title=f"{e['channel']} · {e['at']}", border_style="magenta"))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    mode: Optional[str] = ModeOpt,
) -> None:
    """Run the webhook + approval UI (FastAPI)."""
    import uvicorn

    from .web.app import create_app

    settings = _settings(mode)
    try:
        resolved = settings.resolve_mode()
    except MissingAPIKeyError as exc:
        _fail(str(exc))
    _mode_banner(settings, resolved)
    console.print(f"Approvals UI: http://{host}:{port}/approvals   Webhook: POST http://{host}:{port}/webhook/lead")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")


@app.command()
def reset(yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete local runtime state (var/ checkpoints, run store, CRM mock, outbox/)."""
    settings = _settings()
    if not yes and not typer.confirm(f"Delete {settings.var_dir} and {settings.outbox_dir}?"):
        raise typer.Abort()
    reset_state(settings)
    console.print("Runtime state cleared.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
