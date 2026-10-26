"""The LangGraph state machine.

normalize -> enrich -> extract_signals -> score -> route
   route --(A/B/C)--> draft_email -> human_review [interrupt]
                         human_review --approve/edit--> write_crm -> notify_slack -> END
                         human_review --reject-------> archive -> END
   route --(disqualified)--> archive -> END

State is plain JSON-able dicts so checkpoints are portable and inspectable.
"""

from __future__ import annotations

import operator
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from .adapters.crm import CRM, ContactPayload, DealPayload, NotePayload
from .adapters.notify import Notifier, build_slack_payload
from .drafting import template_draft
from .enrich import Fetcher, TransientFetchError, enrich
from .heuristics import extract_signals as heuristic_signals
from .llm import ClaudeClient, LLMError, is_transient
from .models import Draft, EmailDraft, Lead, ReviewDecision, ScoreResult, Signals, SignalsCore, SiteProfile
from .prompts import drafting_system, drafting_user, extraction_system, extraction_user
from .rubric import Rubric, score_lead
from .settings import DEFAULT_MODEL


class LeadState(TypedDict, total=False):
    thread_id: str
    mode: Literal["offline", "llm"]
    lead_input: dict[str, Any]
    lead: dict[str, Any]
    site: dict[str, Any]
    signals: dict[str, Any]
    score: dict[str, Any]
    route: str
    draft: dict[str, Any]
    decision: dict[str, Any]
    status: str
    outcome: dict[str, Any]
    # Reducers: every node appends; nothing is overwritten.
    usage: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class Deps:
    rubric: Rubric
    fetcher: Fetcher
    crm: CRM
    notifier: Notifier
    llm: ClaudeClient | None = None
    model: str = DEFAULT_MODEL
    retry_initial_interval: float = 0.5
    retry_max_attempts: int = 3


def _event(node: str, message: str) -> list[dict[str, Any]]:
    return [{"node": node, "at": datetime.now(UTC).isoformat(timespec="seconds"), "message": message}]


def _fetch_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (TransientFetchError, httpx.TransportError))


def _write_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (sqlite3.OperationalError, httpx.TransportError, TransientFetchError))


def review_card(state: dict[str, Any]) -> dict[str, Any]:
    """What a reviewer needs to decide: the interrupt payload."""
    site = state.get("site", {})
    return {
        "thread_id": state.get("thread_id"),
        "mode": state.get("mode"),
        "lead": state.get("lead", {}),
        "site": {k: site.get(k) for k in ("url", "found", "source", "title", "meta_description")},
        "signals": state.get("signals", {}),
        "score": state.get("score", {}),
        "draft": state.get("draft", {}),
        "cost_usd": round(sum(u.get("cost_usd", 0.0) for u in state.get("usage", [])), 6),
        "actions": ["approve", "edit", "reject"],
    }


def build_graph(deps: Deps, checkpointer: Any):
    rubric = deps.rubric

    # ---------------------------------------------------------------- nodes
    def normalize(state: LeadState) -> dict:
        lead = Lead.normalize(state["lead_input"])
        note = f"domain={lead.domain or 'n/a'}" + (" (free email)" if lead.free_email else "")
        return {"lead": lead.model_dump(), "status": "running", "events": _event("normalize", note)}

    def enrich_node(state: LeadState) -> dict:
        lead = Lead.model_validate(state["lead"])
        site = enrich(deps.fetcher, lead.domain, lead.website)
        msg = f"{site.source}: {site.title!r}" if site.found else f"no homepage ({site.error})"
        return {"site": site.model_dump(), "events": _event("enrich", msg)}

    def extract_node(state: LeadState) -> dict:
        lead = Lead.model_validate(state["lead"])
        site = SiteProfile.model_validate(state["site"])
        usage: list[dict] = []
        if state.get("mode") == "llm" and deps.llm is not None:
            try:
                core, rec = deps.llm.parse(
                    node="extract_signals",
                    model=deps.model,
                    system=extraction_system(rubric),
                    user=extraction_user(lead, site),
                    output_format=SignalsCore,
                    effort="low",
                )
                data = core.model_dump()
                data["tech_stack"] = sorted({t.strip().lower() for t in data["tech_stack"] if t.strip()})
                signals = Signals(**data, extracted_by="llm")
                return {
                    "signals": signals.model_dump(),
                    "usage": [rec.model_dump()],
                    "events": _event("extract_signals", f"llm {rec.model}: {rec.input_tokens}+{rec.output_tokens} tok"),
                }
            except LLMError as exc:  # refusal / truncation: degrade gracefully, keep the cost
                if exc.usage:
                    usage.append(exc.usage.model_dump())
                signals = heuristic_signals(lead, site, rubric)
                return {
                    "signals": signals.model_dump(),
                    "usage": usage,
                    "events": _event("extract_signals", f"LLM failed ({exc}); used heuristic fallback"),
                }
        signals = heuristic_signals(lead, site, rubric)
        note = "offline heuristic" if state.get("mode") != "llm" else "heuristic (no LLM client available)"
        return {"signals": signals.model_dump(), "usage": usage, "events": _event("extract_signals", note)}

    def score_node(state: LeadState) -> dict:
        result = score_lead(Signals.model_validate(state["signals"]), rubric)
        return {"score": result.model_dump(), "events": _event("score", result.summary)}

    def route_node(state: LeadState) -> dict:
        tier = state["score"]["tier"]
        route = "archive" if tier == "disqualified" else "draft"
        return {"route": route, "events": _event("route", f"tier {tier} -> {route}")}

    def draft_node(state: LeadState) -> dict:
        lead = Lead.model_validate(state["lead"])
        site = SiteProfile.model_validate(state["site"])
        signals = Signals.model_validate(state["signals"])
        score = ScoreResult.model_validate(state["score"])
        usage: list[dict] = []
        if state.get("mode") == "llm" and deps.llm is not None:
            try:
                email, rec = deps.llm.parse(
                    node="draft_email",
                    model=deps.model,
                    system=drafting_system(rubric),
                    user=drafting_user(lead, site, signals, score),
                    output_format=EmailDraft,
                    effort="medium",
                )
                draft = Draft(subject=email.subject.strip(), body=email.body.strip(), generated_by="llm", model=rec.model)
                return {
                    "draft": draft.model_dump(),
                    "usage": [rec.model_dump()],
                    "events": _event("draft_email", f"llm {rec.model}: {rec.input_tokens}+{rec.output_tokens} tok"),
                }
            except LLMError as exc:
                if exc.usage:
                    usage.append(exc.usage.model_dump())
                draft = template_draft(lead, site, signals, score, rubric)
                return {
                    "draft": draft.model_dump(),
                    "usage": usage,
                    "events": _event("draft_email", f"LLM failed ({exc}); used template fallback"),
                }
        draft = template_draft(lead, site, signals, score, rubric)
        return {"draft": draft.model_dump(), "usage": usage, "events": _event("draft_email", "offline template")}

    def human_review(state: LeadState) -> dict:
        # Pauses the run; the checkpointer persists everything up to here.
        # Resumed with Command(resume={"action": "approve"|"edit"|"reject", ...}).
        raw = interrupt(review_card(state))
        decision = ReviewDecision.model_validate(raw)
        msg = f"{decision.action} by {decision.reviewer}"
        if decision.reason:
            msg += f": {decision.reason}"
        return {"decision": decision.model_dump(), "events": _event("human_review", msg)}

    def write_crm(state: LeadState) -> dict:
        lead, score, signals = state["lead"], state["score"], state["signals"]
        decision, draft = state["decision"], state["draft"]
        subject = decision.get("subject") if decision["action"] == "edit" else draft["subject"]
        body = decision.get("body") if decision["action"] == "edit" else draft["body"]
        first, _, last = lead["name"].partition(" ")
        contact_id = deps.crm.upsert_contact(
            ContactPayload(
                email=lead["email"], first_name=first, last_name=last, company=lead["company"],
                title=lead.get("title", ""), website=lead.get("website", ""), lead_source=lead.get("source", ""),
                tier=score["tier"], score=score["score"], industry=signals["industry"],
                employee_band=signals["employee_band"], region=signals["region"],
            )
        )
        deal_id = None
        if score["tier"] in {"A", "B"}:
            deal_id = deps.crm.create_deal(
                contact_id,
                DealPayload(
                    thread_id=state["thread_id"], name=f"{lead['company'] or lead['name']} - inbound",
                    stage="qualifiedtobuy", tier=score["tier"], score=score["score"],
                ),
            )
        reasons = "\n".join(f"- {c['name']}: {c['points']:g}/{c['weight']:g} ({c['reason']})" for c in score["criteria"])
        qual_note = deps.crm.add_note(
            contact_id,
            NotePayload(thread_id=state["thread_id"], kind="qualification",
                        body=f"{score['summary']}\n{reasons}\nExtracted by: {signals['extracted_by']}"),
        )
        email_note = deps.crm.add_note(
            contact_id,
            NotePayload(thread_id=state["thread_id"], kind="outbound_email",
                        body=f"Subject: {subject}\n\n{body}\n\n[{decision['action']} by {decision.get('reviewer')}]"),
        )
        outcome = {
            "crm": deps.crm.name, "contact_id": contact_id, "deal_id": deal_id,
            "note_ids": [qual_note, email_note], "final_subject": subject, "final_body": body,
        }
        return {"outcome": outcome, "events": _event("write_crm", f"{deps.crm.name} contact {contact_id}, deal {deal_id}")}

    def notify_slack(state: LeadState) -> dict:
        outcome = dict(state["outcome"])
        decision = state["decision"]
        payload = build_slack_payload(
            lead=state["lead"], score=state["score"], subject=outcome["final_subject"], body=outcome["final_body"],
            action=decision["action"], crm_ref=f"{outcome['crm']} contact #{outcome['contact_id']}",
            thread_id=state["thread_id"], reviewer=decision.get("reviewer", "reviewer"),
        )
        delivery = deps.notifier.send(payload)
        outcome["notification"] = delivery
        return {"outcome": outcome, "status": "approved", "events": _event("notify_slack", delivery)}

    def archive(state: LeadState) -> dict:
        decision = state.get("decision") or {}
        if decision.get("action") == "reject":
            status, reason = "rejected", decision.get("reason") or "rejected by reviewer"
        else:
            score = state["score"]
            status, reason = "disqualified", "; ".join(score.get("disqualifiers") or []) or score.get("summary", "")
        return {
            "status": status,
            "outcome": {"archived": True, "reason": reason},
            "events": _event("archive", f"{status}: {reason}"),
        }

    # ---------------------------------------------------------------- wiring
    fetch_retry = RetryPolicy(
        initial_interval=deps.retry_initial_interval, max_attempts=deps.retry_max_attempts, retry_on=_fetch_retryable
    )
    llm_retry = RetryPolicy(
        initial_interval=max(deps.retry_initial_interval, 0.01) * 2, max_attempts=deps.retry_max_attempts,
        retry_on=is_transient,
    )
    write_retry = RetryPolicy(
        initial_interval=deps.retry_initial_interval, max_attempts=deps.retry_max_attempts, retry_on=_write_retryable
    )

    g = StateGraph(LeadState)
    g.add_node("normalize", normalize)
    g.add_node("enrich", enrich_node, retry_policy=fetch_retry)
    g.add_node("extract_signals", extract_node, retry_policy=llm_retry)
    g.add_node("score", score_node)
    g.add_node("route", route_node)
    g.add_node("draft_email", draft_node, retry_policy=llm_retry)
    g.add_node("human_review", human_review)
    g.add_node("write_crm", write_crm, retry_policy=write_retry)
    g.add_node("notify_slack", notify_slack, retry_policy=write_retry)
    g.add_node("archive", archive)

    g.add_edge(START, "normalize")
    g.add_edge("normalize", "enrich")
    g.add_edge("enrich", "extract_signals")
    g.add_edge("extract_signals", "score")
    g.add_edge("score", "route")
    g.add_conditional_edges("route", lambda s: s["route"], {"draft": "draft_email", "archive": "archive"})
    g.add_edge("draft_email", "human_review")
    g.add_conditional_edges(
        "human_review",
        lambda s: "archive" if s["decision"]["action"] == "reject" else "write_crm",
        {"write_crm": "write_crm", "archive": "archive"},
    )
    g.add_edge("write_crm", "notify_slack")
    g.add_edge("notify_slack", END)
    g.add_edge("archive", END)
    return g.compile(checkpointer=checkpointer)
