"""FastAPI app: lead webhook + minimal human approval UI."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from ..models import ReviewDecision
from ..service import LeadQualifier, NotPendingError, UnknownRunError
from ..settings import MissingAPIKeyError, Settings

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
SIGNATURE_HEADER = "X-Leadq-Signature"


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    if not header:
        return False
    return hmac.compare_digest(sign(secret, body), header.strip())


def create_app(settings: Settings | None = None, qualifier: LeadQualifier | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    q = qualifier or LeadQualifier(settings)
    app = FastAPI(title="leadq", version="0.1.0", description="Inbound lead qualifier with human-in-the-loop approval")
    app.state.qualifier = q

    def mode_label() -> str:
        try:
            return settings.resolve_mode()
        except MissingAPIKeyError:
            return "llm (missing key)"

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/approvals", status_code=303)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "mode": mode_label()}

    # ------------------------------------------------------------- webhook
    @app.post("/webhook/lead", status_code=202)
    async def webhook_lead(
        request: Request,
        background: BackgroundTasks,
        x_leadq_signature: str | None = Header(default=None, alias=SIGNATURE_HEADER),
    ) -> JSONResponse:
        body = await request.body()
        if settings.webhook_secret and not verify_signature(settings.webhook_secret, body, x_leadq_signature):
            raise HTTPException(status_code=401, detail="invalid or missing signature")
        try:
            payload = json.loads(body or b"null")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="body must be JSON") from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="body must be a JSON object")
        try:
            thread_id, is_new, graph_input = q.claim(payload)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": json.loads(exc.json(include_url=False))})
        except MissingAPIKeyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        if not is_new:
            row = q.store.get(thread_id) or {}
            return JSONResponse(
                status_code=200,
                content={"thread_id": thread_id, "status": row.get("status"), "tier": row.get("tier"), "duplicate": True},
            )
        # Ack fast; run the graph after the response (Starlette runs sync tasks in a threadpool).
        background.add_task(q.run, thread_id, graph_input)
        return JSONResponse(
            status_code=202,
            content={"thread_id": thread_id, "status": "queued", "duplicate": False,
                     "status_url": f"/api/runs/{thread_id}", "review_url": f"/approvals#{thread_id}"},
        )

    # ----------------------------------------------------------------- API
    @app.get("/api/runs")
    def api_runs() -> dict:
        return {"runs": q.runs(), "totals": q.store.totals()}

    @app.get("/api/runs/{thread_id}")
    def api_run(thread_id: str) -> dict:
        try:
            return {"summary": q.summary(thread_id), "card": q.card(thread_id)}
        except UnknownRunError:
            raise HTTPException(status_code=404, detail="unknown thread") from None

    @app.post("/api/runs/{thread_id}/decision")
    def api_decision(thread_id: str, decision: ReviewDecision) -> dict:
        try:
            return q.resume(thread_id, decision)
        except UnknownRunError:
            raise HTTPException(status_code=404, detail="unknown thread") from None
        except NotPendingError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    # ------------------------------------------------------------------ UI
    @app.get("/approvals", response_class=HTMLResponse)
    def approvals(request: Request, flash: str | None = None) -> HTMLResponse:
        cards = q.pending()
        recent = [r for r in q.runs() if r["status"] != "awaiting_review"][:15]
        return TEMPLATES.TemplateResponse(
            request,
            "approvals.html",
            {"cards": cards, "recent": recent, "flash": flash, "mode": mode_label(), "totals": q.store.totals(),
             "crm_backend": settings.crm_backend, "notifier": q.notifier.name},
        )

    def _redirect(msg: str) -> RedirectResponse:
        return RedirectResponse(f"/approvals?flash={quote(msg)}", status_code=303)

    @app.post("/approvals/{thread_id}/approve")
    def approve(
        thread_id: str,
        subject: str = Form(...),
        body: str = Form(...),
        reviewer: str = Form("reviewer"),
    ) -> RedirectResponse:
        try:
            draft = q.card(thread_id)["draft"]
        except UnknownRunError:
            raise HTTPException(status_code=404, detail="unknown thread") from None
        body_norm = body.replace("\r\n", "\n").strip()
        edited = subject.strip() != draft["subject"].strip() or body_norm != draft["body"].strip()
        decision = ReviewDecision(
            action="edit" if edited else "approve",
            subject=subject.strip() if edited else None,
            body=body_norm if edited else None,
            reviewer=reviewer.strip() or "reviewer",
        )
        try:
            result = q.resume(thread_id, decision)
        except NotPendingError as exc:
            return _redirect(str(exc))
        return _redirect(f"{'Edited and approved' if edited else 'Approved'} {thread_id}: {result['status']} (CRM + Slack written)")

    @app.post("/approvals/{thread_id}/reject")
    def reject(thread_id: str, reason: str = Form(""), reviewer: str = Form("reviewer")) -> RedirectResponse:
        try:
            result = q.resume(thread_id, ReviewDecision(action="reject", reason=reason.strip() or None, reviewer=reviewer.strip() or "reviewer"))
        except UnknownRunError:
            raise HTTPException(status_code=404, detail="unknown thread") from None
        except NotPendingError as exc:
            return _redirect(str(exc))
        return _redirect(f"Rejected {thread_id}: {result['status']} (nothing written)")

    return app
