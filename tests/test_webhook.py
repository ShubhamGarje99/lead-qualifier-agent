import json

import pytest
from fastapi.testclient import TestClient

from leadq.adapters.crm import SQLiteCRM
from leadq.service import LeadQualifier
from leadq.web.app import SIGNATURE_HEADER, create_app, sign

from .conftest import IDEAL_LEAD


@pytest.fixture
def client(settings):
    q = LeadQualifier(settings)
    with TestClient(create_app(settings, q)) as c:
        yield c
    q.close()


@pytest.fixture
def signed_client(settings):
    s = settings.with_overrides(webhook_secret="s3cret")
    q = LeadQualifier(s)
    with TestClient(create_app(s, q)) as c:
        yield c
    q.close()


def test_webhook_accepts_and_runs_lead(client):
    r = client.post("/webhook/lead", json=IDEAL_LEAD)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and body["duplicate"] is False
    # TestClient completes background tasks before returning
    run = client.get(body["status_url"]).json()
    assert run["summary"]["status"] == "awaiting_review" and run["summary"]["tier"] == "A"


def test_webhook_validation(client):
    assert client.post("/webhook/lead", json={**IDEAL_LEAD, "email": "not-an-email"}).status_code == 422
    assert client.post("/webhook/lead", json={"email": "a@b.example"}).status_code == 422  # name missing
    assert client.post("/webhook/lead", content=b"{not json", headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/webhook/lead", json=["a", "list"]).status_code == 422
    assert client.get("/api/runs").json()["runs"] == []


def test_webhook_is_idempotent(client):
    first = client.post("/webhook/lead", json=IDEAL_LEAD).json()
    second = client.post("/webhook/lead", json={**IDEAL_LEAD, "message": "resent by the form provider"})
    assert second.status_code == 200
    assert second.json()["duplicate"] is True and second.json()["thread_id"] == first["thread_id"]
    assert len(client.get("/api/runs").json()["runs"]) == 1


def test_hmac_signature_required_when_secret_set(signed_client):
    raw = json.dumps(IDEAL_LEAD).encode()
    headers = {"content-type": "application/json"}
    assert signed_client.post("/webhook/lead", content=raw, headers=headers).status_code == 401
    bad = {**headers, SIGNATURE_HEADER: "sha256=" + "0" * 64}
    assert signed_client.post("/webhook/lead", content=raw, headers=bad).status_code == 401
    tampered = {**headers, SIGNATURE_HEADER: sign("s3cret", raw)}
    assert signed_client.post("/webhook/lead", content=raw + b" ", headers=tampered).status_code == 401
    good = {**headers, SIGNATURE_HEADER: sign("s3cret", raw)}
    assert signed_client.post("/webhook/lead", content=raw, headers=good).status_code == 202


def test_approval_page_and_form_resume(client, settings):
    tid = client.post("/webhook/lead", json=IDEAL_LEAD).json()["thread_id"]
    page = client.get("/approvals")
    assert page.status_code == 200
    assert "Maya Chen" in page.text and tid in page.text and "OFFLINE mode" in page.text
    card = client.get(f"/api/runs/{tid}").json()["card"]
    r = client.post(
        f"/approvals/{tid}/approve",
        data={"subject": card["draft"]["subject"], "body": card["draft"]["body"].replace("\n", "\r\n"), "reviewer": "web"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and "Approved" in r.headers["location"]
    assert len(SQLiteCRM(settings.crm_db).contacts()) == 1
    assert (settings.outbox_dir / "slack.jsonl").read_text().count("\n") == 1
    assert client.get(f"/api/runs/{tid}").json()["card"]["decision"]["action"] == "approve"  # unchanged text = approve
    again = client.post(f"/approvals/{tid}/reject", data={"reason": "late"}, follow_redirects=False)
    assert again.status_code == 303 and "not awaiting review" in again.headers["location"].replace("%20", " ")


def test_reject_via_form_and_json_api(client, settings):
    tid = client.post("/webhook/lead", json=IDEAL_LEAD).json()["thread_id"]
    r = client.post(f"/approvals/{tid}/reject", data={"reason": "duplicate account"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get(f"/api/runs/{tid}").json()["summary"]["status"] == "rejected"
    assert SQLiteCRM(settings.crm_db).contacts() == []
    assert client.post(f"/api/runs/{tid}/decision", json={"action": "approve"}).status_code == 409
    assert client.post("/api/runs/lead-unknown/decision", json={"action": "approve"}).status_code == 404
