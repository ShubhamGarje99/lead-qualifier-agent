import json

import pytest

from leadq.adapters.crm import SQLiteCRM
from leadq.enrich import FixtureFetcher, TransientFetchError
from leadq.models import ReviewDecision
from leadq.service import LeadQualifier, NotPendingError

from .conftest import IDEAL_LEAD, REPO, STUDENT_LEAD


def _outbox(settings):
    path = settings.outbox_dir / "slack.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _crm(settings):
    return SQLiteCRM(settings.crm_db)


def test_run_pauses_at_human_review(qualifier):
    r = qualifier.submit(IDEAL_LEAD)
    assert r["status"] == "awaiting_review" and r["tier"] == "A" and not r["duplicate"]
    snap = qualifier.snapshot(r["thread_id"])
    assert snap.next == ("human_review",)
    payload = snap.interrupts[0].value
    assert payload["lead"]["email"] == IDEAL_LEAD["email"]
    assert payload["score"]["tier"] == "A" and payload["draft"]["subject"]
    assert payload["actions"] == ["approve", "edit", "reject"]
    nodes = [e["node"] for e in snap.values["events"]]
    assert nodes == ["normalize", "enrich", "extract_signals", "score", "route", "draft_email"]
    assert [c["thread_id"] for c in qualifier.pending()] == [r["thread_id"]]


def test_resume_approve_writes_crm_and_outbox(qualifier, settings):
    tid = qualifier.submit(IDEAL_LEAD)["thread_id"]
    draft = qualifier.card(tid)["draft"]
    result = qualifier.resume(tid, {"action": "approve", "reviewer": "shubham"})
    assert result["status"] == "approved"
    crm = _crm(settings)
    [contact] = crm.contacts()
    assert contact["email"] == IDEAL_LEAD["email"] and contact["tier"] == "A"
    assert len(crm.deals()) == 1
    notes = {n["kind"]: n["body"] for n in crm.notes()}
    assert draft["subject"] in notes["outbound_email"] and "[approve by shubham]" in notes["outbound_email"]
    [entry] = _outbox(settings)
    assert entry["payload"]["text"].startswith("Tier A lead approved")
    assert any(b["type"] == "header" for b in entry["payload"]["blocks"])
    assert qualifier.pending() == []
    with pytest.raises(NotPendingError):
        qualifier.resume(tid, {"action": "approve"})


def test_resume_reject_writes_nothing(qualifier, settings):
    tid = qualifier.submit(IDEAL_LEAD)["thread_id"]
    result = qualifier.resume(tid, ReviewDecision(action="reject", reason="existing customer"))
    assert result["status"] == "rejected"
    assert _crm(settings).contacts() == [] and _crm(settings).notes() == []
    assert _outbox(settings) == []
    card = qualifier.card(tid)
    assert card["outcome"] == {"archived": True, "reason": "existing customer"}


def test_edit_path_uses_reviewer_text(qualifier, settings):
    tid = qualifier.submit(IDEAL_LEAD)["thread_id"]
    with pytest.raises(ValueError):
        ReviewDecision(action="edit", subject="", body="")
    qualifier.resume(tid, {"action": "edit", "subject": "Your inbound routing", "body": "Hi Maya, edited body."})
    notes = {n["kind"]: n["body"] for n in _crm(settings).notes()}
    assert notes["outbound_email"].startswith("Subject: Your inbound routing\n\nHi Maya, edited body.")
    assert "[edit by" in notes["outbound_email"]
    assert "Your inbound routing" in json.dumps(_outbox(settings)[0]["payload"])


def test_disqualified_lead_is_archived_without_review(qualifier, settings):
    r = qualifier.submit(STUDENT_LEAD)
    assert r["status"] == "disqualified" and r["tier"] == "disqualified"
    snap = qualifier.snapshot(r["thread_id"])
    assert snap.next == () and "draft" not in snap.values
    assert snap.values["outcome"]["archived"] is True
    assert _crm(settings).contacts() == [] and _outbox(settings) == []


def test_idempotent_submission(qualifier):
    first = qualifier.submit(IDEAL_LEAD)
    again = qualifier.submit({**IDEAL_LEAD, "email": IDEAL_LEAD["email"].upper(), "message": "different text"})
    assert again["duplicate"] is True and again["thread_id"] == first["thread_id"]
    assert len(qualifier.runs()) == 1


def test_checkpoint_survives_restart(settings):
    q1 = LeadQualifier(settings)
    tid = q1.submit(IDEAL_LEAD)["thread_id"]
    q1.close()  # simulate the process exiting while the run waits for approval

    q2 = LeadQualifier(settings)  # new graph + new SqliteSaver on the same DB file
    try:
        assert [c["thread_id"] for c in q2.pending()] == [tid]
        assert q2.resume(tid, {"action": "approve"})["status"] == "approved"
        assert len(_crm(settings).contacts()) == 1
    finally:
        q2.close()


class FlakyFetcher(FixtureFetcher):
    def __init__(self, failures: int) -> None:
        super().__init__(REPO / "fixtures" / "sites")
        self.failures = failures
        self.calls = 0

    def fetch(self, domain, url):
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientFetchError(f"simulated timeout #{self.calls}")
        return super().fetch(domain, url)


def test_retry_policy_recovers_flaky_fetch(settings):
    fetcher = FlakyFetcher(failures=2)
    with LeadQualifier(settings, fetcher=fetcher) as q:
        r = q.submit(IDEAL_LEAD)
    assert fetcher.calls == 3  # two transient failures retried by RetryPolicy, third succeeds
    assert r["status"] == "awaiting_review" and r["tier"] == "A"


def test_exhausted_retries_fail_then_resume_from_checkpoint(settings):
    fetcher = FlakyFetcher(failures=10)
    with LeadQualifier(settings, fetcher=fetcher) as q:
        r = q.submit(IDEAL_LEAD)
        assert r["status"] == "failed" and "TransientFetchError" in r["error"]
        assert fetcher.calls == settings.retry_max_attempts
        fetcher.failures = 0  # the site comes back
        r2 = q.retry(r["thread_id"])
    assert r2["status"] == "awaiting_review" and r2["error"] is None
