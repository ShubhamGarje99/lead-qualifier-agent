import json

import httpx
import pytest

from leadq.adapters.crm import ContactPayload, DealPayload, HubSpotCRM, NotePayload, SQLiteCRM
from leadq.adapters.notify import OutboxNotifier, SlackWebhookNotifier, build_slack_payload
from leadq.enrich import TransientFetchError

CONTACT = ContactPayload(email="maya@ledgerloop.example", first_name="Maya", last_name="Chen", company="LedgerLoop",
                         tier="A", score=92.5, industry="fintech")


def test_sqlite_crm_is_idempotent(tmp_path):
    crm = SQLiteCRM(tmp_path / "crm.sqlite")
    cid = crm.upsert_contact(CONTACT)
    assert crm.upsert_contact(CONTACT.model_copy(update={"score": 95.0})) == cid
    deal = DealPayload(thread_id="t1", name="LedgerLoop - inbound", stage="qualifiedtobuy", tier="A", score=95)
    assert crm.create_deal(cid, deal) == crm.create_deal(cid, deal)
    note = NotePayload(thread_id="t1", kind="qualification", body="x")
    assert crm.add_note(cid, note) == crm.add_note(cid, note)
    assert len(crm.contacts()) == 1 and crm.contacts()[0]["score"] == 95.0
    assert len(crm.deals()) == 1 and len(crm.notes()) == 1


def test_hubspot_dry_run_maps_payloads_without_network(tmp_path):
    hs = HubSpotCRM(token=None, log_path=tmp_path / "hs.jsonl")
    assert hs.dry_run
    cid = hs.upsert_contact(CONTACT)
    hs.create_deal(cid, DealPayload(thread_id="t1", name="d", stage="qualifiedtobuy", tier="A", score=90))
    hs.add_note(cid, NotePayload(thread_id="t1", kind="qualification", body="why"))
    urls = [e["url"] for e in hs.dry_run_log]
    assert urls == [
        "https://api.hubapi.com/crm/v3/objects/contacts/batch/upsert",
        "https://api.hubapi.com/crm/v3/objects/deals",
        "https://api.hubapi.com/crm/v3/objects/notes",
    ]
    contact_input = hs.dry_run_log[0]["body"]["inputs"][0]
    assert contact_input["idProperty"] == "email" and contact_input["properties"]["lifecyclestage"] == "salesqualifiedlead"
    assert hs.dry_run_log[1]["body"]["associations"][0]["types"][0]["associationTypeId"] == 3
    assert hs.dry_run_log[2]["body"]["associations"][0]["types"][0]["associationTypeId"] == 202
    assert len((tmp_path / "hs.jsonl").read_text().splitlines()) == 3


def _payload():
    return build_slack_payload(
        lead={"name": "Maya", "email": "m@x.example", "company": "LedgerLoop", "source": "form"},
        score={"tier": "A", "score": 92.5, "criteria": [{"name": "intent", "points": 25, "weight": 25, "reason": "high"}]},
        subject="Hi", body="Body", action="approve", crm_ref="sqlite contact #1", thread_id="t1", reviewer="r",
    )


def test_outbox_notifier_appends_jsonl(tmp_path):
    n = OutboxNotifier(tmp_path / "outbox" / "slack.jsonl")
    assert n.send(_payload()) == "outbox:slack.jsonl#1"
    assert n.send(_payload()) == "outbox:slack.jsonl#2"
    assert n.entries()[0]["payload"]["blocks"][0]["type"] == "header"


def test_slack_webhook_posts_block_kit_via_mock_transport():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, text="ok")

    n = SlackWebhookNotifier("https://hooks.slack.invalid/T/B/X", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert n.send(_payload()) == "slack:ok"
    assert seen[0]["blocks"][1]["type"] == "section"

    flaky = SlackWebhookNotifier("https://hooks.slack.invalid/x",
                                 client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))))
    with pytest.raises(TransientFetchError):
        flaky.send(_payload())
