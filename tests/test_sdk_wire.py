"""Exercise the real `anthropic` SDK end to end against an in-process mock transport (no network).

Verifies the exact request body/headers we send and that `messages.parse` returns our Pydantic models.
"""

import json

import anthropic
import httpx2
import pytest

from evals.judge import CRITERIA, EmailJudgement
from leadq.llm import FALLBACK_BETA, ClaudeClient, LLMRefusal
from leadq.models import EmailDraft, SignalsCore

SIGNALS = {
    "industry": "saas", "employee_band": "51-200", "region": "europe", "business_model": "b2b",
    "tech_stack": ["hubspot"], "pain_points": ["lead_routing"], "pain_summary": "x", "buying_intent": "high",
    "urgency": "low", "budget": "none", "budget_note": "", "disqualifiers": [], "homepage_highlights": ["h"],
    "evidence": ["e"],
}


@pytest.fixture
def wire():
    seen: list[tuple[dict, dict]] = []
    state = {"stop_reason": "end_turn"}

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append((dict(request.headers), body))
        props = body["output_config"]["format"]["schema"]["properties"]
        if "personalization" in props:
            out = {c: {"rationale": "fine", "score": 4} for c in CRITERIA}
        elif "subject" in props:
            out = {"subject": "Hi", "body": "Body"}
        else:
            out = SIGNALS
        content = [] if state["stop_reason"] == "refusal" else [{"type": "text", "text": json.dumps(out)}]
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"], "content": content,
            "stop_reason": state["stop_reason"], "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        })

    client = anthropic.Anthropic(
        api_key="test-key", base_url="https://api.anthropic.com", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeClient("test-key", client=client), seen, state


def test_sonnet_extraction_request_shape(wire):
    cc, seen, _ = wire
    parsed, rec = cc.parse(node="extract_signals", model="claude-sonnet-5-5", system="s", user="u",
                           output_format=SignalsCore, effort="low")
    assert isinstance(parsed, SignalsCore) and parsed.industry == "saas"
    headers, body = seen[-1]
    assert headers["anthropic-beta"] == FALLBACK_BETA and body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low" and body["output_config"]["format"]["type"] == "json_schema"
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(body)
    assert body["messages"][-1]["role"] == "user"  # no assistant prefill
    assert rec.cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1e6)


def test_haiku_judge_request_has_no_fallbacks(wire):
    cc, seen, _ = wire
    parsed, rec = cc.parse(node="judge", model="claude-haiku-5-5", system="s", user="u",
                           output_format=EmailJudgement, effort="low")
    assert isinstance(parsed, EmailJudgement) and parsed.tone.score == 4
    headers, body = seen[-1]
    assert "anthropic-beta" not in headers and "fallbacks" not in body
    assert rec.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1e6)


def test_drafting_effort_medium(wire):
    cc, seen, _ = wire
    parsed, _ = cc.parse(node="draft_email", model="claude-sonnet-5-5", system="s", user="u",
                         output_format=EmailDraft, effort="medium")
    assert parsed.subject == "Hi" and seen[-1][1]["output_config"]["effort"] == "medium"


def test_refusal_from_wire_raises_before_reading_content(wire):
    cc, _, state = wire
    state["stop_reason"] = "refusal"
    with pytest.raises(LLMRefusal):
        cc.parse(node="extract_signals", model="claude-sonnet-5-5", system="s", user="u",
                 output_format=SignalsCore, effort="low")
