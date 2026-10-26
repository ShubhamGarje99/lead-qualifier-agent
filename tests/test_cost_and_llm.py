from types import SimpleNamespace

import pytest

from leadq.llm import FALLBACK_BETA, ClaudeClient, LLMRefusal, cost_usd, usage_record
from leadq.models import SignalsCore
from leadq.service import LeadQualifier
from leadq.settings import MissingAPIKeyError, Settings

from .conftest import IDEAL_LEAD, FakeAnthropic


def _usage(i, o, cw=0, cr=0, iterations=None):
    return SimpleNamespace(input_tokens=i, output_tokens=o, cache_creation_input_tokens=cw,
                           cache_read_input_tokens=cr, iterations=iterations)


def test_cost_math_sonnet_and_haiku():
    # Sonnet 5.5: $2 in / $10 out per MTok
    assert cost_usd("claude-sonnet-5-5", 1_000, 500) == pytest.approx(0.002 + 0.005)
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    # Haiku 5.5: $0.10 in / $0.50 out per MTok
    assert cost_usd("claude-haiku-5-5", 2_000, 400) == pytest.approx(0.0002 + 0.0002)
    # cache write 1.25x, cache read 0.1x of input price
    assert cost_usd("claude-sonnet-5-5", 0, 0, cache_write=1_000_000, cache_read=1_000_000) == pytest.approx(2.5 + 0.2)
    with pytest.raises(KeyError):
        cost_usd("some-unknown-model", 1, 1)


def test_usage_record_from_mocked_response():
    rec = usage_record("extract_signals", "claude-sonnet-5-5", _usage(1_234, 567, cw=100, cr=50))
    assert (rec.input_tokens, rec.output_tokens) == (1_234, 567)
    expected = (1_234 * 2 + 567 * 10 + 100 * 2 * 1.25 + 50 * 2 * 0.1) / 1e6
    assert rec.cost_usd == pytest.approx(expected)
    assert not rec.fallback_used


def test_usage_record_prices_each_fallback_iteration():
    iterations = [
        SimpleNamespace(type="message", model=None, input_tokens=1_000, output_tokens=0,
                        cache_creation_input_tokens=0, cache_read_input_tokens=0),
        SimpleNamespace(type="fallback_message", model="claude-sonnet-5", input_tokens=1_000, output_tokens=200,
                        cache_creation_input_tokens=0, cache_read_input_tokens=0),
    ]
    rec = usage_record("draft_email", "claude-sonnet-5-5", _usage(1_000, 200, iterations=iterations))
    assert rec.fallback_used and rec.model == "claude-sonnet-5-5 -> claude-sonnet-5"
    assert (rec.input_tokens, rec.output_tokens) == (2_000, 200)
    assert rec.cost_usd == pytest.approx((2_000 * 2 + 200 * 10) / 1e6)


def test_sonnet_requests_use_fallback_beta_and_no_sampling_params():
    fake = FakeAnthropic()
    client = ClaudeClient("test-key", client=fake)
    parsed, rec = client.parse(node="n", model="claude-sonnet-5-5", system="s", user="u",
                               output_format=SignalsCore, effort="low")
    call = fake.calls[-1]
    assert call["beta"] is True and call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "low"} and call["output_format"] is SignalsCore
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in call
    assert rec.cost_usd == pytest.approx((1_200 * 2 + 300 * 10) / 1e6)


def test_haiku_requests_never_send_fallbacks():
    fake = FakeAnthropic()
    ClaudeClient("k", client=fake).parse(node="judge", model="claude-haiku-5-5", system="s", user="u",
                                         output_format=SignalsCore, effort="low")
    assert fake.calls[-1]["beta"] is False and "fallbacks" not in fake.calls[-1] and "betas" not in fake.calls[-1]


def test_fallbacks_toggle_off():
    fake = FakeAnthropic()
    ClaudeClient("k", fallbacks=False, client=fake).parse(node="n", model="claude-sonnet-5-5", system="s", user="u",
                                                          output_format=SignalsCore, effort="low")
    assert fake.calls[-1]["beta"] is False and "fallbacks" not in fake.calls[-1]


def test_refusal_is_checked_and_cost_kept():
    fake = FakeAnthropic(refuse=True)
    with pytest.raises(LLMRefusal) as exc:
        ClaudeClient("k", client=fake).parse(node="n", model="claude-sonnet-5-5", system="s", user="u",
                                             output_format=SignalsCore, effort="low")
    assert exc.value.usage is not None and exc.value.usage.cost_usd > 0


def test_llm_mode_accumulates_cost_per_node(settings):
    fake = FakeAnthropic()
    s = settings.with_overrides(anthropic_api_key="test-key", requested_mode="llm")
    with LeadQualifier(s, llm=ClaudeClient("test-key", client=fake)) as q:
        r = q.submit(IDEAL_LEAD)
        card = q.card(r["thread_id"])
    assert [c["output_format"].__name__ for c in fake.calls] == ["SignalsCore", "EmailDraft"]
    assert [c["output_config"]["effort"] for c in fake.calls] == ["low", "medium"]
    assert card["signals"]["extracted_by"] == "llm" and card["signals"]["tech_stack"] == ["hubspot", "slack"]
    assert card["draft"]["generated_by"] == "llm"
    assert [u["node"] for u in card["usage"]] == ["extract_signals", "draft_email"]
    per_call = (1_200 * 2 + 300 * 10) / 1e6
    assert r["input_tokens"] == 2_400 and r["output_tokens"] == 600
    assert r["cost_usd"] == pytest.approx(2 * per_call)
    assert q.store is not None


def test_refusal_degrades_to_offline_paths(settings):
    fake = FakeAnthropic(refuse=True)
    s = settings.with_overrides(anthropic_api_key="test-key", requested_mode="llm")
    with LeadQualifier(s, llm=ClaudeClient("test-key", client=fake)) as q:
        r = q.submit(IDEAL_LEAD)
        card = q.card(r["thread_id"])
    assert r["status"] == "awaiting_review"
    assert card["signals"]["extracted_by"] == "heuristic" and card["draft"]["generated_by"] == "template"
    assert r["cost_usd"] > 0  # refused calls are still accounted for
    assert any("fallback" in e["message"] for e in card["events"])


def test_llm_mode_without_key_fails_fast(tmp_path):
    s = Settings(home=tmp_path, anthropic_api_key=None)
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        s.resolve_mode("llm")
    assert s.resolve_mode("auto") == "offline"
    assert Settings(home=tmp_path, anthropic_api_key="k").resolve_mode("auto") == "llm"
