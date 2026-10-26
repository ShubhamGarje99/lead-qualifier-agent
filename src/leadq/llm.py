"""Claude calls made directly with the official `anthropic` SDK.

Why not langchain-anthropic: we want exact control of request parameters and
exact `response.usage` accounting per graph node.

Request shape (Claude Sonnet 5.5 / Haiku 5.5):
  * structured outputs via `messages.parse(output_format=<PydanticModel>)`
  * explicit `output_config.effort` ("low" for extraction/judging, "medium" for drafting)
  * no temperature/top_p/top_k, no thinking budget, no assistant prefill
  * Sonnet requests opt into server-side refusal fallback
    (beta `server-side-fallback-2026-07-01`, `fallbacks="default"`); Haiku never does.
  * `stop_reason == "refusal"` is checked before reading content.
"""

from __future__ import annotations

from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from .models import UsageRecord

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept server-side `fallbacks="default"` on the Claude API.
FALLBACK_ELIGIBLE = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}

# USD per million tokens: (input, output). Cache writes bill at 1.25x input, reads at 0.1x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5": (2.00, 10.00),  # default fallback target for Sonnet 5.5 refusals
    "claude-opus-5-5": (4.00, 20.00),
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


class LLMError(Exception):
    """Non-retryable LLM failure (refusal, truncated or unparseable output).

    Carries the usage of the failed call so its cost is still accounted for.
    """

    def __init__(self, message: str, usage: UsageRecord | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class LLMRefusal(LLMError):
    pass


def price_for(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    for known, price in PRICES.items():  # tolerate suffixed ids returned by the API
        if model.startswith(known):
            return price
    raise KeyError(f"no price configured for model {model!r}; add it to leadq.llm.PRICES")


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    inp, out = price_for(model)
    total = (
        input_tokens * inp
        + output_tokens * out
        + cache_write * inp * CACHE_WRITE_MULTIPLIER
        + cache_read * inp * CACHE_READ_MULTIPLIER
    ) / 1_000_000
    return round(total, 8)


def usage_record(node: str, requested_model: str, usage: Any) -> UsageRecord:
    """Exact cost from a response's `usage`.

    With server-side fallbacks, top-level usage covers only the attempt that
    produced the message; `usage.iterations` lists every attempt (each with its
    own model), so when present we price each iteration separately.
    """
    iterations = [
        it for it in (getattr(usage, "iterations", None) or [])
        if getattr(it, "type", "") in {"message", "fallback_message"}
    ]
    entries = iterations or [usage]
    rec = UsageRecord(node=node, model=requested_model)
    total = 0.0
    for it in entries:
        model = getattr(it, "model", None) or requested_model
        i = int(getattr(it, "input_tokens", 0) or 0)
        o = int(getattr(it, "output_tokens", 0) or 0)
        cw = int(getattr(it, "cache_creation_input_tokens", 0) or 0)
        cr = int(getattr(it, "cache_read_input_tokens", 0) or 0)
        rec.input_tokens += i
        rec.output_tokens += o
        rec.cache_creation_input_tokens += cw
        rec.cache_read_input_tokens += cr
        total += cost_usd(model, i, o, cw, cr)
        if getattr(it, "type", "") == "fallback_message":
            rec.fallback_used = True
            rec.model = f"{requested_model} -> {model}"
    rec.cost_usd = round(total, 8)
    return rec


def is_transient(exc: BaseException) -> bool:
    """Retry policy predicate for LLM nodes: network blips, 429 and 5xx only."""
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


class ClaudeClient:
    def __init__(self, api_key: str, *, fallbacks: bool = True, client: Any | None = None) -> None:
        # SDK-level retries kept low: the graph's RetryPolicy owns retries.
        self.client = client or anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=120.0)
        self.fallbacks = fallbacks

    def parse(
        self,
        *,
        node: str,
        model: str,
        system: str,
        user: str,
        output_format: type[T],
        effort: str,
        max_tokens: int = 8000,
    ) -> tuple[T, UsageRecord]:
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
            output_config={"effort": effort},
        )
        if self.fallbacks and model in FALLBACK_ELIGIBLE:
            response = self.client.beta.messages.parse(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.parse(**kwargs)

        record = usage_record(node, model, response.usage)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"{model} declined the request (category={category})", usage=record)
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{model} hit max_tokens={max_tokens} before finishing", usage=record)
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"{model} returned no parseable {output_format.__name__}", usage=record)
        return parsed, record
