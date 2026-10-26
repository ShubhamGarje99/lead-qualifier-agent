"""Deterministic email-quality checks (no LLM needed)."""

from __future__ import annotations

import re
from typing import Any

from leadq.enrich import TECH_PATTERNS
from leadq.heuristics import TOOL_NAMES

MIN_WORDS, MAX_WORDS = 60, 220

CTA_PATTERNS = [r"\bcall\b", r"\bbook\b", r"\bschedule\b", r"grab a slot", r"pick (?:any|a) time", r"\breply\b",
                r"send (?:it|over|you)", r"https?://\S*cal\S*", r"\bmeet\b", r"\bchat\b"]
PLACEHOLDER_PATTERNS = [r"\[[A-Za-z][A-Za-z _]*\]", r"\{\{?\s*\w+\s*\}?\}", r"<[A-Z_]{3,}>", r"\bTODO\b", r"\bXXX\b",
                        r"lorem ipsum", r"\bINSERT\b"]
# Claims a first-reply email should never make unless the lead's own material says so.
CLAIM_BLOCKLIST = ["guarantee", "guaranteed", "100%", "#1", "number one", "award-winning", "fortune 500",
                   "10x", "double your", "triple your", "soc 2 certified", "iso certified", "hundreds of clients",
                   "thousands of customers", "industry-leading", "best-in-class", "proven to", "case study",
                   "we've helped", "we have helped", "our clients include"]

# "you mentioned X" must be true: X has to appear in the lead's own message.
MENTION_RE = re.compile(r"you(?:'ve| have)? (?:mentioned|said|told us|are using|use)\b[^.!?\n]*", re.IGNORECASE)
TOOL_VOCAB = sorted({t.replace("_", " ") for t in [*TOOL_NAMES, *TECH_PATTERNS]})

CHECKS = ["mentions_company", "homepage_detail", "has_cta", "length_ok", "no_placeholders", "no_invented_claims",
          "grounded_mentions"]


def false_mentions(text: str, lead_message: str) -> list[str]:
    msg = lead_message.lower()
    bad = []
    for m in MENTION_RE.finditer(text):
        span = m.group(0).lower()
        bad += [t for t in TOOL_VOCAB if re.search(rf"\b{re.escape(t)}\b", span) and t not in msg]
    return sorted(set(bad))


def check_email(draft: dict[str, Any], *, company: str, homepage_facts: list[str], homepage_text: str,
                lead_message: str) -> dict[str, Any]:
    subject, body = draft.get("subject", ""), draft.get("body", "")
    text = f"{subject}\n{body}"
    low = text.lower()
    words = len(body.split())
    grounded_source = f"{homepage_text}\n{lead_message}".lower()

    company_tokens = [company.lower()] + ([company.split()[0].lower()] if company and len(company.split()[0]) > 3 else [])
    hits = [f for f in homepage_facts if f.lower() in low]
    invented = [c for c in CLAIM_BLOCKLIST if c in low and c not in grounded_source]
    placeholders = [m.group(0) for p in PLACEHOLDER_PATTERNS for m in re.finditer(p, text, flags=re.IGNORECASE)]
    ungrounded = false_mentions(text, lead_message)

    results = {
        "mentions_company": bool(company) and any(t and t in low for t in company_tokens),
        "homepage_detail": bool(hits) if homepage_facts else True,
        "has_cta": any(re.search(p, low) for p in CTA_PATTERNS),
        "length_ok": MIN_WORDS <= words <= MAX_WORDS,
        "no_placeholders": not placeholders,
        "no_invented_claims": not invented,
        "grounded_mentions": not ungrounded,
    }
    return {
        "checks": results,
        "passed": all(results.values()),
        "word_count": words,
        "homepage_facts_hit": hits,
        "invented_claims": invented,
        "placeholders": placeholders,
        "ungrounded_mentions": ungrounded,
    }
