"""Offline signal extractor: deterministic keyword/regex heuristics.

Used when no ANTHROPIC_API_KEY is configured (and as the graceful fallback if
an LLM call is refused). Returns the same `Signals` model as the LLM path so
scoring, drafting and evals are mode-agnostic. It is intentionally simple and
transparent; see EVAL_REPORT.md for where it falls short of the LLM path.
"""

from __future__ import annotations

import re
from collections import Counter

from .models import Lead, Signals, SiteProfile
from .rubric import Rubric

# Order matters for tie-breaks: specific verticals before generic SaaS.
INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "fintech": ["payments", "invoicing", "invoices", "lending", "banking", "fintech", "accounts payable",
                "accounts receivable", "expense management", "payroll", "treasury", "reconciliation", "spend management"],
    "healthtech": ["patient", "patients", "clinic", "clinics", "healthcare", "telehealth", "ehr", "clinical",
                   "care teams", "hipaa", "practice management"],
    "logistics_tech": ["freight", "logistics", "shipping lanes", "fleet", "warehouse", "carriers", "shipments",
                       "supply chain", "dispatch", "last-mile"],
    "devtools": ["developers", "developer", "sdk", "ci/cd", "observability", "open source", "open-source",
                 "deployments", "devops", "api-first", "engineering teams"],
    "martech": ["marketing automation", "email marketing", "campaigns", "attribution", "marketers",
                "customer engagement", "audience", "personalization engine"],
    "saas": ["saas", "platform", "software", "dashboard", "workflow", "workflows", "cloud", "integrations",
             "free trial", "start free", "book a demo", "request a demo", "per seat", "subscription"],
    "ecommerce": ["shop now", "add to cart", "free shipping", "online store", "checkout", "collections",
                  "bestsellers", "new arrivals"],
    "marketplace": ["marketplace", "buyers and sellers", "list your", "vendors on"],
    "professional_services": ["accounting", "bookkeeping", "law firm", "legal services", "consulting firm",
                              "tax", "cpa", "advisory", "recruiting firm", "staffing", "audit"],
    "agency": ["agency", "creative studio", "our clients", "case studies", "branding", "studio"],
    "education": ["university", "students", "campus", "curriculum", "courses", "faculty", "school"],
    "nonprofit": ["nonprofit", "non-profit", "donate", "charity", "volunteer"],
    "hospitality": ["restaurant", "bakery", "cafe", "café", "menu", "reservations", "hotel", "catering"],
    "manufacturing": ["manufacturing", "factory", "machining", "industrial", "fabrication"],
}

REGION_KEYWORDS: dict[str, list[str]] = {
    "north_america": ["united states", "usa", "san francisco", "new york", "austin", "seattle", "boston",
                      "chicago", "denver", "toronto", "vancouver", "canada", "california", "texas",
                      "los angeles", "atlanta", "miami", "portland", "montreal"],
    "uk": ["united kingdom", "london", "manchester", "edinburgh", "uk", "bristol", "leeds"],
    "europe": ["berlin", "germany", "paris", "france", "amsterdam", "netherlands", "dublin", "ireland",
               "madrid", "spain", "stockholm", "lisbon", "copenhagen", "warsaw", "munich", "zurich",
               "vienna", "barcelona", "europe", "eu"],
    "india": ["india", "bengaluru", "bangalore", "pune", "mumbai", "hyderabad", "delhi", "chennai",
              "gurugram", "noida"],
    "anz": ["australia", "sydney", "melbourne", "new zealand", "auckland", "brisbane"],
    "latam": ["brazil", "são paulo", "sao paulo", "mexico", "buenos aires", "bogotá", "bogota", "chile",
              "santiago", "lima", "colombia"],
    "apac": ["singapore", "tokyo", "japan", "jakarta", "manila", "seoul", "hong kong", "kuala lumpur"],
    "africa": ["lagos", "nigeria", "nairobi", "kenya", "cape town", "south africa"],
    "middle_east": ["dubai", "uae", "riyadh", "tel aviv", "doha"],
}

B2B_KEYWORDS = ["for teams", "for businesses", "b2b", "enterprise", "book a demo", "request a demo",
                "for sales teams", "for finance teams", "for operations teams", "for ops teams", "customers include",
                "trusted by", "soc 2", "for companies", "for agencies", "for clinics", "for carriers",
                "for brands", "for retailers", "for developers", "talk to sales", "contact sales"]
B2C_KEYWORDS = ["shop now", "add to cart", "free shipping", "for you and your family", "download the app",
                "subscribe and save", "order online", "gift card", "our menu", "book a table", "app store",
                "for everyone", "new arrivals"]

PAIN_KEYWORDS: dict[str, list[str]] = {
    "lead_routing": ["lead routing", "route leads", "routing leads", "routed", "assign leads", "inbound leads",
                     "speed to lead", "speed-to-lead", "qualify leads", "lead qualification", "qualifying leads",
                     "leads sit", "round robin", "round-robin", "triage leads", "demo requests"],
    "crm_hygiene": ["crm is a mess", "crm hygiene", "duplicate", "duplicates", "dirty data", "data quality",
                    "clean up our crm", "crm data", "update the crm", "crm updates", "messy crm", "out of date"],
    "manual_follow_up": ["follow up", "follow-up", "followups", "follow-ups", "chasing", "reminders",
                         "nurture", "sequences"],
    "support_triage": ["support tickets", "ticket triage", "triage tickets", "support queue", "support inbox",
                       "tickets", "helpdesk", "help desk"],
    "reporting": ["reporting", "reports", "dashboards", "weekly report", "pipeline report", "kpis"],
    "data_entry": ["data entry", "copy-paste", "copy paste", "copying", "copy everything", "manual entry",
                   "spreadsheets", "spreadsheet", "re-keying", "rekeying", "by hand"],
}
PAIN_PHRASES = {
    "lead_routing": "inbound lead routing",
    "crm_hygiene": "CRM data quality",
    "manual_follow_up": "manual follow-ups",
    "support_triage": "support-ticket triage",
    "reporting": "reporting",
    "data_entry": "manual data entry",
}

TOOL_NAMES = ["hubspot", "salesforce", "slack", "zapier", "pipedrive", "intercom", "zendesk", "shopify",
              "stripe", "airtable", "notion", "freshdesk", "monday.com", "asana", "jira", "google sheets",
              "excel", "outreach", "apollo", "gmail", "make.com", "n8n"]

DEMO_CUES = ["a demo", "demo call", "demo of", "a call", "quick call", "on a call", "intro call", "meeting", "book a time", "schedule",
             "calendar", "talk to someone", "speak with", "proposal", "quote", "statement of work",
             "get started", "kick off", "kickoff", "start a project", "pilot", "scope this", "scoping"]
TIMELINE_PATTERNS = [r"this (?:week|month|quarter)", r"next (?:week|month|quarter)", r"\bq[1-4]\b",
                     r"by (?:the )?end of", r"\basap\b", r"within (?:\d+|a few|two|three) (?:days|weeks|months)",
                     r"in the next (?:\d+|few|two|three) (?:days|weeks|months)", r"\burgent", r"\bdeadline\b",
                     r"go live", r"live by", r"launch (?:in|by)", r"before (?:our|the) "]
EVALUATING_CUES = ["evaluating", "comparing", "shortlist", "vendors", "pricing", "how much", "what would",
                   "cost", "rfp", "proposal", "estimate"]
URGENT_HIGH = [r"\basap\b", r"\burgent", r"this week", r"this month", r"immediately", r"right away"]
URGENT_MEDIUM = [r"this quarter", r"next month", r"next quarter", r"\bq[1-4]\b", r"within (?:\d+|a few|two|three) (?:weeks|months)",
                 r"by (?:the )?end of", r"live by", r"go live"]
BUDGET_STATED = [r"\bbudget\b[^.!?]{0,60}?\d", r"[$€£₹]\s?\d[^.!?]{0,60}\bbudget\b", r"budget (?:is )?approved",
                 r"approved budget", r"have (?:a |the )?budget", r"budget (?:is )?allocated", r"allocated",
                 r"set aside", r"(?:can|willing to) spend", r"budgeted"]
BUDGET_IMPLIED = [r"\bpricing\b", r"how much", r"\bcost\b", r"\bprice\b", r"\brates\b", r"\bquote\b",
                  r"\bestimate\b", r"\broi\b", r"what would (?:it|this|something like that) (?:cost|run)"]

GENERIC_HEADINGS = {"features", "pricing", "customers", "about", "about us", "careers", "contact", "blog",
                    "product", "resources", "company", "integrations", "faq", "testimonials", "team", "menu"}

_SMALL_CUES = ["solo founder", "one-person", "one person", "just me", "two of us", "the two of us", "three of us",
               "two co-founders", "two cofounders", "i run a small", "tiny team", "side business", "solopreneur"]
_SIZE_PATTERNS = [
    re.compile(r"\b(?:team of|we're|we are|we’re|join(?: our)?|a team of|about|over|nearly)\s+(\d[\d,]*)\+?\s*"
               r"(?:-?\s*person|people|employees|teammates|team members|humans|staff|engineers)\b"),
    re.compile(r"\b(\d[\d,]*)\+?\s*(?:employees|staff members|team members|full-time staff)\b"),
    re.compile(r"\bteam of (\d[\d,]*)\b"),
    re.compile(r"\b(\d[\d,]*)-person\b"),
    re.compile(r"\b(\d[\d,]*)\s+of us\b"),
]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _has(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def _count(text: str, phrases: list[str]) -> int:
    return sum(1 for p in phrases if _has(text, p))


def _any_re(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def band_for(n: int) -> str:
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 200:
        return "51-200"
    if n <= 500:
        return "201-500"
    if n <= 1000:
        return "501-1000"
    return "1001+"


def _employee_band(texts: list[str], evidence: list[str]) -> str:
    for text in texts:  # message first: the lead's own words beat the homepage
        for cue in _SMALL_CUES:
            if cue in text:
                evidence.append(f"size: '{cue}'")
                return "1-10"
        for pat in _SIZE_PATTERNS:
            m = pat.search(text)
            if m:
                n = int(m.group(1).replace(",", ""))
                if 0 < n < 1_000_000:
                    evidence.append(f"size: '{m.group(0).strip()}'")
                    return band_for(n)
    return "unknown"


def _industry(site_text: str, message: str) -> str:
    scores: Counter[str] = Counter()
    for industry, kws in INDUSTRY_KEYWORDS.items():
        scores[industry] = 2 * _count(site_text, kws) + _count(message, kws)
    if not scores or max(scores.values()) == 0:
        return "other"
    order = list(INDUSTRY_KEYWORDS)
    best = max(scores.values())
    verticals = ["fintech", "healthtech", "logistics_tech", "devtools", "martech"]
    # A tech vertical wins over generic "saas" when it has real support.
    for v in verticals:
        if scores[v] >= 4 and scores[v] >= best - 2:
            return v
    return min((i for i, s in scores.items() if s == best), key=order.index)


def _region(texts: list[str]) -> str:
    counts: Counter[str] = Counter()
    for text in texts:
        for region, kws in REGION_KEYWORDS.items():
            counts[region] += _count(text, kws)
    if not counts or max(counts.values()) == 0:
        return "unknown"
    best = max(counts.values())
    order = list(REGION_KEYWORDS)
    return min((r for r, c in counts.items() if c == best), key=order.index)


def _business_model(site_text: str) -> str:
    b2b, b2c = _count(site_text, B2B_KEYWORDS), _count(site_text, B2C_KEYWORDS)
    if b2b >= 2 and b2c >= 2:
        return "both"
    if b2b == 0 and b2c == 0:
        return "unknown"
    return "b2b" if b2b >= b2c else "b2c"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _highlights(site: SiteProfile) -> list[str]:
    out: list[str] = []
    if site.headings:
        out.append(site.headline())
    if site.meta_description:
        out.append(site.meta_description)
    for h in site.headings[1:]:
        if h.lower() not in GENERIC_HEADINGS and len(h.split()) >= 3 and h not in out:
            out.append(h)
            break
    return out[:3]


def extract_signals(lead: Lead, site: SiteProfile, rubric: Rubric) -> Signals:
    message = lead.message.lower()
    site_text = " ".join([site.title, site.meta_description, " ".join(site.headings), site.text]).lower()
    evidence: list[str] = []

    industry = _industry(site_text, message)
    employee_band = _employee_band([message, site_text], evidence)
    region = _region([site_text, message])
    business_model = _business_model(site_text)

    tech = set(site.tech_hints)
    tech.update(t.replace(".com", "") for t in TOOL_NAMES if _has(message, t))
    tech_stack = sorted(tech)

    pains = [p for p, kws in PAIN_KEYWORDS.items() if _count(message, kws)]
    pain_sentences = [s for s in _sentences(lead.message) if any(_count(s.lower(), PAIN_KEYWORDS[p]) for p in pains)]
    pain_summary = (pain_sentences[0] if pain_sentences else (_sentences(lead.message) or [""])[0])[:200]

    # Budget
    budget, budget_note = "none", ""
    for sent in _sentences(lead.message):
        s = sent.lower()
        if _any_re(s, BUDGET_STATED):
            budget, budget_note = "stated", sent[:160]
            break
        if budget == "none" and _any_re(s, BUDGET_IMPLIED):
            budget, budget_note = "implied", sent[:160]

    # Intent
    points = 0
    if _count(message, DEMO_CUES):
        points += 2
    has_timeline = _any_re(message, TIMELINE_PATTERNS)
    if has_timeline:
        points += 2
    points += {"stated": 2, "implied": 1, "none": 0}[budget]
    if _count(message, EVALUATING_CUES):
        points += 1
    points += min(len(pains), 2)
    if not message.strip():
        intent = "none"
    elif points >= 6:
        intent = "high"
    elif points >= 3:
        intent = "medium"
    else:
        intent = "low"

    if _any_re(message, URGENT_HIGH):
        urgency = "high"
    elif _any_re(message, URGENT_MEDIUM):
        urgency = "medium"
    else:
        urgency = "low"

    # Disqualifiers, driven by the rubric's keyword lists
    disqualifiers = []
    for name, rule in rubric.disqualifiers.items():
        scopes = {"message": [message], "homepage": [site_text], "both": [message, site_text]}[rule.scope]
        hit = next((kw for kw in rule.keywords for text in scopes if _has(text, kw.lower())), None)
        if hit:
            disqualifiers.append(name)
            evidence.append(f"{name}: '{hit}'")

    evidence.append(f"intent points={points} ({intent})")
    if budget_note:
        evidence.append(f"budget: '{budget_note[:80]}'")

    return Signals(
        industry=industry,  # type: ignore[arg-type]
        employee_band=employee_band,  # type: ignore[arg-type]
        region=region,  # type: ignore[arg-type]
        business_model=business_model,  # type: ignore[arg-type]
        tech_stack=tech_stack,
        pain_points=pains,  # type: ignore[arg-type]
        pain_summary=pain_summary,
        buying_intent=intent,  # type: ignore[arg-type]
        urgency=urgency,  # type: ignore[arg-type]
        budget=budget,  # type: ignore[arg-type]
        budget_note=budget_note,
        disqualifiers=disqualifiers,  # type: ignore[arg-type]
        homepage_highlights=_highlights(site) if site.found else [],
        evidence=evidence,
        extracted_by="heuristic",
    )
