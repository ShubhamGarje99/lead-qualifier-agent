"""Typed data contracts shared by every node, adapter and interface."""

from __future__ import annotations

import hashlib
import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Enumerations. LLM structured outputs are constrained to exactly these values,
# and the offline heuristic extractor emits the same ones.
# --------------------------------------------------------------------------
Industry = Literal[
    "saas",
    "fintech",
    "devtools",
    "martech",
    "logistics_tech",
    "healthtech",
    "ecommerce",
    "professional_services",
    "agency",
    "marketplace",
    "education",
    "nonprofit",
    "hospitality",
    "manufacturing",
    "other",
]
EmployeeBand = Literal["1-10", "11-50", "51-200", "201-500", "501-1000", "1001+", "unknown"]
Region = Literal[
    "north_america", "europe", "uk", "india", "anz", "latam", "apac", "africa", "middle_east", "unknown"
]
BusinessModel = Literal["b2b", "b2c", "both", "unknown"]
Intent = Literal["high", "medium", "low", "none"]
Urgency = Literal["high", "medium", "low"]
BudgetSignal = Literal["stated", "implied", "none"]
PainPoint = Literal[
    "lead_routing", "crm_hygiene", "manual_follow_up", "support_triage", "reporting", "data_entry", "other"
]
Disqualifier = Literal["student_project", "competitor", "vendor_pitch", "job_seeker"]
Tier = Literal["A", "B", "C", "disqualified"]

TIERS: tuple[str, ...] = ("A", "B", "C", "disqualified")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "proton.me", "protonmail.com",
    "aol.com", "gmx.com", "yandex.com", "mail.com",
}


# --------------------------------------------------------------------------
# Inbound lead
# --------------------------------------------------------------------------
class LeadIn(BaseModel):
    """What a form / CSV row / webhook gives us. Validated at the boundary."""

    name: str = Field(min_length=1, max_length=200)
    email: str = Field(max_length=320)
    company: str = Field(default="", max_length=200)
    title: str = Field(default="", max_length=200)
    website: str = Field(default="", max_length=500)
    message: str = Field(default="", max_length=5000)
    source: str = Field(default="web_form", max_length=100)

    @field_validator("name", "company", "title", "website", "message", "source", mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else ("" if v is None else v)

    @field_validator("email", mode="before")
    @classmethod
    def _email(cls, v: object) -> str:
        if not isinstance(v, str) or not _EMAIL_RE.match(v.strip()):
            raise ValueError("invalid email address")
        return v.strip().lower()

    def idempotency_key(self) -> str:
        return f"{self.email.lower()}|{self.company.strip().lower()}"

    def thread_id(self) -> str:
        """Deterministic per lead (email + company) so re-submissions are idempotent."""
        digest = hashlib.sha256(self.idempotency_key().encode()).hexdigest()[:12]
        return f"lead-{digest}"


def domain_of(website: str, email: str) -> str:
    """Company domain from the website, else from a non-free email address."""
    if website:
        raw = website if "://" in website else f"https://{website}"
        host = (urlparse(raw).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            return host
    email_domain = email.rsplit("@", 1)[-1].lower()
    return "" if email_domain in FREE_EMAIL_DOMAINS else email_domain


class Lead(LeadIn):
    """Normalized lead."""

    domain: str = ""
    first_name: str = ""
    free_email: bool = False

    @classmethod
    def normalize(cls, raw: dict) -> "Lead":
        base = LeadIn.model_validate(raw)
        data = base.model_dump()
        domain = domain_of(base.website, base.email)
        email_domain = base.email.rsplit("@", 1)[-1]
        if not data["company"] and domain:
            data["company"] = domain.split(".")[0].replace("-", " ").title()
        if domain and not data["website"]:
            data["website"] = f"https://{domain}"
        data["name"] = " ".join(base.name.split())
        data["message"] = " ".join(base.message.split())
        return cls(
            **data,
            domain=domain,
            first_name=data["name"].split(" ")[0].title(),
            free_email=email_domain in FREE_EMAIL_DOMAINS,
        )


# --------------------------------------------------------------------------
# Enrichment
# --------------------------------------------------------------------------
class SiteProfile(BaseModel):
    url: str = ""
    domain: str = ""
    found: bool = False
    source: Literal["fixture", "live", "none"] = "none"
    title: str = ""
    meta_description: str = ""
    headings: list[str] = Field(default_factory=list)
    text: str = ""
    has_pricing: bool = False
    has_careers: bool = False
    tech_hints: list[str] = Field(default_factory=list)
    error: str | None = None

    def headline(self) -> str:
        for h in self.headings:
            if len(h.split()) >= 2:
                return h
        return self.title


# --------------------------------------------------------------------------
# Signals (the extraction contract)
# --------------------------------------------------------------------------
class SignalsCore(BaseModel):
    """Exactly the schema the LLM fills via structured outputs."""

    industry: Industry = Field(description="Best-fit industry of the lead's company")
    employee_band: EmployeeBand = Field(description="Estimated headcount band; 'unknown' if no evidence")
    region: Region = Field(description="Where the company is based; 'unknown' if no evidence")
    business_model: BusinessModel = Field(description="Who the company sells to")
    tech_stack: list[str] = Field(description="Lowercase tool names seen on the site or in the message, e.g. hubspot, slack")
    pain_points: list[PainPoint] = Field(description="Problems the lead says they want solved")
    pain_summary: str = Field(description="One sentence, in the lead's terms, of what they want solved")
    buying_intent: Intent = Field(description="high = explicit ask + timeline/budget; medium = concrete ask; low = vague interest")
    urgency: Urgency
    budget: BudgetSignal = Field(description="stated = an amount or approved budget; implied = asks about cost; none")
    budget_note: str = Field(description="Verbatim budget/timeline hint, or empty string")
    disqualifiers: list[Disqualifier] = Field(description="Only flags clearly supported by evidence")
    homepage_highlights: list[str] = Field(description="2-4 specific, verifiable facts from the homepage text")
    evidence: list[str] = Field(description="Short verbatim quotes supporting the classification")


class Signals(SignalsCore):
    extracted_by: Literal["heuristic", "llm"] = "heuristic"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
class CriterionScore(BaseModel):
    name: str
    weight: float
    points: float
    value: str
    reason: str


class ScoreResult(BaseModel):
    score: float
    tier: Tier
    criteria: list[CriterionScore]
    disqualifiers: list[str] = Field(default_factory=list)
    caps: list[str] = Field(default_factory=list)
    summary: str = ""


# --------------------------------------------------------------------------
# Drafting & review
# --------------------------------------------------------------------------
class EmailDraft(BaseModel):
    subject: str = Field(description="Plain-text subject line, under 80 characters")
    body: str = Field(description="Plain-text email body with greeting and sign-off")


class Draft(EmailDraft):
    generated_by: Literal["template", "llm"] = "template"
    model: str | None = None


class ReviewDecision(BaseModel):
    action: Literal["approve", "edit", "reject"]
    subject: str | None = None
    body: str | None = None
    reason: str | None = None
    reviewer: str = "reviewer"

    @model_validator(mode="after")
    def _check(self) -> "ReviewDecision":
        if self.action == "edit" and not ((self.subject or "").strip() and (self.body or "").strip()):
            raise ValueError("edit requires a non-empty subject and body")
        return self


# --------------------------------------------------------------------------
# Cost accounting
# --------------------------------------------------------------------------
class UsageRecord(BaseModel):
    node: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    fallback_used: bool = False
