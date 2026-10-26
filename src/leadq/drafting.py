"""Offline email drafting: a deterministic, tier-aware template filler.

Grounding rule (same as the LLM prompt): only reference facts that appear in
the lead's message or on their homepage. No invented metrics or claims.
"""

from __future__ import annotations

from .heuristics import PAIN_PHRASES
from .models import Draft, Lead, ScoreResult, Signals, SiteProfile
from .rubric import Rubric

INTEGRATIONS = {"hubspot": "HubSpot", "salesforce": "Salesforce", "slack": "Slack", "zapier": "Zapier",
                "pipedrive": "Pipedrive", "intercom": "Intercom", "zendesk": "Zendesk"}

CTA_BY_TIER = {
    "A": "Would you be open to a 30-minute call this week to map out a first version? You can grab a slot here: {calendar_url}",
    "B": "Would a 20-minute intro call next week be useful? If so, pick any time that suits you: {calendar_url}",
    "C": "If it would help, I can send over a short write-up of how we would approach this. Just reply and I will send it across.",
}


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _homepage_line(lead: Lead, site: SiteProfile, signals: Signals) -> str:
    company = lead.company or "your team"
    if site.found and site.headline():
        headline = site.headline().rstrip(".")
        detail = next(
            (h for h in signals.homepage_highlights if h != site.headline() and h != site.meta_description),
            "",
        )
        line = f'I had a look at {_possessive(company)} site: "{headline}" is a clear pitch'
        if detail:
            line += f', and "{detail.rstrip(".")}" stood out'
        return line + "."
    return f"I tried to find more about {company} online but could not load your homepage, so forgive any gaps."


def template_draft(lead: Lead, site: SiteProfile, signals: Signals, score: ScoreResult, rubric: Rubric) -> Draft:
    seller = rubric.seller
    pains = [PAIN_PHRASES[p] for p in signals.pain_points if p in PAIN_PHRASES]
    topic = _join(pains[:2]) if pains else "automating parts of your workflow"
    # Only echo tools the lead actually named; site-detected tech (tracking scripts) is not "mentioned".
    message = lead.message.lower()
    tools = [t for t in signals.tech_stack if t in INTEGRATIONS and t in message]

    para1 = f"Thanks for reaching out about {topic}. " + _homepage_line(lead, site, signals)
    if pains:
        para2 = (
            f"{seller.name} builds small, tested automations for exactly this kind of work: "
            f"an agent handles the repetitive part of {topic}, and a person on your team approves anything "
            "before it goes out."
        )
    else:
        para2 = (
            f"{seller.name} builds small, tested automations for B2B revenue and operations teams, "
            "with a human approval step so nothing goes out without your sign-off."
        )
    if tools:
        para2 += f" Since you mentioned {_join([INTEGRATIONS[t] for t in tools])}, we would plug into what you already use."
    if signals.budget == "stated" and score.tier == "A":
        shared = "budget and timeline" if signals.urgency != "low" else "budget"
        para2 += f" Thanks also for sharing your {shared}; that makes it much easier to scope."

    tier = score.tier if score.tier in CTA_BY_TIER else "C"
    cta = CTA_BY_TIER[tier].format(calendar_url=seller.calendar_url)
    body = (
        f"Hi {lead.first_name or 'there'},\n\n{para1}\n\n{para2}\n\n{cta}\n\n"
        f"Best,\n{seller.sender_name}\n{seller.name}"
    )
    subject_topic = pains[0] if pains else "your inquiry"
    subject = f"Re: {subject_topic} at {lead.company}" if lead.company else f"Re: {subject_topic}"
    return Draft(subject=subject[:78], body=body, generated_by="template")
