from leadq.enrich import FixtureFetcher, enrich, parse_html
from leadq.heuristics import band_for, extract_signals
from leadq.models import Lead, SiteProfile

from .conftest import IDEAL_LEAD, REPO, STUDENT_LEAD


def _site(domain: str) -> SiteProfile:
    return enrich(FixtureFetcher(REPO / "fixtures" / "sites"), domain, f"https://{domain}")


def test_ideal_lead_signals(rubric):
    lead = Lead.normalize(IDEAL_LEAD)
    sig = extract_signals(lead, _site(lead.domain), rubric)
    assert sig.extracted_by == "heuristic"
    assert sig.industry == "fintech"
    assert sig.employee_band == "51-200"  # "We're 140 people" on the careers section
    assert sig.region == "north_america"
    assert sig.business_model == "b2b"
    assert sig.buying_intent == "high"
    assert sig.budget == "stated" and "$15-20k" in sig.budget_note
    assert {"lead_routing", "data_entry"} <= set(sig.pain_points)
    assert {"hubspot", "slack"} <= set(sig.tech_stack)
    assert sig.disqualifiers == []
    assert sig.homepage_highlights[0] == "Close the books without chasing invoices"


def test_student_without_site_is_flagged(rubric):
    lead = Lead.normalize(STUDENT_LEAD)
    assert lead.domain == "" and lead.free_email
    site = enrich(FixtureFetcher(REPO / "fixtures" / "sites"), lead.domain, lead.website)
    assert not site.found
    sig = extract_signals(lead, site, rubric)
    assert "student_project" in sig.disqualifiers
    assert sig.homepage_highlights == []


def test_intent_levels(rubric):
    base = {"name": "A B", "email": "a@x.example", "company": "X"}
    vague = extract_signals(Lead.normalize({**base, "message": "Can you tell me more about what you do?"}), SiteProfile(), rubric)
    medium = extract_signals(
        Lead.normalize({**base, "message": "We want to automate support ticket triage in Zendesk. What would that cost?"}),
        SiteProfile(), rubric,
    )
    empty = extract_signals(Lead.normalize({**base, "message": ""}), SiteProfile(), rubric)
    assert vague.buying_intent == "low"
    assert medium.buying_intent == "medium" and medium.budget == "implied"
    assert "support_triage" in medium.pain_points and "zendesk" in medium.tech_stack
    assert empty.buying_intent == "none"


def test_size_and_small_cues(rubric):
    html = "<html><head><title>T</title></head><body><h1>Tiny shop</h1><p>It's just the two of us in Leeds.</p></body></html>"
    site = parse_html(html, url="https://t.example", domain="t.example", source="fixture")
    sig = extract_signals(Lead.normalize({"name": "A", "email": "a@t.example", "message": "hello"}), site, rubric)
    assert sig.employee_band == "1-10"
    assert sig.region == "uk"
    assert [band_for(n) for n in (1, 10, 11, 50, 51, 200, 201, 500, 501, 1000, 1001)] == [
        "1-10", "1-10", "11-50", "11-50", "51-200", "51-200", "201-500", "201-500", "501-1000", "501-1000", "1001+",
    ]


def test_vendor_pitch_and_competitor(rubric):
    vendor = Lead.normalize({"name": "C D", "email": "c@growthrank.example", "website": "growthrank.example",
                             "message": "Our agency can help you rank on page one. Our services include link building."})
    sig = extract_signals(vendor, _site("growthrank.example"), rubric)
    assert "vendor_pitch" in sig.disqualifiers
    comp = Lead.normalize({"name": "L F", "email": "l@pixelforge.example", "website": "pixelforge.example",
                           "message": "Open to a partnership call?"})
    assert "competitor" in extract_signals(comp, _site("pixelforge.example"), rubric).disqualifiers


def test_parse_html_extracts_structure():
    site = _site("stackpilot.example")
    assert site.found and site.source == "fixture"
    assert site.title.startswith("StackPilot")
    assert "internal developer platform" in site.meta_description
    assert site.headings[0] == "Self-service infrastructure for every engineering team"
    assert site.has_pricing and site.has_careers
    assert "hubspot" in site.tech_hints and "segment" in site.tech_hints
    assert "<script" not in site.text
