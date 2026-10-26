from evals import run as eval_run
from evals.checks import check_email, false_mentions
from leadq.drafting import template_draft
from leadq.enrich import FixtureFetcher, enrich
from leadq.heuristics import extract_signals
from leadq.models import Lead
from leadq.rubric import score_lead
from leadq.settings import Settings

from .conftest import IDEAL_LEAD, REPO


def _draft(raw, rubric):
    lead = Lead.normalize(raw)
    site = enrich(FixtureFetcher(REPO / "fixtures" / "sites"), lead.domain, lead.website)
    signals = extract_signals(lead, site, rubric)
    score = score_lead(signals, rubric)
    return lead, template_draft(lead, site, signals, score, rubric)


def test_template_only_echoes_tools_the_lead_named(rubric):
    # The LedgerLoop homepage loads HubSpot + Segment scripts; this lead never mentions a tool.
    raw = {**IDEAL_LEAD, "message": "Our inbound leads sit for days before anyone follows up. Can we book a call this week?"}
    lead, draft = _draft(raw, rubric)
    assert "you mentioned" not in draft.body.lower()
    assert false_mentions(draft.body, lead.message) == []
    _, named = _draft(IDEAL_LEAD, rubric)  # mentions HubSpot explicitly
    assert "Since you mentioned HubSpot" in named.body


def test_template_is_grounded_and_tier_aware(rubric):
    lead, draft = _draft(IDEAL_LEAD, rubric)
    assert draft.generated_by == "template"
    assert draft.body.startswith("Hi Maya,")
    assert '"Close the books without chasing invoices"' in draft.body
    assert "30-minute call" in draft.body and rubric.seller.calendar_url in draft.body
    checks = check_email(draft.model_dump(), company="LedgerLoop", homepage_facts=["invoices"], homepage_text="",
                         lead_message=lead.message)
    assert checks["passed"], checks
    _, c_draft = _draft({"name": "Sam Okafor", "email": "sam@shopwillow.example", "company": "Willow & Wick",
                         "website": "https://shopwillow.example",
                         "message": "Our support inbox is drowning and we miss follow-ups. Can we do a quick call?"}, rubric)
    assert "30-minute" not in c_draft.body and "write-up" in c_draft.body  # tier C: no call ask


def test_email_checks_catch_bad_drafts():
    bad = {"subject": "Hi [First Name]", "body": "We guarantee 10x pipeline. Since you mentioned Salesforce, great."}
    result = check_email(bad, company="Acme", homepage_facts=["widgets"], homepage_text="", lead_message="We use HubSpot.")
    c = result["checks"]
    assert not c["no_placeholders"] and not c["no_invented_claims"] and not c["grounded_mentions"]
    assert not c["length_ok"] and not c["mentions_company"] and not c["homepage_detail"] and not c["has_cta"]
    assert result["ungrounded_mentions"] == ["salesforce"]
    # a claim is fine when the lead's own homepage makes it
    ok = check_email({"subject": "s", "body": "Your page says award-winning support."}, company="x",
                     homepage_facts=[], homepage_text="Award-winning support team", lead_message="")
    assert ok["checks"]["no_invented_claims"]


def test_tier_metrics_math():
    pairs = [("A", "A"), ("A", "B"), ("B", "B"), ("C", "disqualified"), ("disqualified", "disqualified")]
    m = eval_run.tier_metrics(pairs)
    assert m["accuracy"] == 0.6
    assert m["confusion"]["A"]["B"] == 1 and m["confusion"]["C"]["disqualified"] == 1
    assert m["per_class"]["A"] == {"precision": 1.0, "recall": 0.5, "f1": 0.667, "support": 2}
    assert m["per_class"]["C"]["f1"] == 0.0
    assert m["per_class"]["B"]["f1"] == 0.667 and m["per_class"]["disqualified"]["f1"] == 0.667
    assert m["macro_f1"] == 0.5  # (0.667 * 3 + 0) / 4


def test_eval_llm_without_key_fails_fast(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(eval_run.Settings, "from_env", classmethod(lambda cls, home=None: Settings(home=tmp_path)))
    assert eval_run.main(["--mode", "llm"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "var").exists()
