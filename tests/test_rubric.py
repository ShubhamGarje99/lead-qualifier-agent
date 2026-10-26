import pytest
import yaml
from pydantic import ValidationError

from leadq.models import Signals
from leadq.rubric import load_rubric, score_lead

from .conftest import REPO


def _signals(**overrides) -> Signals:
    base = dict(
        industry="saas", employee_band="51-200", region="north_america", business_model="b2b",
        tech_stack=["hubspot", "slack"], pain_points=["lead_routing", "crm_hygiene"], pain_summary="",
        buying_intent="high", urgency="high", budget="stated", budget_note="", disqualifiers=[],
        homepage_highlights=[], evidence=[],
    )
    base.update(overrides)
    return Signals(**base)


def test_yaml_loads_with_expected_shape(rubric):
    assert rubric.total_weight == 100
    assert (rubric.tiers.A, rubric.tiers.B, rubric.tiers.C) == (75, 50, 30)
    assert set(rubric.disqualifiers) == {"student_project", "competitor", "vendor_pitch", "job_seeker"}
    assert rubric.seller.name == "Acme Automation"


def test_ideal_lead_scores_100_tier_a(rubric):
    result = score_lead(_signals(), rubric)
    assert result.score == 100.0 and result.tier == "A"
    assert all(c.points == c.weight for c in result.criteria)
    assert {c.name for c in result.criteria} == set(rubric.criteria)


def test_partial_credit_and_reasons(rubric):
    result = score_lead(_signals(industry="ecommerce", business_model="both", tech_stack=["hubspot"]), rubric)
    by = {c.name: c for c in result.criteria}
    assert by["industry_fit"].points == 7.5 and "partial" in by["industry_fit"].reason
    assert by["business_model"].points == 5
    assert by["tech_stack"].points == 2.5
    assert result.score == pytest.approx(100 - 7.5 - 5 - 2.5)


def test_vague_ideal_company_is_b(rubric):
    result = score_lead(_signals(buying_intent="low", pain_points=[], budget="none"), rubric)
    # 15 + 15 + 10 + 5 + 25*0.2 + 0 + 0 + 5 = 55
    assert result.score == 55.0 and result.tier == "B"


def test_disqualifier_overrides_score(rubric):
    result = score_lead(_signals(disqualifiers=["competitor"]), rubric)
    assert result.tier == "disqualified"
    assert result.score == 100.0  # fit score kept for transparency
    assert result.disqualifiers and result.disqualifiers[0].startswith("competitor")


def test_tier_cap_for_tiny_company(rubric):
    result = score_lead(_signals(employee_band="1-10"), rubric)
    assert result.score == 85.0
    assert result.tier == "C"
    assert result.caps and "1-10" in result.caps[0]


def test_below_c_threshold_is_disqualified(rubric):
    result = score_lead(
        _signals(industry="hospitality", employee_band="1-10", business_model="b2c", region="unknown",
                 buying_intent="low", pain_points=[], budget="none", tech_stack=[]),
        rubric,
    )
    assert result.score == 5.0 and result.tier == "disqualified" and "below" in result.summary


def test_custom_rubric_normalises_weights(tmp_path):
    data = yaml.safe_load((REPO / "config" / "icp.yaml").read_text())
    data["criteria"] = {"intent": {"field": "buying_intent", "kind": "levels", "weight": 3,
                                   "levels": {"high": 1, "medium": 0.5, "low": 0, "none": 0}}}
    data["tier_caps"] = []
    path = tmp_path / "icp.yaml"
    path.write_text(yaml.safe_dump(data))
    rubric = load_rubric(path)
    assert score_lead(_signals(buying_intent="medium"), rubric).score == 50.0


def test_invalid_rubric_is_rejected(tmp_path):
    data = yaml.safe_load((REPO / "config" / "icp.yaml").read_text())
    data["criteria"]["bogus"] = {"field": "not_a_signal", "kind": "match", "weight": 5}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="unknown signal field"):
        load_rubric(path)
    data = yaml.safe_load((REPO / "config" / "icp.yaml").read_text())
    data["tiers"] = {"A": 40, "B": 60, "C": 30}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="A >= B >= C"):
        load_rubric(path)
