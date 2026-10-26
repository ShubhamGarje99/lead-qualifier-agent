"""ICP rubric: YAML loading + deterministic, explainable scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .models import CriterionScore, ScoreResult, Signals, Tier

TIER_ORDER = ["A", "B", "C", "disqualified"]


class Seller(BaseModel):
    name: str
    sender_name: str
    offering: str
    calendar_url: str = ""


class Criterion(BaseModel):
    field: str
    kind: Literal["match", "levels", "multi"]
    weight: float = Field(ge=0)
    target: list[str] = Field(default_factory=list)
    partial: list[str] = Field(default_factory=list)
    partial_credit: float = Field(default=0.5, ge=0, le=1)
    levels: dict[str, float] = Field(default_factory=dict)
    per_match: float = Field(default=0.5, gt=0, le=1)
    description: str = ""


class DisqualifierRule(BaseModel):
    description: str
    keywords: list[str] = Field(default_factory=list)
    scope: Literal["message", "homepage", "both"] = "both"


class TierCap(BaseModel):
    field: str
    values: list[str]
    max_tier: Literal["A", "B", "C"]
    reason: str


class Tiers(BaseModel):
    A: float
    B: float
    C: float

    @model_validator(mode="after")
    def _ordered(self) -> "Tiers":
        if not (self.A >= self.B >= self.C >= 0):
            raise ValueError("tier thresholds must satisfy A >= B >= C >= 0")
        return self


class Rubric(BaseModel):
    seller: Seller
    tiers: Tiers
    criteria: dict[str, Criterion]
    disqualifiers: dict[str, DisqualifierRule] = Field(default_factory=dict)
    tier_caps: list[TierCap] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fields_exist(self) -> "Rubric":
        known = set(Signals.model_fields)
        for name, c in self.criteria.items():
            if c.field not in known:
                raise ValueError(f"criterion {name!r} reads unknown signal field {c.field!r}")
        for cap in self.tier_caps:
            if cap.field not in known:
                raise ValueError(f"tier cap reads unknown signal field {cap.field!r}")
        if not self.criteria or sum(c.weight for c in self.criteria.values()) <= 0:
            raise ValueError("rubric needs at least one criterion with positive weight")
        return self

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria.values())


def load_rubric(path: Path | str) -> Rubric:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Rubric.model_validate(data)


# --------------------------------------------------------------------------
def _label(v: str) -> str:
    return v.replace("_", " ")


def _score_criterion(name: str, c: Criterion, signals: Signals) -> CriterionScore:
    raw = getattr(signals, c.field)
    if c.kind == "match":
        value = str(raw)
        if value in c.target:
            credit, why = 1.0, f"{_label(value)} is a target {_label(c.field)}"
        elif value in c.partial:
            credit, why = c.partial_credit, f"{_label(value)} is an adjacent {_label(c.field)} (partial credit)"
        elif value == "unknown":
            credit, why = 0.0, f"{_label(c.field)} unknown (no evidence found)"
        else:
            credit, why = 0.0, f"{_label(value)} is outside the target {_label(c.field)}"
    elif c.kind == "levels":
        value = str(raw)
        credit = float(c.levels.get(value, 0.0))
        why = f"{_label(c.field)} = {value}"
    else:  # multi
        values = [str(v) for v in (raw or [])]
        matched = sorted({v for v in values if v in c.target})
        credit = min(1.0, len(matched) * c.per_match)
        value = ", ".join(values) if values else "none"
        why = f"matches: {', '.join(matched)}" if matched else "no matching items"
    points = round(c.weight * credit, 2)
    return CriterionScore(name=name, weight=c.weight, points=points, value=value, reason=why)


def _tier_for(score: float, tiers: Tiers) -> Tier:
    if score >= tiers.A:
        return "A"
    if score >= tiers.B:
        return "B"
    if score >= tiers.C:
        return "C"
    return "disqualified"


def score_lead(signals: Signals, rubric: Rubric) -> ScoreResult:
    criteria = [_score_criterion(n, c, signals) for n, c in rubric.criteria.items()]
    raw_points = sum(c.points for c in criteria)
    score = round(100.0 * raw_points / rubric.total_weight, 1)

    dq_hits = [d for d in signals.disqualifiers if d in rubric.disqualifiers]
    if dq_hits:
        reasons = [f"{d}: {rubric.disqualifiers[d].description}" for d in dq_hits]
        return ScoreResult(
            score=score,
            tier="disqualified",
            criteria=criteria,
            disqualifiers=reasons,
            summary=f"Disqualified ({', '.join(dq_hits)}) despite fit score {score:.0f}/100.",
        )

    tier = _tier_for(score, rubric.tiers)
    caps: list[str] = []
    for cap in rubric.tier_caps:
        value = getattr(signals, cap.field)
        values = value if isinstance(value, list) else [value]
        if any(str(v) in cap.values for v in values) and tier != "disqualified":
            if TIER_ORDER.index(tier) < TIER_ORDER.index(cap.max_tier):
                caps.append(f"capped {tier} -> {cap.max_tier}: {cap.reason}")
                tier = cap.max_tier

    if tier == "disqualified":
        summary = f"Score {score:.0f}/100 is below the C threshold ({rubric.tiers.C:.0f}): poor fit."
    else:
        top = sorted(criteria, key=lambda c: c.points, reverse=True)[:2]
        summary = f"Tier {tier} ({score:.0f}/100); strongest: " + ", ".join(
            f"{c.name} {c.points:g}/{c.weight:g}" for c in top
        )
        if caps:
            summary += f"; {caps[0]}"
    return ScoreResult(score=score, tier=tier, criteria=criteria, caps=caps, summary=summary)
