from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leadq.models import EmailDraft, SignalsCore
from leadq.rubric import load_rubric
from leadq.service import LeadQualifier
from leadq.settings import Settings

REPO = Path(__file__).resolve().parents[1]

IDEAL_LEAD = {
    "name": "Maya Chen",
    "email": "maya.chen@ledgerloop.example",
    "company": "LedgerLoop",
    "title": "VP Revenue Operations",
    "website": "https://ledgerloop.example",
    "message": (
        "We get about 300 inbound demo requests a month and they sit in a shared inbox until someone routes them. "
        "Reps copy everything into HubSpot by hand. Budget is approved ($15-20k) and we want this live this quarter. "
        "Could we book a call next week?"
    ),
    "source": "website_form",
}
STUDENT_LEAD = {
    "name": "Arjun Mehta",
    "email": "arjun.mehta2004@gmail.com",
    "message": "Hi! I'm a student working on my final year project on lead scoring. Could you share the code?",
}


@pytest.fixture
def rubric():
    return load_rubric(REPO / "config" / "icp.yaml")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path,
        icp_path=REPO / "config" / "icp.yaml",
        fixtures_dir=REPO / "fixtures" / "sites",
        requested_mode="offline",
        retry_initial_interval=0.01,
    )


@pytest.fixture
def qualifier(settings: Settings):
    q = LeadQualifier(settings)
    yield q
    q.close()


# --------------------------------------------------------------------------
# A fake Anthropic client: records every request, returns canned parsed output.
# --------------------------------------------------------------------------
class _Endpoint:
    def __init__(self, owner: "FakeAnthropic", beta: bool) -> None:
        self.owner, self.beta = owner, beta

    def parse(self, **kwargs: Any):
        self.owner.calls.append({"beta": self.beta, **kwargs})
        fmt = kwargs["output_format"]
        if self.owner.refuse:
            parsed = None
            stop = "refusal"
        else:
            parsed = self.owner.outputs[fmt.__name__]
            stop = "end_turn"
        usage = self.owner.usage_factory(kwargs["model"])
        return SimpleNamespace(
            stop_reason=stop,
            stop_details=SimpleNamespace(category="cyber") if stop == "refusal" else None,
            usage=usage,
            parsed_output=parsed,
        )


class FakeAnthropic:
    def __init__(self, refuse: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.refuse = refuse
        self.messages = _Endpoint(self, beta=False)
        self.beta = SimpleNamespace(messages=_Endpoint(self, beta=True))
        self.outputs = {
            "SignalsCore": SignalsCore(
                industry="fintech", employee_band="51-200", region="north_america", business_model="b2b",
                tech_stack=["HubSpot", "slack"], pain_points=["lead_routing", "data_entry"],
                pain_summary="Inbound demo requests sit in a shared inbox.", buying_intent="high", urgency="medium",
                budget="stated", budget_note="Budget is approved ($15-20k)", disqualifiers=[],
                homepage_highlights=["Close the books without chasing invoices"], evidence=["300 inbound demo requests"],
            ),
            "EmailDraft": EmailDraft(subject="Routing LedgerLoop's demo requests", body="Hi Maya,\n\nDraft body.\n\nBest,\nJordan\nAcme Automation"),
        }

    @staticmethod
    def usage_factory(model: str):
        return SimpleNamespace(
            input_tokens=1200, output_tokens=300, cache_creation_input_tokens=0, cache_read_input_tokens=0, iterations=None
        )
