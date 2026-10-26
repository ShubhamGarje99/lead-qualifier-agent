"""Render EVAL_REPORT.md from evals/results/*.json (only numbers that were actually produced)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from leadq.models import TIERS

from .checks import CHECKS, CLAIM_BLOCKLIST, MAX_WORDS, MIN_WORDS
from .judge import CRITERIA

PENDING = "pending — run `make eval-llm` with ANTHROPIC_API_KEY set"
LABEL_COMMIT = "0225e23"  # commit that added evals/leads.jsonl, before the scorer ever ran on it
MODE_TITLES = {
    "offline": "Offline mode (heuristic extraction + template drafts)",
    "llm": "LLM mode (Claude structured-output extraction + Claude drafts)",
}


# Hand-written analysis of the offline run, rendered into EVAL_REPORT.md. Written after reviewing the first
# offline run; update it whenever the offline numbers change.
FINDINGS_DESCRIBE = (0.55, 0.542, 0.8)  # (accuracy, macro-F1, email all-checks pass rate) of the reviewed run
OFFLINE_FINDINGS = """## Findings from the offline run

*Hand-written after reviewing the first offline run on this set. Labels were not changed; the extractor was not
re-tuned on these leads, so the offline numbers are a held-out baseline.*

**Tier accuracy is 0.55 (11/20), macro-F1 0.54.** The rubric and scorer behave as designed. All nine errors come
from the keyword extractor misreading phrasing it was never tuned on (it was built on the 8 leads in
`data/leads.sample.csv`):

1. **Headcount phrasing (5 of 9 misses).** "70 accountants and advisers" (E09), "25 designers, writers and
   strategists" (E13), "more than 5,000 staff" (E15) and "built by three friends" (E12) were not recognised, so size
   came out `unknown` (0 points, and in E12 the 1-10 size cap never applied). In the other direction, "two of us spend
   the first two hours..." (E02) describes workload, but the extractor read it as a 1-10 company and capped an A lead
   to C.
2. **Disqualifier recall.** The keyword lists caught the student/thesis (E16) and job-seeker (E19) leads, and did
   *not* misfire on the university office that mentions "prospective students" (E15). They missed a competitor that
   calls itself a "boutique AI consultancy" (E17 -> B) and an offshore dev shop pitching "dedicated React and Python
   engineers at $25/hour" (E18 -> C). Both would reach a human reviewer instead of being archived automatically.
3. **Intent and pain wording.** "Let's talk next week" (E03), "Could you share some examples" (E11), "re-key" and
   "routes and follows up" (E03, E15) were not matched, which lowered intent or pain-fit scores.
4. **Business model** came out `unknown` for most sites, because their B2B cues ("Get a quote", "Start free") were
   not in the cue list. That cost up to 10 points on E02, E03, E09, E11 and E17.

These are meaning-level distinctions (headcount vs. workload, consultancy vs. customer, a pitch vs. a request),
which is where the LLM extraction path is expected to help. That is still a hypothesis until `make eval-llm` has
been run.

**Email quality.** Reading the first-run drafts showed a grounding bug in the offline template: it wrote "Since you
mentioned HubSpot/Intercom" when the tool had only been detected from a tracking script on the lead's homepage
(E04, E05, E06, E07, E10). The original six checks did not catch this. A seventh check (`grounded_mentions`: any
"you mentioned X" must appear in the lead's own message) was added and the template now only echoes tools the lead
named. With all seven checks, the first-run drafts would have passed 9/15; after the fix they pass 12/15. Tier
predictions were identical before and after the fix. The 3 remaining failures (E06, E10, E18) quote a homepage
headline that is not in the hand-listed facts for that lead. That is a strict check, but a fair one: quoting the
tagline is weaker personalisation than naming what the company actually does."""


def _load(results_dir: Path, mode: str) -> dict[str, Any] | None:
    path = results_dir / f"{mode}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _summary_row(name: str, extraction: str, drafting: str, r: dict[str, Any] | None) -> str:
    if r is None:
        return f"| {name} | {extraction} | {drafting} | {PENDING} | pending | pending | pending | pending |"
    m, e = r["tier_metrics"], r["email"]
    correct = sum(1 for row in r["rows"] if row["correct"])
    judge = f"{r['judge']['mean']['overall']:.2f} (n={r['judge']['n']})" if r["judge"]["mean"] else (
        "not run (offline; needs key)" if r["mode"] == "offline" else "n/a")
    checks = f"{sum(1 for row in r['rows'] if row['email_checks'] and row['email_checks']['passed'])}/{e['n_drafts']}"
    return (f"| {name} | {extraction} | {drafting} | {m['accuracy']:.2f} ({correct}/{m['n']}) | {m['macro_f1']:.2f} | "
            f"{checks} ({_pct(e['all_checks_pass_rate'])}) | {judge} | ${r['cost']['pipeline_per_lead_usd']:.4f} |")


def _confusion(m: dict[str, Any]) -> list[str]:
    head = "| expected \\ predicted | " + " | ".join(TIERS) + " | recall |"
    lines = [head, "|" + "---|" * (len(TIERS) + 2)]
    for e in TIERS:
        cells = []
        for p in TIERS:
            n = m["confusion"][e][p]
            cells.append(f"**{n}**" if e == p and n else str(n))
        lines.append(f"| **{e}** | " + " | ".join(cells) + f" | {m['per_class'][e]['recall']:.2f} |")
    lines.append("| precision | " + " | ".join(f"{m['per_class'][p]['precision']:.2f}" for p in TIERS) + " | |")
    return lines


def _mode_section(mode: str, r: dict[str, Any]) -> list[str]:
    m = r["tier_metrics"]
    out = [f"## {MODE_TITLES[mode]}", ""]
    meta = f"Run {r['run_at']} at commit `{r['git_commit']}`, {r['n_leads']} leads, wall time {r['wall_time_s']}s."
    if r.get("model"):
        meta += f" Model: `{r['model']}`."
    if r.get("judge_model"):
        meta += f" Judge: `{r['judge_model']}`."
    out += [meta, "", f"**Accuracy {m['accuracy']:.2f} · macro-F1 {m['macro_f1']:.2f}**", "",
            "### Confusion matrix (rows = expected tier, columns = predicted tier)", "", *_confusion(m), "",
            "| tier | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for t in TIERS:
        pc = m["per_class"][t]
        out.append(f"| {t} | {pc['precision']:.2f} | {pc['recall']:.2f} | {pc['f1']:.2f} | {pc['support']} |")
    out.append("")

    misses = [row for row in r["rows"] if not row["correct"]]
    out += ["### Misclassifications", ""]
    if not misses:
        out += ["None.", ""]
    for row in misses:
        s = row["signals"]
        extra = "; ".join(row["caps"] + row["disqualifiers"])
        out += [
            f"- **{row['id']} {row['company']}**: expected **{row['expected']}**, predicted **{row['predicted']}** "
            f"(score {row['score']:.1f}).",
            f"  - Label rationale: {row['why_expected']}",
            f"  - Extracted ({s['extracted_by']}): industry={s['industry']}, size={s['employee_band']}, "
            f"region={s['region']}, model={s['business_model']}, intent={s['buying_intent']}, budget={s['budget']}, "
            f"pains={','.join(s['pain_points']) or 'none'}, disqualifiers={','.join(s['disqualifiers']) or 'none'}",
            f"  - Scorer: {row['score_summary']}" + (f" ({extra})" if extra else ""),
        ]
    out.append("")

    out += ["### Per-lead results", "", "| id | company | expected | predicted | score | ok | intent | size | email checks |",
            "|---|---|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        ec = row["email_checks"]
        ec_cell = "no draft" if not ec else ("pass" if ec["passed"] else "fail: " + ", ".join(
            k for k, v in ec["checks"].items() if not v))
        out.append(f"| {row['id']} | {row['company']} | {row['expected']} | {row['predicted']} | {row['score']:.1f} | "
                   f"{'✅' if row['correct'] else '❌'} | {row['signals']['buying_intent']} | "
                   f"{row['signals']['employee_band']} | {ec_cell} |")
    out.append("")

    e = r["email"]
    out += ["### Email quality: deterministic checks", "",
            f"{e['n_drafts']} drafts (disqualified leads get no draft). Pass rate per check:", "",
            "| check | pass rate |", "|---|---|"]
    for c in CHECKS:
        out.append(f"| {c} | {_pct(e['per_check'].get(c))} |")
    out.append(f"| **all checks** | **{_pct(e['all_checks_pass_rate'])}** |")
    out.append("")

    if r["judge"]["mean"]:
        jm = r["judge"]["mean"]
        out += [f"### Email quality: LLM judge (`{r['judge_model']}`, 1-5)", "",
                "| criterion | mean |", "|---|---|", *[f"| {c} | {jm[c]:.2f} |" for c in CRITERIA],
                f"| **overall** | **{jm['overall']:.2f}** |", ""]
        lows = [(row["id"], c, row["judge"][c]) for row in r["rows"] if row["judge"] for c in CRITERIA
                if row["judge"][c]["score"] <= 2]
        if lows:
            out += ["Lowest-scored items:", ""]
            out += [f"- {i} {c} = {j['score']}: {j['rationale']}" for i, c, j in lows[:10]]
            out.append("")
    else:
        out += ["### Email quality: LLM judge", "",
                "Not run for this mode (" + ("offline run; " if mode == "offline" else "") + PENDING + ").", ""]

    c = r["cost"]
    by_node = Counter()
    for row in r["rows"]:
        for u in row["usage"]:
            by_node[u["node"]] += u["cost_usd"]
    out += ["### Cost", "",
            f"- Pipeline (extraction + drafting): ${c['pipeline_usd']:.4f} total, **${c['pipeline_per_lead_usd']:.4f} per lead**",
            f"- Judge: ${c['judge_usd']:.4f} total",
            f"- Tokens: {c['input_tokens']:,} input / {c['output_tokens']:,} output",
            ]
    if by_node:
        out.append("- By node: " + ", ".join(f"{n} ${v:.4f}" for n, v in sorted(by_node.items())))
    if mode == "offline":
        out.append("- Offline mode makes no LLM calls, so its cost is genuinely $0.")
    out.append("")
    return out


def write_report(results_dir: Path, path: Path) -> Path:
    offline, llm = _load(results_dir, "offline"), _load(results_dir, "llm")
    leads_file = results_dir.parent / "leads.jsonl"
    leads = [json.loads(x) for x in leads_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    dist = Counter(x["expected_tier"] for x in leads)

    lines = [
        "# Eval report",
        "",
        "> Generated by `evals/run.py` (`make eval`, `make eval-llm`). Every number below comes from an actual run "
        "saved in `evals/results/*.json`; rows that have not been run say so.",
        "",
        "## Summary",
        "",
        "| Mode | Extraction | Drafting | Tier accuracy | Macro-F1 | Drafts passing all email checks | LLM-judge mean (1-5) | Pipeline cost / lead |",
        "|---|---|---|---|---|---|---|---|",
        _summary_row("Offline", "keyword/regex heuristics", "template", offline),
        _summary_row("LLM", f"`{(llm or {}).get('model') or 'claude-sonnet-5-5'}` structured outputs",
                     f"`{(llm or {}).get('model') or 'claude-sonnet-5-5'}`", llm),
        "",
        "## Methodology",
        "",
        f"- **Dataset**: {len(leads)} synthetic inbound leads in `evals/leads.jsonl` for clearly fictional companies "
        f"(`.example` domains), each with an offline homepage fixture in `fixtures/sites/`. Expected tiers: "
        + ", ".join(f"{t} {dist.get(t, 0)}" for t in TIERS) + ". Deliberately tricky cases: an ideal-ICP company "
        "with a vague message (E07), strong intent from a three-person company (E12), a university office that "
        "mentions students but is a real buyer (E15), a great-fit company whose sender is writing a thesis (E16), a "
        "competitor with budget (E17), a vendor pitch full of buying-signal words (E18), and a support lead whose "
        "workload phrase 'two of us' is not a headcount (E02).",
        f"- **Labels** were written from the intent of `config/icp.yaml` (what a sales lead applying the rubric "
        f"should conclude from the true facts), committed in `{LABEL_COMMIT}` before the scorer was run on this set, "
        "and not changed after seeing results. The heuristic extractor and the rubric thresholds were tuned only on "
        "the 8 development leads in `data/leads.sample.csv`. Disagreements below are reported, not fixed by relabelling.",
        "- **Pipeline under test**: the production LangGraph graph (normalize → enrich → extract_signals → score → "
        "route → draft_email) running until the `human_review` interrupt, with homepage fixtures and a throwaway "
        "state directory. Scoring is deterministic; only extraction and drafting differ between modes.",
        "- **Tier metrics**: accuracy, macro-F1 over the four classes (A, B, C, disqualified) and a confusion matrix.",
        f"- **Email checks** (deterministic, every draft): mentions the company; mentions at least one hand-listed "
        f"homepage fact for that lead; has a call to action; body length {MIN_WORDS}-{MAX_WORDS} words; no unfilled "
        f"placeholders; no claim from a {len(CLAIM_BLOCKLIST)}-phrase blocklist (guarantees, invented clients, "
        "'#1', ...) unless the lead's own homepage or message contains it; every \"you mentioned X\" names a tool "
        "that is actually in the lead's message.",
        "- **LLM judge** (LLM mode): Claude Haiku 5.5 with structured output scores personalization, relevance to the "
        "stated pain, CTA clarity, tone and factual grounding (1-5, with rationale), given the lead message, homepage "
        "text and seller description.",
        "- **Cost**: exact, from `response.usage` of every call at list prices (Sonnet 5.5 $2/$10, Haiku 5.5 "
        "$0.10/$0.50 per million input/output tokens).",
        "",
    ]
    if offline:
        m = offline["tier_metrics"]
        current = (m["accuracy"], m["macro_f1"], offline["email"]["all_checks_pass_rate"])
        if current != FINDINGS_DESCRIBE:
            lines += ["> **Note:** the offline numbers have changed since the findings below were written "
                      f"(they describe accuracy/macro-F1/email-pass {FINDINGS_DESCRIBE}; now {current}). "
                      "Re-review before relying on them.", ""]
        lines += [OFFLINE_FINDINGS, ""]
    for mode, result in (("offline", offline), ("llm", llm)):
        if result:
            lines += _mode_section(mode, result)
        else:
            lines += [f"## {MODE_TITLES[mode]}", "", f"Not run yet: {PENDING}.", ""]
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "make eval       # offline: no key needed, deterministic",
        "make eval-llm   # needs ANTHROPIC_API_KEY in .env; costs a few cents",
        "uv run python -m evals.run --report-only   # rebuild this file from saved results",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
