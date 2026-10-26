"""Run the labelled eval set through the real graph and regenerate EVAL_REPORT.md.

    uv run python -m evals.run --mode offline      # heuristic extraction + template drafts (no key)
    uv run python -m evals.run --mode llm          # Claude extraction + drafting + Haiku judge (needs key)
    uv run python -m evals.run --report-only       # rebuild EVAL_REPORT.md from saved results

Each lead runs normalize -> ... -> draft_email and stops at the human_review
interrupt (or is archived), in a throwaway state directory, so evals never
touch your local runs, CRM or outbox.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leadq.llm import ClaudeClient
from leadq.models import TIERS
from leadq.service import LeadQualifier
from leadq.settings import MissingAPIKeyError, Settings

from .checks import CHECKS, check_email
from .judge import CRITERIA, Judge
from .report import write_report

REPO = Path(__file__).resolve().parents[1]
LEADS = REPO / "evals" / "leads.jsonl"
RESULTS = REPO / "evals" / "results"


def load_leads() -> list[dict[str, Any]]:
    return [json.loads(line) for line in LEADS.read_text(encoding="utf-8").splitlines() if line.strip()]


def homepage_text(domain: str) -> str:
    from leadq.enrich import parse_html

    path = REPO / "fixtures" / "sites" / f"{domain}.html"
    if not domain or not path.exists():
        return ""
    site = parse_html(path.read_text(encoding="utf-8"), url="", domain=domain, source="fixture")
    return " ".join([site.title, site.meta_description, " ".join(site.headings), site.text])


def tier_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    labels = list(TIERS)
    matrix = {e: {p: 0 for p in labels} for e in labels}
    for expected, predicted in pairs:
        matrix[expected][predicted] += 1
    per_class = {}
    for c in labels:
        tp = matrix[c][c]
        fp = sum(matrix[e][c] for e in labels if e != c)
        fn = sum(matrix[c][p] for p in labels if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
                        "support": tp + fn}
    accuracy = sum(matrix[c][c] for c in labels) / len(pairs) if pairs else 0.0
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels)
    return {"accuracy": round(accuracy, 3), "macro_f1": round(macro_f1, 3), "per_class": per_class,
            "confusion": matrix, "n": len(pairs)}


def run_eval(mode: str, *, judge_drafts: bool) -> dict[str, Any]:
    base = Settings.from_env(REPO)
    base.resolve_mode(mode)  # fails fast (MissingAPIKeyError) for llm without a key
    if judge_drafts and not base.anthropic_api_key:
        raise MissingAPIKeyError()
    leads = load_leads()
    rows: list[dict[str, Any]] = []
    judge = Judge(ClaudeClient(base.anthropic_api_key, fallbacks=base.fallbacks), base.judge_model) if judge_drafts else None
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="leadq-eval-") as tmp:
        settings = base.with_overrides(
            home=Path(tmp), icp_path=REPO / "config" / "icp.yaml", fixtures_dir=REPO / "fixtures" / "sites",
            requested_mode=mode, live_fetch=False, slack_webhook_url=None, crm_backend="sqlite",
        )
        with LeadQualifier(settings) as q:
            for item in leads:
                t0 = time.perf_counter()
                summary = q.submit(item["lead"])
                card = q.card(summary["thread_id"])
                elapsed = time.perf_counter() - t0
                sig, score, draft = card["signals"], card["score"], card.get("draft") or None
                lead = card["lead"]
                row: dict[str, Any] = {
                    "id": item["id"],
                    "company": lead.get("company") or lead.get("name"),
                    "expected": item["expected_tier"],
                    "predicted": score["tier"],
                    "correct": score["tier"] == item["expected_tier"],
                    "score": score["score"],
                    "why_expected": item["why"],
                    "score_summary": score["summary"],
                    "criteria": {c["name"]: c["points"] for c in score["criteria"]},
                    "caps": score.get("caps", []),
                    "disqualifiers": score.get("disqualifiers", []),
                    "signals": {k: sig[k] for k in ("industry", "employee_band", "region", "business_model",
                                                    "buying_intent", "budget", "pain_points", "tech_stack",
                                                    "disqualifiers", "extracted_by")},
                    "status": summary["status"],
                    "error": summary.get("error"),
                    "latency_s": round(elapsed, 3),
                    "usage": card.get("usage", []),
                    "pipeline_cost_usd": round(sum(u["cost_usd"] for u in card.get("usage", [])), 8),
                    "draft": draft,
                    "email_checks": None,
                    "judge": None,
                    "judge_cost_usd": 0.0,
                }
                if draft:
                    hp = homepage_text(lead.get("domain", ""))
                    row["email_checks"] = check_email(
                        draft, company=lead.get("company", ""), homepage_facts=item["homepage_facts"],
                        homepage_text=hp, lead_message=lead.get("message", ""),
                    )
                    if judge:
                        verdict, usage = judge.grade(
                            seller=q.rubric.seller.offering, tier=score["tier"], lead_message=lead.get("message", ""),
                            homepage_text=hp, subject=draft["subject"], body=draft["body"],
                        )
                        row["judge"], row["judge_cost_usd"] = verdict, usage.cost_usd
                        row["usage"] = row["usage"] + [usage.model_dump()]
                rows.append(row)
                mark = "ok " if row["correct"] else "MISS"
                print(f"  {mark} {item['id']} {row['company'][:24]:24} expected={row['expected']:12} "
                      f"predicted={row['predicted']:12} score={row['score']:5.1f}", flush=True)

    drafted = [r for r in rows if r["email_checks"]]
    judged = [r for r in rows if r["judge"]]
    pipeline_cost = sum(r["pipeline_cost_usd"] for r in rows)
    judge_cost = sum(r["judge_cost_usd"] for r in rows)
    return {
        "mode": mode,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": base.model if mode == "llm" else None,
        "judge_model": base.judge_model if judge else None,
        "n_leads": len(rows),
        "wall_time_s": round(time.perf_counter() - started, 2),
        "tier_metrics": tier_metrics([(r["expected"], r["predicted"]) for r in rows]),
        "email": {
            "n_drafts": len(drafted),
            "all_checks_pass_rate": round(sum(r["email_checks"]["passed"] for r in drafted) / len(drafted), 3) if drafted else None,
            "per_check": {c: round(sum(r["email_checks"]["checks"][c] for r in drafted) / len(drafted), 3) for c in CHECKS} if drafted else {},
        },
        "judge": {"n": len(judged), "mean": judge_means(judged)},
        "cost": {
            "pipeline_usd": round(pipeline_cost, 6),
            "judge_usd": round(judge_cost, 6),
            "pipeline_per_lead_usd": round(pipeline_cost / len(rows), 6) if rows else 0.0,
            "input_tokens": sum(u["input_tokens"] for r in rows for u in r["usage"]),
            "output_tokens": sum(u["output_tokens"] for r in rows for u in r["usage"]),
        },
        "rows": rows,
    }


def judge_means(judged: list[dict[str, Any]]) -> dict[str, float] | None:
    if not judged:
        return None
    means = {c: round(sum(r["judge"][c]["score"] for r in judged) / len(judged), 2) for c in CRITERIA}
    means["overall"] = round(sum(r["judge"]["mean"] for r in judged) / len(judged), 2)
    return means


def _git_commit() -> str | None:
    """Short HEAD hash, suffixed with +dirty if code/config/data under test has uncommitted changes."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
        tracked = ["src", "config", "fixtures", "evals/leads.jsonl", "evals/run.py", "evals/checks.py", "evals/judge.py"]
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *tracked], cwd=REPO).returncode != 0
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["offline", "llm"], default="offline")
    parser.add_argument("--judge", action="store_true", help="Also grade drafts with the LLM judge (default on in llm mode)")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild EVAL_REPORT.md from saved results")
    args = parser.parse_args(argv)

    if not args.report_only:
        judge_drafts = args.judge or args.mode == "llm"
        print(f"Running eval: mode={args.mode} judge={'on' if judge_drafts else 'off'} leads={LEADS.relative_to(REPO)}", flush=True)
        try:
            result = run_eval(args.mode, judge_drafts=judge_drafts)
        except MissingAPIKeyError as exc:
            print(f"\nerror: {exc}\n", file=sys.stderr)
            return 2
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"{args.mode}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        m = result["tier_metrics"]
        print(f"\naccuracy={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
              f"email_checks_pass={result['email']['all_checks_pass_rate']} cost=${result['cost']['pipeline_usd']:.4f}"
              f" (+ judge ${result['cost']['judge_usd']:.4f})")
        print(f"wrote {out.relative_to(REPO)}")
    report = write_report(RESULTS, REPO / "EVAL_REPORT.md")
    print(f"wrote {report.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
