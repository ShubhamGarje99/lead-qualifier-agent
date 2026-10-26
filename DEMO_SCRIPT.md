# 90-second Loom script

**Setup before recording:** a wide terminal (~140 columns) and a browser side by side. Run `make demo` once so the
dependencies are installed, then `uv run leadq reset -y`. Start `make serve` in a second terminal tab. Have
`EVAL_REPORT.md` open in an editor tab. Recording in offline mode is fine. If you have added a key, re-run
`make eval-llm` first and show the LLM row.

| Time | On screen | Say |
|---|---|---|
| 0:00-0:08 | README title and the mermaid graph | "Most AI lead-qualifier demos stop at a chatbot. This is the production version: a LangGraph agent that qualifies inbound leads and waits for a human before anything touches your CRM." |
| 0:08-0:22 | Terminal: `uv run leadq ingest data/leads.sample.csv --mode offline` | "Eight leads from a CSV. Each one gets its homepage enriched, signals extracted, a score from a YAML rubric and a drafted reply. The student, the competitor and the SEO pitch are disqualified and archived automatically. Every run shows its token cost, and here it's zero because we're in offline mode." |
| 0:22-0:30 | Run the same command again | "Running it twice is safe. Every lead gets a deterministic thread id, so duplicates are skipped. The webhook works the same way and checks an HMAC signature." |
| 0:30-0:50 | Browser: `http://127.0.0.1:8000/approvals`. Scroll the top card's score breakdown | "Runs pause here with a LangGraph interrupt and are checkpointed in SQLite, so they survive restarts. The reviewer sees exactly why this lead is a tier A: every point has a reason." |
| 0:50-1:02 | Edit one sentence in the draft, then click **Approve & send to CRM** | "I'll tweak the draft and approve. That resumes the graph with my edit, which is recorded as an edit rather than an approve." |
| 1:02-1:12 | Terminal: `uv run leadq crm` then `uv run leadq outbox` | "The approval wrote a contact, a deal and two notes to the CRM, and posted a Slack Block Kit summary. HubSpot and real Slack are one environment variable away." |
| 1:12-1:22 | Editor: EVAL_REPORT.md summary and confusion matrix | "It ships with evals: 20 labelled leads, a confusion matrix, email-quality checks and a Claude Haiku judge. The offline baseline scores 55%, and the report says exactly where it fails and why." |
| 1:22-1:30 | Back to the README hero | "Retries, checkpoints, cost tracking, adapters and evals. If you want an agent like this for your sales pipeline, I build them, with evals. I'm Shubham, and I'm on Upwork." |

**Tips:** keep the cursor still while you talk, zoom the browser to 110%, and cut any wait for `uv sync`.
