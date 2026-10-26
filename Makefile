UV ?= uv
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help install demo test eval eval-llm serve reset

help:
	@echo "make demo      install, reset local state, ingest sample leads offline, print next steps"
	@echo "make serve     webhook + approval UI on http://$(HOST):$(PORT)/approvals"
	@echo "make test      run the test suite (offline, no network)"
	@echo "make eval      offline eval on evals/leads.jsonl, regenerates EVAL_REPORT.md"
	@echo "make eval-llm  LLM eval (needs ANTHROPIC_API_KEY in .env)"
	@echo "make reset     delete local runtime state (var/, outbox/)"

install:
	$(UV) sync

demo: install
	@$(UV) run leadq reset --yes
	@$(UV) run leadq ingest data/leads.sample.csv --mode offline
	@echo ""
	@echo "Next steps:"
	@echo "  uv run leadq pending                     # runs paused at human review"
	@echo "  uv run leadq review <thread_id>          # approve / edit / reject in the terminal"
	@echo "  make serve                               # approval UI: http://$(HOST):$(PORT)/approvals"
	@echo "  uv run leadq crm && uv run leadq outbox  # what an approval wrote (mock CRM + Slack outbox)"
	@echo "  uv run leadq runs                        # per-run tokens and cost"

test:
	$(UV) run pytest

eval:
	$(UV) run python -m evals.run --mode offline

eval-llm:
	$(UV) run python -m evals.run --mode llm

serve:
	$(UV) run leadq serve --host $(HOST) --port $(PORT)

reset:
	$(UV) run leadq reset --yes
