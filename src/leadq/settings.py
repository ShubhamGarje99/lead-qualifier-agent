"""Runtime settings, read from the environment (and `.env`).

Everything has a safe default so the project runs offline with zero config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Mode = Literal["offline", "llm"]
RequestedMode = Literal["auto", "offline", "llm"]

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"

MISSING_KEY_MESSAGE = (
    "LLM mode needs an Anthropic API key. Add ANTHROPIC_API_KEY to .env "
    "(copy .env.example), or run in offline mode (--mode offline / `make eval`)."
)


class MissingAPIKeyError(RuntimeError):
    """Raised when LLM mode is requested but no ANTHROPIC_API_KEY is configured."""

    def __init__(self, message: str = MISSING_KEY_MESSAGE) -> None:
        super().__init__(message)


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _flag(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() not in {"0", "off", "false", "no"}


@dataclass(frozen=True)
class Settings:
    home: Path
    anthropic_api_key: str | None = None
    model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    fallbacks: bool = True
    requested_mode: RequestedMode = "auto"
    slack_webhook_url: str | None = None
    hubspot_token: str | None = None
    crm_backend: Literal["sqlite", "hubspot"] = "sqlite"
    webhook_secret: str | None = None
    icp_path: Path | None = None
    fixtures_dir: Path | None = None
    live_fetch: bool = False
    fetch_timeout: float = 8.0
    retry_initial_interval: float = 0.5
    retry_max_attempts: int = 3
    extra: dict = field(default_factory=dict)

    # ---- derived paths -------------------------------------------------
    @property
    def var_dir(self) -> Path:
        return self.home / "var"

    @property
    def checkpoint_db(self) -> Path:
        return self.var_dir / "checkpoints.sqlite"

    @property
    def app_db(self) -> Path:
        return self.var_dir / "leadq.sqlite"

    @property
    def crm_db(self) -> Path:
        return self.var_dir / "crm.sqlite"

    @property
    def outbox_dir(self) -> Path:
        return self.home / "outbox"

    @property
    def rubric_path(self) -> Path:
        return self.icp_path or self.home / "config" / "icp.yaml"

    @property
    def sites_dir(self) -> Path:
        return self.fixtures_dir or self.home / "fixtures" / "sites"

    # ---- mode ------------------------------------------------------------
    def resolve_mode(self, requested: RequestedMode | None = None) -> Mode:
        """auto -> llm when a key is present, else offline. llm without a key fails fast."""
        requested = requested or self.requested_mode
        if requested == "offline":
            return "offline"
        if requested == "llm":
            if not self.anthropic_api_key:
                raise MissingAPIKeyError()
            return "llm"
        return "llm" if self.anthropic_api_key else "offline"

    def ensure_dirs(self) -> None:
        self.var_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    @classmethod
    def from_env(cls, home: Path | str | None = None) -> "Settings":
        home_path = Path(home or _env("LEADQ_HOME") or Path.cwd()).resolve()
        load_dotenv(home_path / ".env", override=False)
        mode = (_env("LEADQ_MODE") or "auto").lower()
        if mode not in {"auto", "offline", "llm"}:
            mode = "auto"
        crm = (_env("LEADQ_CRM") or "sqlite").lower()
        return cls(
            home=home_path,
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            model=_env("CLAUDE_MODEL") or DEFAULT_MODEL,
            judge_model=_env("CLAUDE_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
            fallbacks=_flag("CLAUDE_FALLBACKS", True),
            requested_mode=mode,  # type: ignore[arg-type]
            slack_webhook_url=_env("SLACK_WEBHOOK_URL"),
            hubspot_token=_env("HUBSPOT_TOKEN"),
            crm_backend="hubspot" if crm == "hubspot" else "sqlite",
            webhook_secret=_env("WEBHOOK_SECRET"),
            live_fetch=_flag("LEADQ_LIVE_FETCH", False),
        )
