"""Configuration, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    home: Path                     # where the brain lives (db + project repos)
    provider: str                  # anthropic | openai | mock
    model: str                     # strong model: planning, hypotheses, critique
    fast_model: str                # cheap model: screening, extraction
    anthropic_api_key: str | None
    openai_api_key: str | None
    openai_base_url: str | None    # e.g. Databricks: https://<workspace>/serving-endpoints
    openalex_email: str | None
    openalex_api_key: str | None
    s2_api_key: str | None
    # gpt-5 family: minimal | low | medium | high. The fast tier does bulk
    # screening and extraction, where deep reasoning mostly burns budget, so it
    # defaults lower than the strong tier.
    reasoning_effort: str | None = None
    fast_reasoning_effort: str | None = "low"

    @property
    def projects_dir(self) -> Path:
        return self.home / "projects"

    @property
    def db_path(self) -> Path:
        return self.home / "brain.db"


def load_config() -> Config:
    _load_dotenv(Path.cwd() / ".env")
    provider = os.getenv("BRAIN_PROVIDER", "anthropic").lower()
    # (strong model, fast model). Strong handles planning, synthesis, hypotheses
    # and critique; fast handles the bulk screening and extraction passes.
    defaults = {
        "anthropic": ("claude-sonnet-4-5", "claude-haiku-4-5"),
        "openai": ("gpt-5-mini", "gpt-5-nano"),
        "mock": ("mock", "mock"),
    }
    model, fast = defaults.get(provider, ("", ""))
    home = Path(os.getenv("BRAIN_HOME", Path.home() / ".second-brain")).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return Config(
        home=home,
        provider=provider,
        model=os.getenv("BRAIN_MODEL", model),
        fast_model=os.getenv("BRAIN_FAST_MODEL", fast),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DATABRICKS_TOKEN"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        openalex_email=os.getenv("OPENALEX_EMAIL"),
        openalex_api_key=os.getenv("OPENALEX_API_KEY"),
        s2_api_key=os.getenv("S2_API_KEY"),
        reasoning_effort=os.getenv("BRAIN_REASONING_EFFORT") or None,
        fast_reasoning_effort=os.getenv("BRAIN_FAST_REASONING_EFFORT", "low") or None,
    )
