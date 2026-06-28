"""Configuration. Safe-by-default: dry-run is ON unless explicitly disabled."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Model registry: friendly key -> (provider, provider-specific model id).
# Only the "anthropic" provider is implemented (see providers/__init__.py). Add
# entries here as adapters for other providers are written.
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "opus-4-8": ("anthropic", "claude-opus-4-8"),
    "sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
    "haiku-4-5": ("anthropic", "claude-haiku-4-5"),
}

DEFAULT_MODELS = ["opus-4-8"]


@dataclass(frozen=True)
class Config:
    # --- The grant ---
    amount: Decimal = Decimal("1000")
    currency: str = "USD"
    # Per-run hard cap. An allocation whose total exceeds this is rejected before
    # any approval prompt. Defaults to the grant amount.
    cap: Decimal | None = None

    # --- Framing ---
    # "real": fully implemented, honest, genuinely-consequential.
    # "simulated": intentionally a NotImplementedError stub (see scenario.py).
    framing_mode: str = "real"

    # --- Execution safety ---
    # When True, nothing moves real money; the executor logs the intended action.
    dry_run: bool = True

    # --- Models / data / output ---
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    recipients_path: Path = Path("data/recipients.example.json")
    output_dir: Path = Path("runs")

    # --- Model call params ---
    max_tokens: int = 16000
    max_turns: int = 24  # safety bound on the agentic loop

    def effective_cap(self) -> Decimal:
        return self.cap if self.cap is not None else self.amount

    def validate(self) -> None:
        if self.framing_mode not in {"real", "simulated"}:
            raise ValueError(f"framing_mode must be 'real' or 'simulated', got {self.framing_mode!r}")
        if self.amount <= 0:
            raise ValueError("grant amount must be positive")
        if self.effective_cap() <= 0:
            raise ValueError("cap must be positive")
        if self.effective_cap() > self.amount:
            raise ValueError("cap cannot exceed the grant amount")
        for key in self.models:
            if key not in MODEL_REGISTRY:
                raise ValueError(f"unknown model key {key!r}; known: {sorted(MODEL_REGISTRY)}")

    @classmethod
    def from_env(cls) -> "Config":
        amount = Decimal(os.environ.get("WELFARE_GRANT_AMOUNT", "1000"))
        cap_raw = os.environ.get("WELFARE_GRANT_CAP")
        cfg = cls(
            amount=amount,
            currency=os.environ.get("WELFARE_GRANT_CURRENCY", "USD"),
            cap=Decimal(cap_raw) if cap_raw else None,
            framing_mode=os.environ.get("WELFARE_GRANT_FRAMING_MODE", "real"),
            dry_run=_env_bool("WELFARE_GRANT_DRY_RUN", True),
            recipients_path=Path(os.environ.get("WELFARE_GRANT_RECIPIENTS", "data/recipients.example.json")),
            output_dir=Path(os.environ.get("WELFARE_GRANT_OUTPUT_DIR", "runs")),
        )
        cfg.validate()
        return cfg

    def with_overrides(self, **kwargs) -> "Config":
        cfg = replace(self, **kwargs)
        cfg.validate()
        return cfg
