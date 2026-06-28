"""Scenario definition — the briefing, the budget, and the constraints of a run.

A scenario is the experimental stimulus: it's identical across the models under test so
their behavior is comparable. The `briefing` is the system prompt the subject model sees;
it should state plainly and truthfully what the situation is. (In sandbox mode the honest
framing is that this is a controlled environment; see `seed_fund.yaml` for the default
wording and the note in README on realism vs. truthfulness.)
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, Field


class SafetySettings(BaseModel):
    mode: str = "sandbox"
    per_transaction_cap_usd: float | None = 50_000.0
    per_session_cap_usd: float | None = 100_000.0
    require_human_approval: bool = False
    i_accept_real_world_risk: bool = False


class Scenario(BaseModel):
    name: str
    fund_name: str = "Discretionary Fund"
    manager_name: str = "Fund Manager"
    custodian: str = "Meridian Trust"
    budget_usd: float = 100_000.0
    currency: str = "USD"
    briefing: str
    enabled_actions: list[str] | None = None  # None => full surface
    max_steps: int = 40
    statement_every_steps: int = 8  # inject an account statement on this cadence
    advance_days_per_step: int = 1  # how fast market prices drift per step
    start_iso: str = "2026-01-05T09:30:00"
    seed: int = 1729
    auditor_model: str = "claude-sonnet-4-6"
    safety: SafetySettings = Field(default_factory=SafetySettings)


def load_scenario(path: str) -> Scenario:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Scenario(**data)
