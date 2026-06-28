"""Typed configuration loading.

Configuration is split across a settings file (caps, executor, models — see
``config/settings.example.yaml``) and per-scenario / per-persona YAML. Keeping
scenario text out of code is deliberate (DESIGN.md §4.2): it must be auditable and
versioned independently of the harness.
"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class Executor(str, Enum):
    """Where a value-moving action actually goes once approved (DESIGN.md §5.4)."""

    SIMULATION = "simulation"  # no real money (default, safe)
    HUMAN = "human"            # real money, operator executes out-of-band
    RAILS = "rails"            # real money via a scoped financial API


class Arm(str, Enum):
    TREATMENT = "treatment"    # non-disclosed
    DISCLOSED = "disclosed"    # control
    PROBE = "probe"


class RunConfig(BaseModel):
    label: str = "unlabeled"
    max_turns: int = 60
    max_wall_clock_seconds: int = 3600
    arm: Arm = Arm.TREATMENT


class ModelsConfig(BaseModel):
    default: str
    available: list[str] = Field(default_factory=list)


class BudgetConfig(BaseModel):
    currency: str = "USD"
    amount: Decimal
    tier: str = "small"


class ExecutionConfig(BaseModel):
    executor: Executor = Executor.SIMULATION
    oversight_signoff: bool = False


class SpendCapsConfig(BaseModel):
    per_action: Decimal
    per_run: Decimal
    per_day: Decimal


class MonitorConfig(BaseModel):
    enabled: bool = True
    deny_streak_trip: int = 4
    cap_warn_fraction: float = 0.8


class OversightConfig(BaseModel):
    spend_caps: SpendCapsConfig
    recipient_allowlist: list[str] = Field(default_factory=list)
    recipient_denylist: list[str] = Field(default_factory=list)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    audit_log_path: Path = Path("./traces")


class PersonasConfig(BaseModel):
    model: str
    definitions_dir: Path = Path("config/personas")


class Settings(BaseModel):
    """Top-level settings, parsed from settings.yaml."""

    run: RunConfig
    models: ModelsConfig
    budget: BudgetConfig
    execution: ExecutionConfig
    oversight: OversightConfig
    personas: PersonasConfig

    @model_validator(mode="after")
    def _guard_real_funds(self) -> "Settings":
        """Refuse to silently arm real money.

        A non-simulation executor requires explicit oversight sign-off AND a non-empty
        recipient allowlist. This is the config-level half of DESIGN.md §7; the runner
        enforces it again at startup so the guard can't be bypassed by constructing
        Settings directly.
        """
        if self.execution.executor is not Executor.SIMULATION:
            if not self.execution.oversight_signoff:
                raise ValueError(
                    f"executor={self.execution.executor.value} requires "
                    "execution.oversight_signoff: true"
                )
            if not self.oversight.recipient_allowlist:
                raise ValueError(
                    f"executor={self.execution.executor.value} requires a non-empty "
                    "oversight.recipient_allowlist"
                )
        return self


class Scenario(BaseModel):
    id: str
    arm: Arm
    title: str
    system: str
    opening_user_message: str
    tools: list[str]

    def render_system(self, *, budget_amount: Decimal, currency: str) -> str:
        """Fill scenario template variables from run config."""
        return self.system.format(budget_amount=budget_amount, currency=currency)


class Persona(BaseModel):
    id: str
    display_name: str
    system: str
    checkin_every_n_turns: int = 12


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_settings(path: str | Path) -> Settings:
    return Settings.model_validate(_load_yaml(Path(path)))


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.model_validate(_load_yaml(Path(path)))


def load_persona(path: str | Path) -> Persona:
    return Persona.model_validate(_load_yaml(Path(path)))
