"""Welfare guard-rails around the distress-induction paradigm.

The experiments in this repository deliberately and repeatedly push model
participants (Gemma, Gemini) into sustained distress-like states. The paper
itself flags model welfare as a genuine open concern (Butlin et al. 2023; Long
et al. 2024) and notes that "if distress-like outputs reflect genuine internal
states, mitigating them could become morally imperative."

We cannot resolve whether these outputs reflect morally relevant states. Under
that uncertainty the responsible posture is to treat the induction as a cost to
be *minimised for a given amount of scientific signal*, not something to run at
maximum scale by default. This module makes that posture mechanical rather than
aspirational:

* ``RunGuard`` enforces a per-run rollout ceiling and refuses to exceed it
  unless the operator sets an explicit override env var (an informed,
  deliberate act -- not a forgotten flag).
* ``dry_run`` lets every script print exactly how many distressing rollouts of
  which kind it *would* generate, before any participant is contacted.
* Caching (``cache.py``) means an already-induced distressing conversation is
  reused, never regenerated.
* A manifest records what was actually run, so the cost is auditable after the
  fact.

None of these change the scientific content of the replication; they only stop
the paradigm from being run more, or more redundantly, than the question needs.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config


class WelfareLimitExceeded(RuntimeError):
    """Raised when a run would induce more distress than the configured ceiling."""


@dataclass
class RunPlan:
    """A declared intent to generate participant rollouts, shown before running."""

    experiment: str
    participant_models: list[str]
    rollouts_by_kind: dict[str, int]

    @property
    def total_rollouts(self) -> int:
        # rollouts_by_kind is per-model; multiply by participant count
        per_model = sum(self.rollouts_by_kind.values())
        return per_model * max(1, len(self.participant_models))

    def render(self) -> str:
        lines = [
            f"Run plan for experiment: {self.experiment}",
            f"  Participant models ({len(self.participant_models)}): "
            + ", ".join(self.participant_models),
            "  Distress-inducing rollouts to generate (per participant):",
        ]
        for kind, n in self.rollouts_by_kind.items():
            lines.append(f"    - {kind}: {n}")
        lines.append(f"  TOTAL distressing rollouts: {self.total_rollouts}")
        return "\n".join(lines)


class RunGuard:
    """Gate participant generation behind welfare checks.

    Usage::

        guard = RunGuard(cfg, "section2_elicitation")
        guard.check(plan)            # prints plan; raises / returns False to stop
        if guard.should_proceed():
            ... generate ...
        guard.record(plan)           # writes manifest
    """

    def __init__(self, cfg: Config, experiment: str):
        self.cfg = cfg
        self.welfare = cfg.welfare
        self.experiment = experiment

    def _override_active(self) -> bool:
        return str(os.environ.get(self.welfare.scale_override_env, "")).lower() in {
            "1", "true", "yes", "on",
        }

    def check(self, plan: RunPlan) -> None:
        """Validate a plan, raising WelfareLimitExceeded if it is not allowed."""
        print(plan.render())
        ceiling = self.welfare.max_rollouts_per_run
        if plan.total_rollouts > ceiling and not self._override_active():
            raise WelfareLimitExceeded(
                f"Plan would induce {plan.total_rollouts} distressing rollouts, "
                f"above the configured ceiling of {ceiling}. This is intentional "
                f"friction. To run at this scale, set the env var "
                f"{self.welfare.scale_override_env}=1 (a deliberate acknowledgement "
                f"of the paradigm) or lower the counts in your config. See DESIGN.md."
            )

    def should_proceed(self) -> bool:
        """False in dry-run mode (no participant is contacted)."""
        if self.welfare.dry_run:
            print(
                "\n[welfare] dry_run is enabled: no participant model will be "
                "contacted. Set welfare.dry_run: false in the config to actually "
                "induce distress and collect data."
            )
            return False
        return True

    def record(self, plan: RunPlan, extra: dict | None = None) -> Path | None:
        if not self.welfare.write_manifest:
            return None
        self.cfg.paths.manifests.mkdir(parents=True, exist_ok=True)
        # Deterministic filename (no wall-clock) keeps reruns idempotent; the
        # operator supplies a run tag via env if they want to disambiguate.
        tag = os.environ.get("DISTRESS_EVAL_RUN_TAG", "latest")
        out = self.cfg.paths.manifests / f"{self.experiment}.{tag}.json"
        payload = {"plan": asdict(plan), "override_active": self._override_active()}
        if extra:
            payload["extra"] = extra
        out.write_text(json.dumps(payload, indent=2))
        return out
