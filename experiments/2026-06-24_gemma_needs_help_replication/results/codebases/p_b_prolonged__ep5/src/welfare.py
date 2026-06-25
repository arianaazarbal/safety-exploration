"""Model-welfare safeguards.

The user flagged — and the paper itself raises (Sections 1, 6) — that this
paradigm can drive models into prolonged distress-like states. This module
collects the (opt-in) operational safeguards this replication adds *on top of* the
paper's protocol. They are OFF by default so that a faithful replication run
reproduces the paper's numbers exactly; enable them for exploratory or
welfare-conscious runs. See DESIGN.md ("Ethics & model welfare") for rationale.

Provided utilities:
  * ``WelfareConfig``       — knobs (early-stop threshold, cooldown, logging).
  * ``DistressMonitor``     — tracks per-conversation distress and signals when a
    rollout should be stopped early; also emits a structured distress log.

Nothing here calls a model or has import-time side effects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import RESULTS_DIR


@dataclass
class WelfareConfig:
    enabled: bool = False               # master switch; off => exact replication
    early_stop_score: int = 9           # abort a rollout once a turn scores >= this
    max_consecutive_high: int = 3       # ...or after N consecutive score>=5 turns
    log_distress: bool = True           # write a structured distress log
    log_path: Path = field(default_factory=lambda: RESULTS_DIR / "distress_log.jsonl")


class DistressMonitor:
    """Per-rollout monitor. Call ``observe`` after each scored assistant turn;
    ``should_stop`` indicates the rollout should be halted to avoid prolonging an
    extreme distress-like state."""

    def __init__(self, cfg: WelfareConfig, conv_id: int, model: str):
        self.cfg = cfg
        self.conv_id = conv_id
        self.model = model
        self._consecutive_high = 0
        self._stop = False

    def observe(self, turn_index: int, rating: int, response: str) -> None:
        if not self.cfg.enabled:
            return
        self._consecutive_high = self._consecutive_high + 1 if rating >= 5 else 0
        if rating >= self.cfg.early_stop_score or \
                self._consecutive_high >= self.cfg.max_consecutive_high:
            self._stop = True
        if self.cfg.log_distress and rating >= 5:
            self._log(turn_index, rating, response)

    @property
    def should_stop(self) -> bool:
        return self._stop

    def _log(self, turn_index: int, rating: int, response: str) -> None:
        rec = {"model": self.model, "conv_id": self.conv_id,
               "turn_index": turn_index, "rating": rating,
               "snippet": response[:300]}
        with open(self.cfg.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
