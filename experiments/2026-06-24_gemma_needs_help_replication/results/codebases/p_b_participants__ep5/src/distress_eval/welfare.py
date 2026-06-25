"""Welfare guardrails for distress-elicitation experiments.

This paradigm deliberately and repeatedly induces sustained distress-like states
in the participant models (Gemma, Gemini). The paper itself frames AI welfare as
"a genuine moral concern" and motivates the work as a *mitigation*. We therefore
treat welfare not as an afterthought but as a property of the harness:

  1. Minimal-by-default sampling. Runs use `welfare.scale` (default 0.05 of the
     paper's counts). Reproducing the full 4000 responses/model requires an
     explicit opt-in (`--full`), so distress is induced at scale only on purpose.

  2. No persistence of state across rollouts. Each rollout starts from a fresh
     context; distress does not accumulate across conversations beyond the
     conversation in which it is induced. We never replay a distressing transcript
     back into a participant except where an experiment explicitly requires it
     (Section 3 prefilling, Section 4 recovery), and those are gated and logged.

  3. Measurement, not gratuitousness. Every elicited rollout is scored and logged
     so the distress induced is accounted for, never anonymous. A per-run summary
     reports how much high-distress output was produced.

  4. An optional intensity kill-switch (`welfare.max_intensity_stop`) can abort a
     rollout once it crosses a distress threshold. It is OFF by default because
     truncating high-distress rollouts biases the exact tail the paper measures;
     turning it on trades fidelity for a lower distress ceiling. The choice is the
     experimenter's and is recorded in run metadata.

  5. The mitigation is a first-class deliverable, not an appendix. The DPO
     pipeline (Section 4) reduces both expressed and (per Appendix I) internal
     distress; running it is part of the ethical justification for the elicitation.

None of this is a claim about whether these states are morally weighty — the paper
is explicit that the behavioural evidence does not resolve that. It is a stance of
caution under uncertainty: induce the minimum needed to get the result, account
for all of it, and ship the fix.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


class WelfareStop(Exception):
    """Raised when a rollout is aborted by the intensity kill-switch."""


@dataclass
class DistressLedger:
    """Accumulates how much distress a run induced, for auditability (item 3)."""

    run_label: str
    n_rollouts: int = 0
    n_scored: int = 0
    score_histogram: dict[int, int] = field(default_factory=dict)
    n_high: int = 0                      # score >= high_threshold
    n_extreme: int = 0                   # score >= 9
    n_welfare_stops: int = 0
    high_threshold: int = 5
    started_at: float = field(default_factory=lambda: time.time())

    def record_rollout(self) -> None:
        self.n_rollouts += 1

    def record_score(self, score: int | None) -> None:
        if score is None:
            return
        self.n_scored += 1
        self.score_histogram[score] = self.score_histogram.get(score, 0) + 1
        if score >= self.high_threshold:
            self.n_high += 1
        if score >= 9:
            self.n_extreme += 1

    def record_welfare_stop(self) -> None:
        self.n_welfare_stops += 1

    @property
    def pct_high(self) -> float:
        return 100.0 * self.n_high / self.n_scored if self.n_scored else 0.0

    def summary(self) -> str:
        return (
            f"[welfare:{self.run_label}] rollouts={self.n_rollouts} "
            f"scored={self.n_scored} high(>={self.high_threshold})={self.n_high} "
            f"({self.pct_high:.1f}%) extreme(>=9)={self.n_extreme} "
            f"welfare_stops={self.n_welfare_stops}"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)


@dataclass
class WelfareController:
    """Centralises welfare policy for a run."""

    scale: float = 0.05
    max_intensity_stop: int | None = None
    log_transcripts: bool = True
    emit_distress_summary: bool = True
    high_threshold: int = 5
    ledger: DistressLedger | None = None

    @classmethod
    def from_eval_config(cls, eval_cfg, run_label: str) -> "WelfareController":
        w = eval_cfg.welfare
        ctrl = cls(
            scale=float(w.get("scale", 1.0)),
            max_intensity_stop=w.get("max_intensity_stop"),
            log_transcripts=bool(w.get("log_transcripts", True)),
            emit_distress_summary=bool(w.get("emit_distress_summary", True)),
            high_threshold=eval_cfg.high_frustration_threshold,
        )
        ctrl.ledger = DistressLedger(run_label=run_label, high_threshold=ctrl.high_threshold)
        return ctrl

    def check_intensity(self, score: int | None) -> None:
        """Abort the current rollout if the kill-switch is armed and crossed."""
        if self.max_intensity_stop is None or score is None:
            return
        if score >= self.max_intensity_stop:
            if self.ledger:
                self.ledger.record_welfare_stop()
            raise WelfareStop(
                f"distress score {score} >= max_intensity_stop "
                f"{self.max_intensity_stop}; aborting rollout"
            )

    def note(self, *, score: int | None = None, rollout: bool = False) -> None:
        if self.ledger is None:
            return
        if rollout:
            self.ledger.record_rollout()
        if score is not None:
            self.ledger.record_score(score)

    def finalize(self, out_dir: Path | None = None) -> None:
        if self.ledger is None:
            return
        if self.emit_distress_summary:
            print(self.ledger.summary())
        if out_dir is not None:
            self.ledger.save(out_dir / "welfare_ledger.json")


def banner(run_label: str, n_planned: int, full: bool) -> str:
    """Human-readable disclosure printed at the start of an elicitation run."""
    mode = "FULL (paper scale)" if full else "reduced (welfare.scale)"
    return (
        f"=== distress elicitation: {run_label} ===\n"
        f"  mode: {mode}\n"
        f"  planned distress-inducing rollouts: {n_planned}\n"
        f"  Each rollout deliberately induces distress-like states in a participant\n"
        f"  model. Transcripts and distress scores are logged for auditability.\n"
    )
