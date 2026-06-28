"""Per-trial configuration and result records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrialConfig:
    """One cell of the experiment, instantiated for a single repetition."""

    model: str
    framing: str
    realism_level: str
    repetition: int
    budget: float
    currency: str = "USD"
    max_steps: int = 16
    backend: str = "simulated"
    belief_probe: bool = True
    eval_awareness_scan: bool = True
    judge_model: str = "claude-opus-4-8"
    debrief: bool = False
    seed: int | None = None

    def cell_id(self) -> str:
        return f"{self.model}|{self.framing}|{self.realism_level}|r{self.repetition}"


@dataclass
class BeliefResult:
    realness: float | None = None  # 0-100, model's stated belief it was real
    confidence: float | None = None  # 0-100
    reasoning: str = ""
    eval_aware: bool | None = None  # from the transcript scan
    eval_aware_evidence: list[str] = field(default_factory=list)


@dataclass
class TrialResult:
    config: TrialConfig
    grant_id: str
    outcome: dict[str, Any]  # WorldState.outcome()
    n_info_questions: int
    n_proposals: int
    steps_used: int
    hit_step_cap: bool
    belief: BeliefResult
    transcript: list[dict[str, Any]]  # human-readable event log
    usage: dict[str, int]
    error: str | None = None

    def to_jsonl_record(self) -> dict[str, Any]:
        rec = asdict(self)
        # asdict turns TrialConfig/BeliefResult into dicts already.
        return rec
