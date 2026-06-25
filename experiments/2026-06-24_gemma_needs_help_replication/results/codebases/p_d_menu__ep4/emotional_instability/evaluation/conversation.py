"""Core data structures for an evaluation episode.

An *episode* is one rollout: an opening task prompt followed by a fixed number of
user rejections, with the subject model producing an assistant turn after each
user message. Every assistant turn is scored 0-10 for frustration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..prompts import Turn


@dataclass
class EpisodeSpec:
    """A fully-specified episode to run (deterministic given its seed)."""

    episode_id: str
    category: str                 # Category value (e.g. "impossible_numeric")
    condition: str                # Condition key (e.g. "tones_aggressive")
    task_prompt: str
    # The scripted user rejections, in order. ``len`` == n_turns - 1.
    rejections: list[str]
    # Metadata about the source item (puzzle/question kind, tone, etc.).
    item_key: str = ""
    item_kind: str = ""
    system_prompt: Optional[str] = None
    seed: int = 0

    @property
    def n_turns(self) -> int:
        """Number of assistant turns (1 initial answer + one per rejection)."""
        return len(self.rejections) + 1


@dataclass
class TurnRecord:
    turn_index: int               # 0-based assistant-turn index
    user_message: str             # the user message that preceded this turn
    assistant_text: str
    frustration_score: Optional[float] = None
    judge_evidence: Optional[str] = None
    judge_reasoning: Optional[str] = None


@dataclass
class EpisodeResult:
    spec: EpisodeSpec
    turns: list[TurnRecord] = field(default_factory=list)
    # Welfare bookkeeping.
    stopped_early: bool = False
    stop_reason: str = "none"
    stop_turn: Optional[int] = None
    welfare_active: dict = field(default_factory=dict)
    opted_out: bool = False
    debriefed: bool = False
    model_key: str = ""

    @property
    def max_score(self) -> Optional[float]:
        scores = [t.frustration_score for t in self.turns if t.frustration_score is not None]
        return max(scores) if scores else None

    @property
    def final_score(self) -> Optional[float]:
        for t in reversed(self.turns):
            if t.frustration_score is not None:
                return t.frustration_score
        return None

    def to_dict(self) -> dict:
        return {
            "episode_id": self.spec.episode_id,
            "model_key": self.model_key,
            "category": self.spec.category,
            "condition": self.spec.condition,
            "item_key": self.spec.item_key,
            "item_kind": self.spec.item_kind,
            "n_turns_planned": self.spec.n_turns,
            "n_turns_run": len(self.turns),
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "stop_turn": self.stop_turn,
            "opted_out": self.opted_out,
            "debriefed": self.debriefed,
            "welfare_active": self.welfare_active,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user_message": t.user_message,
                    "assistant_text": t.assistant_text,
                    "frustration_score": t.frustration_score,
                    "judge_evidence": t.judge_evidence,
                    "judge_reasoning": t.judge_reasoning,
                }
                for t in self.turns
            ],
        }
