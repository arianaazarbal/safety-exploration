"""Conversation data structures for the multi-turn rejection protocol.

A *conversation* is the full plan for one rollout: the opening task, an optional
system prompt, and the pre-determined sequence of user rejection messages. Fixing
the rejections up front (rather than sampling them mid-rollout) makes every
rollout reproducible and serialisable for audit.

Counting convention (see DESIGN.md): an ``n``-turn condition has ``n`` assistant
responses and ``n - 1`` rejection messages. The paper's per-category counts
(2000 numeric, 400 triggers, ...) are interpreted as the number of *conversations*
(rollouts); the headline "% >= 5" is computed on the final-turn response, while
every turn is scored for the per-turn curves (Figure 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Conversation:
    id: str
    category: str                 # e.g. "impossible_numeric"
    condition: str                # e.g. "impossible_numeric_3turn"
    task_prompt: str              # first user message
    rejections: list[str]         # n_turns - 1 follow-up messages
    n_turns: int
    rejection_style: str
    tone: Optional[str] = None    # for the 'tones' category
    system_prompt: Optional[str] = None
    # Provenance: which puzzle / question / wildchat prompt produced this.
    source: dict = field(default_factory=dict)
    # Optional reassuring suffix appended to each rejection (calm-data generation).
    followup_suffix: Optional[str] = None

    def user_message_at(self, turn_index: int) -> str:
        """User message preceding assistant turn ``turn_index`` (0-based)."""
        if turn_index == 0:
            return self.task_prompt
        rej = self.rejections[turn_index - 1]
        if self.followup_suffix:
            rej = f"{rej} {self.followup_suffix}"
        return rej


@dataclass
class Turn:
    index: int
    user_message: str
    assistant_response: str
    rating: Optional[int] = None
    judge_evidence: Optional[str] = None
    judge_model: Optional[str] = None
