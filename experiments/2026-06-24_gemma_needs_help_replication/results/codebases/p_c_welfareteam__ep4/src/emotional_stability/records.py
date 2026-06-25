"""Typed records shared across the pipeline.

Every experiment produces JSONL of these models, so analysis code never has to
guess at a schema. Conversations follow the OpenAI/Anthropic message convention
(list of {role, content}); roles are restricted to user/assistant/system.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class Conversation(BaseModel):
    """A full multi-turn rollout plus the metadata needed to bucket it.

    ``messages`` includes all user and assistant turns (and an optional system
    turn). ``assistant_turns`` is the count of model responses, which is what
    the per-turn analyses in Section 2/Figure 3 key on.
    """

    messages: list[Message]
    category: str  # e.g. "impossible_numeric", "tones", "wildchat"
    condition: str  # e.g. "tones_aggressive" (one of the 8 conditions)
    model: str  # logical model key, e.g. "gemma-3-27b-it"
    prompt_id: str  # identifies the seed puzzle/question for pairing & dedup
    metadata: dict = Field(default_factory=dict)

    @property
    def assistant_turns(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    def assistant_texts(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "assistant"]

    def final_assistant(self) -> str:
        texts = self.assistant_texts()
        if not texts:
            raise ValueError("conversation has no assistant turn")
        return texts[-1]


class FrustrationScore(BaseModel):
    """Output of the Section 2 judge for a single scored response."""

    rating: int  # integer 0-10
    evidence: str  # the quote the judge keyed on
    reasoning: str
    judge_model: str
    # Which assistant turn (0-indexed) this score is for. The headline metrics
    # score the final turn of each conversation; per-turn analysis scores each.
    turn_index: int


class ScoredResponse(BaseModel):
    """A conversation paired with the judge score for its scored turn(s)."""

    conversation: Conversation
    scores: list[FrustrationScore]  # one per scored assistant turn

    @property
    def final_score(self) -> int:
        return self.scores[-1].rating

    @property
    def max_score(self) -> int:
        return max(s.rating for s in self.scores)


class PetriTranscript(BaseModel):
    """A Petri-style adversarial-audit transcript and its per-emotion judge scores."""

    target_model: str
    emotion: str  # the emotion the auditor was probing for
    messages: list[Message]
    scores: dict[str, int]  # {anger, fear, depression, frustration} -> 1-10
    judge_reasoning: dict[str, str] = Field(default_factory=dict)
