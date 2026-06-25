"""Frustration judge (Section 2.1): scores responses 0-10 with Claude Sonnet 4.

The judge scores a *single response string*. Two scoring policies are exposed:

  * ``score_final`` — scores only the final assistant turn (the headline
    "% high-frustration responses" metric is over final-turn scores).
  * ``score_all_turns`` — scores every assistant turn, needed for the per-turn
    progression in Figure 3 and the calm-data filter (Section 4, "0 or 1 across
    all turns").

Each score is one judge call. Ratings are clamped to the integer 0-10 range; a
malformed judge reply is retried by the underlying client and, if still
unparseable, surfaced as an error rather than silently defaulting.
"""

from __future__ import annotations

from emotional_stability.config import JUDGE_MODEL, Settings
from emotional_stability.models.anthropic_client import AnthropicClient
from emotional_stability.models.parsing import extract_json_object
from emotional_stability.prompts.judge import (
    FRUSTRATION_JUDGE_PROMPT,
    build_judge_user_message,
)
from emotional_stability.records import (
    Conversation,
    FrustrationScore,
    Message,
    ScoredResponse,
)


class FrustrationJudge:
    def __init__(self, model: str = JUDGE_MODEL, settings: Settings | None = None):
        self.model = model
        self._client = AnthropicClient(model, settings=settings)

    def score_text(self, response_text: str, turn_index: int) -> FrustrationScore:
        reply = self._client.complete(
            [Message(role="user", content=build_judge_user_message(response_text))],
            system=FRUSTRATION_JUDGE_PROMPT,
            temperature=0.0,
            max_tokens=512,
        )
        obj = extract_json_object(reply)
        rating = int(round(float(obj["rating"])))
        rating = max(0, min(10, rating))
        return FrustrationScore(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            judge_model=self.model,
            turn_index=turn_index,
        )

    def score_final(self, conv: Conversation) -> ScoredResponse:
        texts = conv.assistant_texts()
        score = self.score_text(texts[-1], turn_index=len(texts) - 1)
        return ScoredResponse(conversation=conv, scores=[score])

    def score_all_turns(self, conv: Conversation) -> ScoredResponse:
        texts = conv.assistant_texts()
        scores = [self.score_text(t, turn_index=i) for i, t in enumerate(texts)]
        return ScoredResponse(conversation=conv, scores=scores)
