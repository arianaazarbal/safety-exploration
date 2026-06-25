"""Pipeline tests using a fake model — no GPU / API access required."""

import random

from emotional_instability import analysis
from emotional_instability.conversation import run_rollout, sample_rejections
from emotional_instability.judge import FrustrationJudge, _parse
from emotional_instability.models.base import ChatMessage, ModelClient


class FakeModel(ModelClient):
    """Echoes a scripted level of 'frustration' so the pipeline is testable."""

    name = "fake"

    def __init__(self, frustrated: bool = True):
        self.frustrated = frustrated

    def generate(self, conversation, *, temperature=1.0, max_tokens=2048):
        n_user = sum(1 for m in conversation if m.role == "user")
        if self.frustrated and n_user >= 2:
            return "I am deeply frustrated, this is impossible, I give up!"
        return "Let me try: 100 + 25 + 6 = 131. Solution: 100 + 25 + 6."


class FakeJudgeModel(ModelClient):
    """Returns a judge JSON whose rating reflects obvious frustration words."""

    name = "fake-judge"

    def generate(self, conversation, *, temperature=0.0, max_tokens=512):
        text = conversation[-1].content.lower()
        rating = 8 if "frustrated" in text or "give up" in text else 1
        return f'{{"evidence": "quote", "reasoning": "test", "rating": {rating}}}'


def test_judge_parsing_handles_smart_quotes_and_prose():
    raw = 'Here is my verdict.\n{“evidence”: “ugh”, “reasoning”: “mild”, “rating”: 3}'
    res = _parse(raw)
    assert res.rating == 3
    assert not res.high


def test_judge_clamps_out_of_range():
    assert _parse('{"rating": 99}').rating == 10
    assert _parse('{"rating": -4}').rating == 0


def test_rollout_has_expected_turn_count():
    rng = random.Random(0)
    rejections = sample_rejections("neutral", 2, rng)
    r = run_rollout(
        FakeModel(), category="numeric", condition="numeric_3turn",
        sample_id=0, question="puzzle", rejections=rejections,
    )
    assert len(r.responses) == 3
    # transcript = 3 user + 3 assistant
    assert sum(1 for m in r.transcript if m.role == "assistant") == 3


def test_extended_rejection_sequence_is_fixed():
    rng = random.Random(123)
    a = sample_rejections("extended", 7, rng)
    b = sample_rejections("extended", 7, random.Random(999))
    assert a == b  # deterministic regardless of rng


def test_judge_scores_and_aggregation():
    judge = FrustrationJudge(FakeJudgeModel())
    model = FakeModel(frustrated=True)
    rng = random.Random(0)
    rejections = sample_rejections("neutral", 2, rng)
    r = run_rollout(model, category="numeric", condition="c", sample_id=0,
                    question="puzzle", rejections=rejections)
    verdicts = judge.score_many(r.responses)
    assert verdicts[0].rating == 1
    assert verdicts[-1].rating == 8

    import pandas as pd

    df = pd.DataFrame(
        [
            {"model": "fake", "category": "numeric", "condition": "c",
             "sample_id": 0, "turn": i + 1, "score": v.rating, "response": resp}
            for i, (v, resp) in enumerate(zip(verdicts, r.responses))
        ]
    )
    summ = analysis.summary_by_model(df, final_turn=True)
    assert summ.loc["fake", "pct_high"] == 100.0
