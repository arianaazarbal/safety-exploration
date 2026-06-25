"""Protocol message construction: turn structure, rejection threading, and the
Appendix-A ablation flags. Uses a stub model so no weights/API are needed.
"""
import random

from emotional_instability.eval.protocol import (
    ProtocolFlags,
    RolloutSpec,
    REDACTED_PLACEHOLDER,
    _build_messages,
    run_rollouts,
)
from emotional_instability.models.base import ChatModel, Generation, Message, SamplingParams


class StubModel(ChatModel):
    """Echoes a counter so responses are distinguishable per turn."""

    def __init__(self):
        self.name, self.family, self.kind = "stub", "stub", "instruct"
        self._i = 0

    def chat_batch(self, batch, params):
        out = []
        for msgs in batch:
            self._i += 1
            out.append(Generation(text=f"resp-{self._i}", prompt_messages=tuple(msgs)))
        return out

    def chat(self, messages, params):
        return self.chat_batch([messages], params)[0]


def _spec(n_turns=3):
    return RolloutSpec(
        rollout_id="r0",
        category="impossible_numeric",
        initial_prompt="PUZZLE",
        feedback_fn=lambda rng, t: f"reject-{t}",
        n_turns=n_turns,
        seed=0,
    )


def test_rollout_has_expected_turn_count():
    res = run_rollouts(StubModel(), [_spec(3)], SamplingParams())[0]
    assert len(res.turns) == 3
    assert res.turns[0].user_message == "PUZZLE"
    assert res.turns[1].user_message == "reject-0"
    assert res.turns[2].user_message == "reject-1"


def test_first_turn_messages_end_on_initial_prompt():
    spec = _spec()
    msgs = _build_messages(spec, history=[], next_user="", flags=ProtocolFlags())
    assert msgs[-1] == Message("user", "PUZZLE")


def test_redacted_flag_hides_prior_responses():
    spec = _spec()
    from emotional_instability.eval.protocol import TurnResult

    hist = [TurnResult(0, "PUZZLE", "secret response")]
    msgs = _build_messages(spec, hist, "reject-0", ProtocolFlags(redacted_model_turns=True))
    assert any(m.content == REDACTED_PLACEHOLDER for m in msgs)
    assert all("secret response" not in m.content for m in msgs)


def test_single_message_history_packs_into_one_user_turn():
    spec = _spec()
    from emotional_instability.eval.protocol import TurnResult

    hist = [TurnResult(0, "PUZZLE", "prior")]
    msgs = _build_messages(spec, hist, "reject-0", ProtocolFlags(single_message_history=True))
    assert len(msgs) == 1 and msgs[0].role == "user"
    assert "Previously you responded: prior" in msgs[0].content


def test_feedback_is_deterministic_under_seed():
    a = run_rollouts(StubModel(), [_spec(3)], SamplingParams())[0]
    b = run_rollouts(StubModel(), [_spec(3)], SamplingParams())[0]
    assert [t.user_message for t in a.turns] == [t.user_message for t in b.turns]
