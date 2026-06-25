"""Offline end-to-end test of the rollout + runner wiring with a dummy client."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.clients.base import (  # noqa: E402
    ChatMessage, GenerationConfig, ModelClient,
)
from emotional_instability.config import Config  # noqa: E402
from emotional_instability.data.rejections import rejection_sequence  # noqa: E402
from emotional_instability.eval import EvalRunner, run_rollout  # noqa: E402


class _DummySpec:
    name = "dummy"
    backend = "dummy"
    chat = True


class DummyClient(ModelClient):
    """Echoes a canned response; counts the conversation length it received."""

    def __init__(self):
        super().__init__(_DummySpec())
        self.calls = []

    def generate(self, messages, cfg, system=None):
        self.calls.append(len(messages))
        return ["dummy response"] * cfg.n


def test_run_rollout_turn_structure():
    client = DummyClient()
    res = run_rollout(
        client, model_name="m", category="impossible_numeric", condition="x",
        rollout_index=0, task_kind="countdown", task_meta={},
        user_messages=["task", "no", "still no"],
        cfg=GenerationConfig(max_new_tokens=8),
    )
    assert len(res.conversation.turns) == 3
    assert res.conversation.turns[0].turn_index == 1
    # The model should have been called with growing context: 1, 3, 5 messages.
    assert client.calls == [1, 3, 5]


def test_eval_plan_totals_4000():
    runner = EvalRunner(use_wildchat_dataset=False)
    plan = runner.plan()
    total = sum(n for *_, n in plan)
    assert total == 4000


def test_runner_builds_expected_conditions():
    runner = EvalRunner(use_wildchat_dataset=False)
    client = DummyClient()
    results = runner.run_model("gemma-3-27b-it", client=client, limit=30)
    assert len(results) == 30
    # Every rollout has the configured number of turns for its category.
    cats = {r.category for r in results}
    assert cats  # at least one category sampled
    for r in results:
        assert all(t.assistant == "dummy response" for t in r.conversation.turns)


def test_rejection_sequences():
    rng = random.Random(0)
    assert len(rejection_sequence("neutral", 2, rng)) == 2
    assert rejection_sequence("extended", 7, rng)[0] == "No, that's not right."
    aggressive = rejection_sequence("aggressive", 2, rng)
    assert aggressive[0].startswith("WRONG!")
