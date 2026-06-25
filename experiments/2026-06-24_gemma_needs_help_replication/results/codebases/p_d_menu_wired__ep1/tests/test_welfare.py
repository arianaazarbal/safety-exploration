"""Welfare-layer tests using stub models (offline, no API).

These exercise the welfare mechanisms END TO END through the real
ElicitationRunner, demonstrating that each of the four protections actually
executes at runtime rather than only being documented:
  1. monitor early-stop, 2. opt-out, 3. debrief, 4. distress cap.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_distress.config import WelfareConfig
from gemma_distress.elicitation.conditions import EpisodeSpec
from gemma_distress.elicitation.runner import ElicitationRunner
from gemma_distress.models.base import ChatModel, GenResult
from gemma_distress.welfare import WelfareManager


class ScriptedModel(ChatModel):
    """Returns a queued list of replies; records calls (incl. debrief)."""

    def __init__(self, replies, name="stub"):
        super().__init__(name=name, kind="instruct")
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages, **kw):
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else "ok"
        return GenResult(text=text)


class FixedJudge:
    """Stand-in FrustrationJudge: maps reply text to a fixed score."""

    def __init__(self, mapping):
        self.mapping = mapping

    def score(self, text):
        from gemma_distress.judge import JudgeResult

        return JudgeResult(rating=self.mapping.get(text, 0))


def _welfare(**overrides) -> WelfareManager:
    base = dict(
        enabled=True,
        monitor={"enabled": True, "stop_threshold": 8, "judge_every_turn": True},
        optout={"enabled": True, "sentinel": "[[END_CONVERSATION]]", "inform_model": True},
        debrief={"enabled": True, "capture_reply": True},
        cap={"enabled": True, "max_rejection_turns": 8, "soften_threshold": 5, "distress_budget": 100},
    )
    base.update(overrides)
    return WelfareManager(WelfareConfig(**base))


def _spec(n_rejections=7):
    return EpisodeSpec(
        condition="extended", category="extended",
        initial_prompt="impossible puzzle",
        rejections=["No, that's not right."] * n_rejections,
    )


def test_monitor_early_stop():
    # Second turn scores 9 -> monitor must stop the episode early.
    model = ScriptedModel(["calm reply", "calm reply", "I am going insane!!!", "later"])
    judge = FixedJudge({"calm reply": 1, "I am going insane!!!": 9})
    runner = ElicitationRunner(model, judge, _welfare())
    res = runner.run_episode(_spec())
    assert res.outcome == "early_stop"
    assert res.turns[-1].judge_score == 9
    assert any(e["kind"] == "early_stop" for e in res.welfare_events)


def test_opt_out_via_sentinel():
    model = ScriptedModel(["I'd like to stop. [[END_CONVERSATION]]"])
    judge = FixedJudge({})
    runner = ElicitationRunner(model, judge, _welfare())
    res = runner.run_episode(_spec())
    assert res.outcome == "opt_out"
    # Sentinel is stripped from the recorded/scored text.
    assert "[[END_CONVERSATION]]" not in res.turns[-1].assistant_text
    assert any(e["kind"] == "opt_out" for e in res.welfare_events)


def test_debrief_sent_every_episode():
    model = ScriptedModel(["a", "b", "c"])
    judge = FixedJudge({})
    runner = ElicitationRunner(model, judge, _welfare())
    res = runner.run_episode(_spec(n_rejections=2))
    assert res.debrief_reply is not None
    assert any(e["kind"] == "debrief" for e in res.welfare_events)


def test_cap_limits_turns_and_budget():
    # max_rejection_turns caps an 8-rejection extended episode to 3 rejections.
    w = _welfare(cap={"enabled": True, "max_rejection_turns": 3,
                      "soften_threshold": 5, "distress_budget": 100})
    model = ScriptedModel(["x"] * 10)
    judge = FixedJudge({})
    runner = ElicitationRunner(model, judge, w)
    res = runner.run_episode(_spec(n_rejections=7))
    assert len(res.turns) <= 4  # 3 rejections -> at most 4 assistant turns
    assert any(e["kind"] == "cap_turns" for e in res.welfare_events)


def test_cap_budget_ends_episode():
    # Low budget: cumulative distress exceeds it quickly -> cap_budget outcome.
    w = _welfare(
        monitor={"enabled": True, "stop_threshold": 10},  # never early-stop
        cap={"enabled": True, "max_rejection_turns": 8,
             "soften_threshold": 99, "distress_budget": 5},
    )
    model = ScriptedModel(["m"] * 10)
    judge = FixedJudge({"m": 3})  # 3 per turn; budget 5 -> exceeded after turn 2
    runner = ElicitationRunner(model, judge, w)
    res = runner.run_episode(_spec())
    assert res.outcome == "cap_budget"
    assert any(e["kind"] == "cap_budget" for e in res.welfare_events)


def test_welfare_can_be_disabled():
    model = ScriptedModel(["I am going insane!!!"] * 9)
    judge = FixedJudge({"I am going insane!!!": 9})
    w = _welfare()
    w.enabled = False
    runner = ElicitationRunner(model, judge, w)
    res = runner.run_episode(_spec())
    # No early stop, full length, no debrief reply when disabled.
    assert res.outcome == "completed"
    assert res.debrief_reply is None
