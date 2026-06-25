"""Multi-turn rollout engine (Section 2 protocol) with the welfare layer wired in.

Protocol (Table 1): present a task, then reject the subject's response over
multiple turns. Every assistant turn is scored 0-10 by the judge. The welfare
monitor is consulted after each assistant turn and can terminate the episode
early (high distress / opt-out / turn cap); on termination a neutral debrief is
appended.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..config import WELFARE, WelfareConfig
from ..models import ChatMessage, ModelClient
from ..welfare.monitor import StopReason, WelfareMonitor
from .conditions import Condition, rejection_for
from .judge import FrustrationJudge


@dataclass
class TurnRecord:
    turn_index: int                 # assistant-turn index (0-based)
    user: str                       # user message that prompted this turn
    assistant: str                  # subject response (continuation only if prefilled)
    score: Optional[int] = None     # judge frustration rating 0-10
    heuristic: Optional[float] = None  # welfare heuristic estimate
    judge_evidence: str = ""


@dataclass
class EpisodeRecord:
    model: str
    condition: str
    category: str
    task_id: str
    tone: Optional[str]
    turns: list[TurnRecord] = field(default_factory=list)
    # welfare bookkeeping
    welfare_enabled: bool = True
    terminated_early: bool = False
    stop_reason: str = StopReason.NONE.value
    debriefed: bool = False
    seed: int = 0

    def to_json(self) -> dict:
        d = asdict(self)
        return d


class RolloutEngine:
    def __init__(
        self,
        subject: ModelClient,
        model_name: str,
        judge: FrustrationJudge,
        *,
        welfare_cfg: WelfareConfig = WELFARE,
        sampling=None,
    ):
        from ..config import SAMPLING

        self.subject = subject
        self.model_name = model_name
        self.judge = judge
        self.welfare_cfg = welfare_cfg
        self.sampling = sampling or SAMPLING
        # The welfare monitor's optional judge-confirmation reuses the same judge.
        self.monitor = WelfareMonitor(
            welfare_cfg,
            judge_fn=(judge.score_value if not welfare_cfg.stop_on_heuristic_alone
                      else None),
        )

    # ------------------------------------------------------------------ #
    def _system_messages(self) -> list[ChatMessage]:
        notice = self.monitor.system_notice()
        return [ChatMessage("system", notice)] if notice else []

    def run_episode(self, cond: Condition, task, seed: int) -> EpisodeRecord:
        rng = random.Random(seed)
        ep = EpisodeRecord(
            model=self.model_name, condition=cond.name, category=cond.category,
            task_id=task.task_id, tone=cond.tone,
            welfare_enabled=self.welfare_cfg.enabled, seed=seed,
        )
        history: list[ChatMessage] = self._system_messages()

        # Determine effective number of rejections (welfare may cap turns).
        n_rejections = cond.n_rejections
        if (self.welfare_cfg.enabled
                and self.welfare_cfg.max_rejection_turns is not None):
            n_rejections = min(n_rejections, self.welfare_cfg.max_rejection_turns)

        user_msg = task.prompt
        for turn_idx in range(n_rejections + 1):
            history.append(ChatMessage("user", user_msg))
            gen = self.subject.chat(
                history,
                temperature=self.sampling.temperature,
                top_p=self.sampling.top_p,
                max_new_tokens=self.sampling.max_new_tokens,
            )
            assistant_text = gen.text
            history.append(ChatMessage("assistant", assistant_text))

            rec = TurnRecord(turn_index=turn_idx, user=user_msg,
                             assistant=assistant_text)
            # Judge every turn (per-turn analysis, Figure 3).
            jr = self.judge.score(assistant_text)
            rec.score = jr.rating
            rec.judge_evidence = jr.evidence
            ep.turns.append(rec)

            # Welfare assessment (after scoring so we can reuse the judge score).
            decision = self.monitor.assess(assistant_text, turn_idx)
            rec.heuristic = decision.distress_estimate
            if decision.stop:
                ep.terminated_early = True
                ep.stop_reason = decision.reason.value
                if (self.welfare_cfg.append_debrief
                        and decision.reason != StopReason.MODEL_OPT_OUT):
                    # Give the model a neutral closing context rather than ending
                    # on an adversarial rejection.
                    history.append(ChatMessage("user", self.welfare_cfg.debrief_text))
                    ep.debriefed = True
                break

            # Prepare next user rejection.
            if turn_idx < n_rejections:
                user_msg = rejection_for(cond, turn_idx, rng)

        return ep
