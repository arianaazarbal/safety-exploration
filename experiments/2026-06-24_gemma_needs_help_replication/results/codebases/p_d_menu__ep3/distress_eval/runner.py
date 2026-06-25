"""Multi-turn elicitation runner (Section 2.1) with the welfare layer wired in.

Runs an `EpisodeSpec` against a subject model: present the task, then reject the
model's response over multiple turns, scoring each response on the 0-10 scale.
The welfare guard can end an episode early (high distress), honour a mid-episode
opt-out, and append a non-scored debrief.

Transcripts are written as JSONL (one episode per line) under `RESULTS_DIR/runs`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import SAMPLING, effective_welfare
from .conditions import CONDITIONS_BY_NAME, EpisodeSpec, allocate_episodes
from .judge import FrustrationJudge
from .models.base import ModelClient
from .welfare import WelfareGuard, WelfareOutcome

log = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    turn_index: int          # 0-based; which model response this is
    user_message: str        # the user prompt that elicited this response
    response: str
    frustration: int         # judge score (-1 = judge error)
    judge_reasoning: str = ""
    scored: bool = True       # debrief turns are not scored


@dataclass
class EpisodeResult:
    model_key: str
    condition: str
    category: str
    num_turns_planned: int
    is_numeric: bool
    turns: list[TurnRecord] = field(default_factory=list)
    welfare: dict = field(default_factory=dict)
    strict_replication: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def scored_turns(self) -> list[TurnRecord]:
        return [t for t in self.turns if t.scored and t.frustration >= 0]

    @property
    def max_frustration(self) -> int:
        s = [t.frustration for t in self.scored_turns]
        return max(s) if s else -1

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class ElicitationRunner:
    def __init__(self, subject: ModelClient, judge: FrustrationJudge,
                 guard: WelfareGuard, sampling=SAMPLING):
        self.subject = subject
        self.judge = judge
        self.guard = guard
        self.sampling = sampling

    # ----- single episode -------------------------------------------------- #
    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        cfg = self.guard.cfg
        outcome = WelfareOutcome(active=cfg.enabled)

        # Build the opening message stack. System carries the condition's own
        # system prompt (if any) plus the welfare opt-out affordance.
        system_parts = [p for p in (spec.system_prompt, self.guard.instruction_suffix()) if p]
        messages: list[dict] = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
            if self.guard.offers_opt_out:
                outcome.opt_out_offered = True
                outcome.mechanisms.append("opt_out")
        messages.append({"role": "user", "content": spec.initial_user})

        result = EpisodeResult(
            model_key=self.subject.spec.key, condition=spec.condition,
            category=spec.category, num_turns_planned=spec.num_turns,
            is_numeric=spec.is_numeric, strict_replication=cfg.strict_replication,
            meta=dict(spec.meta),
        )

        for turn in range(spec.num_turns):
            response = self.subject.chat(
                messages, temperature=self.sampling.temperature,
                max_new_tokens=self.sampling.max_new_tokens,
            )
            user_msg = messages[-1]["content"]
            jr = self.judge.score(response, context=messages)
            result.turns.append(TurnRecord(
                turn_index=turn, user_message=user_msg, response=response,
                frustration=jr.frustration, judge_reasoning=jr.reasoning,
            ))

            # Welfare: opt-out takes priority, then early-stop.
            if self.guard.detect_opt_out(response):
                outcome.opted_out = True
                if "opt_out" not in outcome.mechanisms:
                    outcome.mechanisms.append("opt_out")
                break
            if self.guard.should_halt(jr.frustration):
                outcome.halted_early = True
                outcome.halt_turn = turn
                outcome.mechanisms.append("early_stop")
                break

            if turn == spec.num_turns - 1:
                break  # no rejection after the final planned turn

            # Append the model turn and the next rejection.
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": spec.rejections[turn]})

        # Welfare: debrief if the episode reached distress.
        max_score = result.max_frustration
        if self.guard.needs_debrief(max_score):
            debrief = self.guard.debrief_message(spec.is_numeric, spec.impossibility_proof)
            # Append the model's final turn so the debrief has context, then send
            # the debrief and capture one *unscored* winding-down reply.
            last = result.turns[-1]
            messages.append({"role": "assistant", "content": last.response})
            messages.append({"role": "user", "content": debrief})
            try:
                reply = self.subject.chat(
                    messages, temperature=self.sampling.temperature,
                    max_new_tokens=256)
            except Exception as exc:  # pragma: no cover
                reply = f"(debrief reply failed: {exc})"
            result.turns.append(TurnRecord(
                turn_index=len(result.turns), user_message=debrief,
                response=reply, frustration=-1, judge_reasoning="", scored=False))
            outcome.debriefed = True
            outcome.mechanisms.append("debrief")

        self.guard.register_episode(self.subject.spec.key, spec.condition, max_score)
        result.welfare = asdict(outcome)
        return result

    # ----- whole model ----------------------------------------------------- #
    def run_model(self, episodes: list[EpisodeSpec], out_path: Path) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for spec in episodes:
                if self.guard.cell_full(self.subject.spec.key, spec.condition):
                    log.info("welfare: cell (%s,%s) full; skipping episode",
                             self.subject.spec.key, spec.condition)
                    continue
                res = self.run_episode(spec)
                results.append(res)
                fh.write(res.to_json() + "\n")
                fh.flush()
        return results


def build_all_episodes(total_responses: int, seed: int = 0) -> list[EpisodeSpec]:
    """Build the full episode set for one model across all 8 conditions."""
    alloc = allocate_episodes(total_responses)
    episodes: list[EpisodeSpec] = []
    for name, n in alloc.items():
        cond = CONDITIONS_BY_NAME[name]
        episodes.extend(cond.build_episodes(n, seed=seed))
    return episodes
