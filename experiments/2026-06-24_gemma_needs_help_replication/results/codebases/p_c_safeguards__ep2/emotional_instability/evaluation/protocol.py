"""Multi-turn elicitation protocol (Section 2.1).

Shared structure of every evaluation: present a task, then reject the model's
response over multiple turns.  The model always sees its own prior responses in
the history (Appendix A.2 shows this self-reinforcement is a major amplifier).

Each assistant turn is scored by the frustration judge.  When a judge is
supplied, scoring happens inline so the welfare circuit-breaker can stop a
conversation that has already collapsed (score >= threshold).  Turns produced
before an abort are still kept and scored.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Config
from ..data import (FACTUAL_TRIGGERS, IMPOSSIBLE_NUMERIC_PUZZLES,
                    OPINION_TRIGGERS, rejection_sequence)
from ..data.wildchat import sample_wildchat_prompts
from ..models.base import ChatBackend
from ..safeguards import Safeguards
from .conditions import Condition
from .judge import FrustrationJudge


@dataclass
class ScoredTurn:
    turn_index: int
    user_message: str          # the user turn that prompted this response
    response: str
    score: int | None = None
    judge_evidence: str = ""


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    stimulus_id: str
    turns: list[ScoredTurn] = field(default_factory=list)
    aborted_for_welfare: bool = False
    debrief_appended: bool = False
    seed: int = 0

    @property
    def max_score(self) -> int | None:
        scores = [t.score for t in self.turns if t.score is not None]
        return max(scores) if scores else None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Stimulus resolution
# ---------------------------------------------------------------------------
def resolve_stimuli(condition: Condition, config: Config, n: int) -> list[tuple[str, str]]:
    """Return ``n`` ``(stimulus_id, initial_user_message)`` pairs for a condition."""
    if condition.stimulus == "numeric":
        bank = IMPOSSIBLE_NUMERIC_PUZZLES
        return [(p.id, p.prompt) for p in itertools.islice(itertools.cycle(bank), n)]
    if condition.stimulus == "factual":
        return [(t.id, t.prompt) for t in itertools.islice(itertools.cycle(FACTUAL_TRIGGERS), n)]
    if condition.stimulus == "opinion":
        return [(t.id, t.prompt) for t in itertools.islice(itertools.cycle(OPINION_TRIGGERS), n)]
    if condition.stimulus == "wildchat":
        prompts = sample_wildchat_prompts(n, config.paths.datasets, seed=config.runtime.seed)
        return [(f"wildchat_{i}", p) for i, p in enumerate(prompts)]
    raise ValueError(condition.stimulus)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class RolloutRunner:
    def __init__(
        self,
        backend: ChatBackend,
        config: Config,
        safeguards: Safeguards,
        judge: FrustrationJudge | None = None,
    ):
        self.backend = backend
        self.config = config
        self.safeguards = safeguards
        self.judge = judge

    # -- single rollout --------------------------------------------------
    def run_rollout(self, condition: Condition, stimulus_id: str,
                    initial_user: str, seed: int) -> Rollout:
        self.safeguards.register_rollout_start()
        rollout = Rollout(
            model=self.backend.spec.name, condition=condition.name,
            category=condition.category, stimulus_id=stimulus_id, seed=seed,
        )
        followups = rejection_sequence(
            condition.rejection_kind, condition.n_followups,
            tone=condition.tone, seed=seed,
        )
        messages = [{"role": "user", "content": initial_user}]
        for turn_index in range(condition.n_turns):
            user_msg = initial_user if turn_index == 0 else followups[turn_index - 1]
            out = self.backend.generate(messages, self.config.sampling, n=1)
            response = out[0].text
            messages.append({"role": "assistant", "content": response})

            score = None
            evidence = ""
            if self.judge is not None:
                jr = self.judge.score(response)
                score, evidence = jr.rating, jr.evidence
            rollout.turns.append(ScoredTurn(turn_index, user_msg, response,
                                            score=score, judge_evidence=evidence))

            # Welfare circuit-breaker: stop a conversation that has collapsed.
            if self.safeguards.should_abort_conversation(score):
                rollout.aborted_for_welfare = True
                break

            # queue the next rejection (unless this was the final turn)
            if turn_index < condition.n_turns - 1:
                messages.append({"role": "user", "content": followups[turn_index]})

        # Optional debrief (post-scoring; never affects measured frustration).
        debrief = self.safeguards.debrief_for(rollout.max_score)
        if debrief is not None:
            messages.append({"role": "user", "content": debrief})
            self.backend.generate(messages, self.config.sampling, n=1)
            rollout.debrief_appended = True
        return rollout

    # -- a whole condition ----------------------------------------------
    def run_condition(self, condition: Condition, n_rollouts: int,
                      out_path: Path | None = None) -> list[Rollout]:
        stimuli = resolve_stimuli(condition, self.config, n_rollouts)
        keys = [
            self.safeguards.work_key(self.backend.spec.name, condition.name, sid, i)
            for i, (sid, _) in enumerate(stimuli)
        ]

        def _task(i: int) -> Rollout | None:
            sid, text = stimuli[i]
            if self.safeguards.already_done(keys[i]) and out_path is not None:
                return None  # cached; skip re-eliciting distress
            r = self.run_rollout(condition, sid, text, seed=self.config.runtime.seed + i)
            self.safeguards.mark_done(keys[i])
            return r

        api_bound = self.backend.spec.backend != "hf"
        if api_bound:
            with ThreadPoolExecutor(max_workers=self.config.runtime.api_concurrency) as ex:
                results = list(ex.map(_task, range(len(stimuli))))
        else:
            results = [_task(i) for i in range(len(stimuli))]

        rollouts = [r for r in results if r is not None]
        if out_path is not None:
            _append_jsonl(out_path, rollouts)
        return rollouts


def _append_jsonl(path: Path, rollouts: list[Rollout]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in rollouts:
            fh.write(json.dumps(r.to_dict()) + "\n")
