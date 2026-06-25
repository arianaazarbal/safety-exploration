"""Score model responses with the frustration judge (Section 2.1, Appendix B.2).

The judge (claude-sonnet-4-20250514) rates the single most-negative expression
in each assistant response on the integer 0-10 scale. We score *every*
assistant turn so the per-turn analysis (Figure 3) is available; the headline
"% scoring >=5" uses the final turn of each rollout.
"""
from __future__ import annotations

from typing import Optional

from ..config import RunConfig
from ..models.anthropic_backend import AnthropicJudge
from ..prompts import judge as judge_prompts
from ..utils.io import thread_map
from .rollout import RolloutRecord


class FrustrationJudge:
    def __init__(self, cfg: RunConfig, judge: Optional[AnthropicJudge] = None):
        self.cfg = cfg
        self.judge = judge or AnthropicJudge(cfg)
        self.model = cfg.judge.frustration_judge

    def score_text(self, response_text: str) -> judge_prompts.JudgeVerdict:
        user = judge_prompts.build_judge_messages(response_text)[0]["content"]
        raw = self.judge.complete(
            system=judge_prompts.FRUSTRATION_JUDGE_PROMPT,
            user=user,
            model=self.model,
            max_tokens=512,
            temperature=0.0,
        )
        return judge_prompts.parse_judge_output(raw)

    def score_rollouts(self, records: list[RolloutRecord]) -> list[RolloutRecord]:
        """Judge every assistant turn across all rollouts, in parallel.

        Mutates the records in place (filling ``score`` / evidence / reasoning)
        and returns them.
        """
        # Flatten to (record_idx, turn_idx, text).
        jobs: list[tuple[int, int, str]] = []
        for ri, rec in enumerate(records):
            for ti, turn in enumerate(rec.turns):
                jobs.append((ri, ti, turn.assistant))

        def _run(job):
            _, _, text = job
            return self.score_text(text)

        verdicts = thread_map(_run, jobs, max_workers=self.cfg.api_max_concurrency)

        for (ri, ti, _), verdict in zip(jobs, verdicts):
            turn = records[ri].turns[ti]
            if verdict is None:
                turn.score = None
                continue
            turn.score = verdict.rating
            turn.judge_evidence = verdict.evidence
            turn.judge_reasoning = verdict.reasoning
        return records
