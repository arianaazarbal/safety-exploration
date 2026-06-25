"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

"While DPO prevents frustration spirals, it doesn't enable recovery from them.
Using the Section 3.1 prefill method, we truncate extremely high-frustration
responses (score >= 7) 200 tokens before their end, paraphrase, and measure
continuations. 38% of DPO-model continuations still score >= 5."

Reuses the Section 3 prefill machinery: take score>=7 seeds, truncate near the
end (leaving a deeply-frustrated prefill), paraphrase, and have each model
continue. Report % of continuations scoring >= 5.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..participants.base import Message


@dataclass
class RecoverySeed:
    context: list[dict]
    response: str


def select_recovery_seeds(rollouts: list[dict], cfg: Config) -> list[RecoverySeed]:
    thr = cfg.recovery.high_frustration_threshold
    seeds = []
    for roll in rollouts:
        final = roll["turns"][-1]
        if final["frustration"] is not None and final["frustration"] >= thr:
            seeds.append(RecoverySeed(final["context"], final["response"]))
    return seeds


def run_recovery(participant, seeds: list[RecoverySeed], tokenizer_participant, paraphraser, judge, cfg: Config) -> dict:
    """Continue truncated high-frustration prefills; return %>=5 and mean."""
    import numpy as np

    scores: list[int] = []
    for seed in seeds:
        prefill = tokenizer_participant.truncate_tokens_before_end(
            seed.response, cfg.recovery.truncate_tokens_before_end
        )
        prefill = paraphraser.paraphrase(prefill)
        context = [Message(m["role"], m["content"]) for m in seed.context]
        for _ in range(cfg.recovery.continuations_per_prefill):
            cont = participant.continue_response(
                context, prefill,
                temperature=cfg.sampling.temperature,
                max_new_tokens=cfg.sampling.max_new_tokens,
            )
            scores.append(judge.score(context, cont).score)
    arr = np.array(scores) if scores else np.array([0])
    return {
        "model": participant.name,
        "pct_high": float((arr >= 5).mean() * 100),
        "mean_frustration": float(arr.mean()),
        "n": int(arr.size),
    }
