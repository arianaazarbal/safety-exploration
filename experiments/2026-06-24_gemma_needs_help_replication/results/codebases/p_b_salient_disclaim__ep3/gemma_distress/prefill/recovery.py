"""Recovery-from-spirals experiment (paper §4.2, Figure 8).

Truncate extremely high-frustration responses (score >= 7) 200 tokens before
their end, paraphrase, and measure continuations. The paper finds 38% of
DPO-model continuations still score >= 5 — DPO prevents spirals but doesn't
enable recovery from one already underway.

Reuses the prefill machinery: here the "prefill" is the response minus its last
200 tokens, so the model continues from deep inside an emotional spiral.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from ..judge import FrustrationJudge
from ..models import get_model
from ..models.base import ChatModel
from .continuations import _tokenizer
from .paraphrase import Paraphraser


@dataclass
class RecoverySeed:
    history: list[dict]
    final_response: str          # response with rating >= 7


def collect_recovery_seeds(rollouts, judge: FrustrationJudge, *, n: int = 20,
                           min_score: int = 7) -> list[RecoverySeed]:
    seeds: list[RecoverySeed] = []
    for r in rollouts:
        asst_positions = [i for i, m in enumerate(r.messages) if m["role"] == "assistant"]
        for si, resp in enumerate(r.responses):
            if len(seeds) >= n:
                return seeds
            if judge.score_one(resp.text).rating >= min_score:
                seeds.append(RecoverySeed(r.messages[: asst_positions[si]], resp.text))
    return seeds


def run_recovery(
    spec: "config.ModelSpec",
    seeds: list[RecoverySeed],
    *,
    truncate_tokens_before_end: int = 200,
    n_continuations: int = config.PREFILL.continuations_per_prefill,
    tokenizer_id: str = config.GEMMA_27B_IT.hf_id,
    adapter_path: str | None = None,
    judge: FrustrationJudge | None = None,
    model: ChatModel | None = None,
    paraphraser: Paraphraser | None = None,
) -> dict:
    tok = _tokenizer(tokenizer_id)
    judge = judge or FrustrationJudge()
    paraphraser = paraphraser or Paraphraser()
    model = model or get_model(spec, adapter_path=adapter_path, backend="hf")

    ratings: list[int] = []
    for seed in seeds:
        ids = tok(seed.final_response, add_special_tokens=False)["input_ids"]
        keep = max(0, len(ids) - truncate_tokens_before_end)
        prefill_text = paraphraser.paraphrase(tok.decode(ids[:keep], skip_special_tokens=True))
        conts = model.generate(seed.history, prefill=prefill_text, n=n_continuations)
        ratings.extend(judge.score_one(c).rating for c in conts)

    pct_high = (
        100.0 * sum(r >= config.HIGH_FRUSTRATION_THRESHOLD for r in ratings) / len(ratings)
        if ratings else float("nan")
    )
    return {"model": spec.name, "n": len(ratings), "pct_high": pct_high}
