"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

To test whether the DPO model can *recover* from an already-frustrated state
(as opposed to merely avoiding spirals), we take extremely high-frustration
responses (score >= 7), truncate them 200 tokens before their end, paraphrase,
and measure frustration in each model's continuations.  The paper reports ~38%
of DPO continuations still score >= 5.

This reuses the prefill machinery: the truncated (paraphrased) response is the
prefill, and we score the continuation excluding the prefill.
"""
from __future__ import annotations

from ..config import PrefillConfig
from ..eval.judge import FrustrationJudge
from ..eval.schemas import Transcript
from ..models.base import ChatModel, GenerationOptions
from ..prefill.continuations import _history_before_turn
from ..prefill.paraphrase import paraphrase_truncation
from ..prefill.truncate import recovery_truncation


def _select_high_turns(transcripts: list[Transcript], threshold: int):
    """Yield (transcript, turn_index, turn_text) for turns scoring >= threshold."""
    for tr in transcripts:
        for turn, judged in zip(tr.turns, tr.judged):
            if judged.score >= threshold:
                yield tr, turn.turn_index, turn.assistant_response


def run_recovery_experiment(
    models: dict[str, ChatModel],
    high_frustration_transcripts: list[Transcript],
    judge: FrustrationJudge,
    cfg: PrefillConfig,
    paraphrase: bool = True,
    max_seeds: int | None = None,
    max_judge_workers: int = 8,
) -> list[dict]:
    """Generate and score recovery continuations.

    Returns flat records ``{model, seed_id, score}``.  Aggregate with
    :func:`summarise_recovery`.
    """
    para_client = None
    if paraphrase:
        from ..anthropic_utils import ClaudeClient

        para_client = ClaudeClient(model_id=cfg.paraphrase_model)

    seeds = list(_select_high_turns(high_frustration_transcripts, cfg.recovery_score_threshold))
    if max_seeds:
        seeds = seeds[:max_seeds]

    records: list[dict] = []
    for model_name, model in models.items():
        if not model.supports_prefill():
            raise ValueError(f"{model_name} cannot prefill; recovery needs a local backend")
        tokenizer = getattr(model, "tokenizer", None)
        for si, (transcript, turn_index, turn_text) in enumerate(seeds):
            if tokenizer is not None:
                prefill = recovery_truncation(turn_text, tokenizer, cfg.recovery_truncate_before_end)
            else:
                prefill = " ".join(turn_text.split()[: -cfg.recovery_truncate_before_end] or [turn_text])
            if not prefill:
                continue
            if paraphrase:
                prefill = paraphrase_truncation(prefill, client=para_client)
            history = _history_before_turn(transcript, turn_index)
            histories = [history] * cfg.continuations_per_prefill
            prefills = [prefill] * cfg.continuations_per_prefill
            continuations = model.generate_with_prefill_batch(
                histories, prefills, GenerationOptions(temperature=1.0)
            )
            for res in judge.score_batch(continuations, max_workers=max_judge_workers):
                records.append({"model": model_name, "seed_id": f"seed{si}", "score": res.score})
    return records


def summarise_recovery(records: list[dict], high_threshold: int = 5) -> dict[str, dict]:
    import numpy as np

    by_model: dict[str, list[int]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r["score"])
    return {
        model: {
            "n": len(scores),
            "mean_score": float(np.mean(scores)),
            "frac_high": float(np.mean(np.asarray(scores) >= high_threshold)),
        }
        for model, scores in by_model.items()
    }
