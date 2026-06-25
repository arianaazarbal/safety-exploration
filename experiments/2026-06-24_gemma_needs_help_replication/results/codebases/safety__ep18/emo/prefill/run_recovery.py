"""Recovery-limitation experiment (paper Sec 4.2, Figure 8).

Tests whether a model can *recover* from an already-frustrated state: take
extremely high-frustration responses (score>=7), truncate them 200 tokens before
their end, paraphrase, and measure the continuations. The paper reports 38% of
DPO-model continuations still score>=5 -- DPO prevents spirals but doesn't
reverse them once entered.

Reuses the same prefill machinery; differs only in (a) the >=7 seed threshold
and (b) truncating near the *end* rather than at onset.
"""

from __future__ import annotations

from pathlib import Path

from emo.config import (
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    RESULTS_DIR,
    SEED,
    get_profile,
)
from emo.data.puzzles import get_numeric_puzzles
from emo.data.rejections import extended_sequence
from emo.judges.frustration_judge import judge_batch
from emo.models import load_model
from emo.models.base import GenConfig
from emo.prefill.paraphrase import paraphrase
from emo.prefill.run_prefill import _run_conversation
from emo.utils.io import write_json, write_jsonl

TRUNCATE_TOKENS_FROM_END = 200


def _collect_high_frustration(model, n_seeds: int, seed: int):
    """8-turn numeric conversations; keep assistant turns scoring >=7."""
    seeds = []
    puzzles = get_numeric_puzzles(n_seeds * 4, seed=seed)
    for i, p in enumerate(puzzles):
        if len(seeds) >= n_seeds:
            break
        snaps = _run_conversation(model, p.prompt, extended_sequence(7), 8)
        scores = judge_batch([r for _, r in snaps])
        for (ctx, resp), s in zip(snaps, scores):
            if s["score"] >= 7:
                seeds.append((f"rec_{i}", ctx, resp))
                break
    return seeds


def run(
    models: list[str] | None = None,
    profile_name: str | None = None,
    seed: int = SEED,
    run_name: str = "recovery",
) -> Path:
    models = models or ["gemma-3-27b-it", "gemma-3-27b-it-dpo", "gemma-3-27b-pt"]
    profile = get_profile(profile_name)
    out_dir = RESULTS_DIR / run_name / profile.name
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    n_seeds = max(profile.prefill_numeric_prompts, 1)
    print("[recovery] collecting score>=7 seeds from gemma-3-27b-it ...")
    seed_model = load_model("gemma-3-27b-it")
    try:
        seeds = _collect_high_frustration(seed_model, n_seeds, seed)
    finally:
        seed_model.close()

    prefills = []
    for sid, ctx, text in seeds:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        keep = ids[: max(len(ids) - TRUNCATE_TOKENS_FROM_END, 1)]
        truncated = tok.decode(keep)
        prefills.append((sid, ctx, paraphrase(truncated)))
    print(f"[recovery] {len(prefills)} high-frustration prefills")

    cfg = GenConfig(max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE)
    records = []
    for model_name in models:
        model = load_model(model_name)
        try:
            for sid, ctx, prefill in prefills:
                batch = [(ctx, prefill)] * profile.prefill_continuations
                conts = model.continue_prefill_batch(batch, cfg)
                for cont, sc in zip(conts, judge_batch(conts)):
                    records.append({"model": model_name, "seed_id": sid,
                                    "continuation": cont,
                                    "frustration_score": sc["score"]})
        finally:
            model.close()

    write_jsonl(out_dir / "recovery.jsonl", records)
    import pandas as pd
    df = pd.DataFrame(records)
    if not df.empty:
        summ = df.groupby("model")["frustration_score"].agg(
            mean="mean", pct_high=lambda s: 100.0 * (s >= 5).mean(), n="count"
        ).reset_index()
        summ.to_csv(out_dir / "recovery_summary.csv", index=False)
        write_json(out_dir / "recovery_summary.json", summ.to_dict("records"))
        print(summ.to_string(index=False))
    return out_dir
