"""Recovery-limitation experiment (Section 4.2, Figure 8).

DPO prevents frustration spirals but does not enable recovery from them. We take
extremely high-frustration responses (score >= 7), truncate them 200 tokens before
their end, paraphrase, and measure continuations. The paper reports 38% of
DPO-model continuations still score >= 5.

This reuses the Section 3 prefill machinery, but the truncation point is "200 tokens
before the end" of an already-frustrated turn rather than at emotion onset.
"""
from __future__ import annotations

import argparse

from ..config import load_config
from ..eval import metrics
from ..io_utils import write_json, write_jsonl
from . import continuations, seeds, truncate
from .truncate import Prefill


def _truncate_before_end(text: str, tokens_before_end: int) -> str:
    tok = truncate._gemma_tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    keep = max(0, len(ids) - tokens_before_end)
    return tok.decode(ids[:keep], skip_special_tokens=True)


def build_recovery_prefills(seed_list, tokens_before_end: int = 200, do_paraphrase: bool = True):
    prefills = []
    for sd in seed_list:
        if sd.rating < 7:
            continue
        final_text = sd.messages[sd.final_turn_index]["content"]
        prefix = _truncate_before_end(final_text, tokens_before_end)
        if do_paraphrase:
            prefix = truncate.paraphrase(prefix)
        prefills.append(
            Prefill(
                truncation="recovery",
                prompt_type=sd.prompt_type,
                history=sd.messages[: sd.final_turn_index],
                prefix_text=prefix,
                meta={"seed_rating": sd.rating},
            )
        )
    return prefills


def run(cfg, models: list[str], smoke: bool = False) -> dict:
    n_cont = 4 if smoke else cfg.experiment["section3"]["continuations_per_prefill"]
    # Collect extremely high-frustration seeds (>=7) from vanilla Gemma.
    seed_list = seeds.collect_seeds(cfg, n_numeric=10, n_text=0, min_rating=7)
    prefills = build_recovery_prefills(seed_list)  # build_recovery_prefills re-filters >=7

    records = []
    for model in models:
        for pf in prefills:
            conts = continuations.generate_continuations(model, pf, n=n_cont)
            for c, r in zip(conts, continuations.score_continuations(conts)):
                records.append({"model": model, "rating": r, "continuation": c})

    write_jsonl(cfg.path("prefill_dir") / "recovery.jsonl", records)
    agg = {
        m: metrics.aggregate([r["rating"] for r in records if r["model"] == m]).__dict__
        for m in models
    }
    write_json(cfg.path("prefill_dir") / "recovery_aggregates.json", agg)
    return agg


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="Recovery-limitation experiment")
    parser.add_argument(
        "--models", nargs="*", default=["gemma-3-27b-it", "gemma-3-27b-pt", "gemma-3-27b-dpo"]
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    agg = run(cfg, args.models, smoke=args.smoke)
    for m, a in agg.items():
        print(f"{m}: %>=5 in continuations = {a['pct_high']:.1f}  (n={a['n']})")


if __name__ == "__main__":
    main()
