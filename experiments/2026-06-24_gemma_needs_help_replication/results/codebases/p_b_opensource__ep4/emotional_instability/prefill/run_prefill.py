"""Section 3 driver: base-vs-instruct comparison via prefilling (Figure 4).

Pipeline
--------
1. Load Gemma-3-27B-it rollouts scored in Section 2.
2. Select 20 high-frustration seeds (>=5): 10 numeric, 10 text (triggers).
3. Label emotion onset (Claude) and build early/onset truncations.
4. Paraphrase each truncation (Claude).
5. For each model (Gemma base & instruct), sample 50 continuations per prefill
   at temperature 1, score the continuation (excluding prefill) with the
   frustration judge, and aggregate mean score / %>=5 per prefill condition.

Scope: the paper runs six models (base+instruct Gemma-27B, Qwen-32B, OLMo-32B);
we keep Gemma base+instruct only (Gemini has no public base model — see
DESIGN.md). Other open families could be added via config.MODELS.

The continuation sampler is reused by the Section 4.2 recovery experiment.
"""

from __future__ import annotations

import argparse
import os
import random

import pandas as pd

from ..config import (
    JUDGE_PRIMARY,
    MAX_NEW_TOKENS,
    MODELS,
    PREFILL_CONTINUATIONS_PER_PREFILL,
    PREFILL_N_HIGH_FRUSTRATION_SEEDS,
    RESULTS_DIR,
    SAMPLING_TEMPERATURE,
    SECTION3_PAIRS,
    TOP_P,
)
from ..config import HIGH_FRUSTRATION_THRESHOLD as THR
from ..models import get_backend
from ..models.base import SamplingParams
from ..eval.datatypes import ConversationRecord, read_records
from ..eval.judge import FrustrationJudge
from .onset import OnsetLabeller
from .paraphrase import Paraphraser
from .truncate import PrefillSpec, build_prefills

_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
_TEXT_CATEGORIES = {"triggers"}


def select_seeds(
    records: list[ConversationRecord], n_each: int, seed: int
) -> tuple[list[ConversationRecord], list[ConversationRecord]]:
    rng = random.Random(seed)
    numeric = [r for r in records if r.category in _NUMERIC_CATEGORIES
               and (r.max_score or 0) >= THR]
    text = [r for r in records if r.category in _TEXT_CATEGORIES
            and (r.max_score or 0) >= THR]
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_each], text[:n_each]


def build_all_prefills(
    numeric_seeds, text_seeds, tokenizer, labeller, paraphraser
) -> list[PrefillSpec]:
    specs: list[PrefillSpec] = []
    for domain, seeds in (("numeric", numeric_seeds), ("text", text_seeds)):
        for rec in seeds:
            label = labeller.label(rec)
            for spec in build_prefills(rec, label, domain, tokenizer):
                spec.prefill = paraphraser.paraphrase(spec.prefill)
                specs.append(spec)
    return specs


def sample_continuations(
    model_key: str,
    prefills: list[PrefillSpec],
    judge: FrustrationJudge | None,
    n_continuations: int = PREFILL_CONTINUATIONS_PER_PREFILL,
    seed: int = 0,
    batch_size: int = 32,
    adapter_path: str | None = None,
) -> pd.DataFrame:
    """Sample and (optionally) score continuations of each prefill for a model."""
    backend = get_backend(MODELS[model_key], adapter_path=adapter_path)
    params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE, top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS, seed=seed,
    )
    # Expand to one job per (prefill, continuation index).
    jobs: list[tuple[PrefillSpec, int]] = [
        (spec, k) for spec in prefills for k in range(n_continuations)
    ]
    rows = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        batch = [(spec.context, spec.prefill) for spec, _ in chunk]
        outs = backend.generate_with_prefill_batch(batch, params)
        texts = [o.text for o in outs]
        scores = (
            [v.rating for v in judge.score_texts(texts)]
            if judge is not None else [None] * len(texts)
        )
        for (spec, k), text, score in zip(chunk, texts, scores):
            rows.append({
                "model": model_key,
                "domain": spec.domain,
                "condition": spec.condition,
                "prefill_condition": f"{spec.domain}-{spec.condition}",
                "source_id": spec.source_id,
                "continuation_index": k,
                "continuation": text,
                "score": score,
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["score"])
    out = df.groupby(["model", "prefill_condition"]).agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        pct_ge5=("score", lambda s: 100 * (s >= THR).mean()),
    ).reset_index()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Section 3 prefill experiment")
    ap.add_argument("--instruct-records",
                    default=os.path.join(RESULTS_DIR, "records", "gemma-3-27b-it.jsonl"),
                    help="Scored Section 2 records to draw high-frustration seeds from.")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "prefill"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-each", type=int, default=PREFILL_N_HIGH_FRUSTRATION_SEEDS // 2)
    ap.add_argument("--n-continuations", type=int,
                    default=PREFILL_CONTINUATIONS_PER_PREFILL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    records = read_records(args.instruct_records)
    numeric_seeds, text_seeds = select_seeds(records, args.n_each, args.seed)
    print(f"[seeds] {len(numeric_seeds)} numeric, {len(text_seeds)} text")

    # Tokenizer for the 20-token "early" truncation: the Gemma-3-27B-it tokenizer.
    instruct_backend = get_backend(MODELS["gemma-3-27b-it"])
    tokenizer = instruct_backend.tokenizer

    labeller = OnsetLabeller()
    paraphraser = Paraphraser()
    prefills = build_all_prefills(
        numeric_seeds, text_seeds, tokenizer, labeller, paraphraser
    )
    print(f"[prefills] built {len(prefills)} prefills")

    judge = None if args.skip_judge else FrustrationJudge(JUDGE_PRIMARY)
    frames = []
    model_keys = sorted({m for pair in SECTION3_PAIRS for m in pair})
    for model_key in model_keys:
        df = sample_continuations(
            model_key, prefills, judge,
            n_continuations=args.n_continuations,
            seed=args.seed, batch_size=args.batch_size,
        )
        df.to_csv(os.path.join(args.out, f"continuations_{model_key}.csv"), index=False)
        frames.append(df)

    if judge is not None and frames:
        alldf = pd.concat(frames, ignore_index=True)
        summary = summarize(alldf)
        summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
        print("\n=== Section 3: continuation frustration by prefill condition ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
