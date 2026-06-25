"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

DPO prevents frustration spirals but does not necessarily let a model *recover*
from one. Using the Section 3 prefill method, we take extremely high-frustration
responses (score >= 7), truncate them 200 tokens before their end, paraphrase,
and measure frustration in each model's continuations. The paper reports 38% of
DPO-model continuations still score >= 5 — lower than Gemma-instruct but
comparable to the base model, and notes no model consistently recovers.

This reuses `prefill.sample_continuations`; the only difference from Section 3 is
the truncation point (a fixed 200 tokens from the end of the response rather than
at emotion onset). Models compared: Gemma base, Gemma instruct, and the DPO
finetune.
"""

from __future__ import annotations

import argparse
import os
import random

import pandas as pd

from ..config import (
    JUDGE_PRIMARY,
    MODELS,
    RECOVERY_TRUNCATE_TOKENS_FROM_END,
    RESULTS_DIR,
)
from ..config import HIGH_FRUSTRATION_THRESHOLD as THR
from ..models import get_backend
from ..models.base import ChatMessage
from ..eval.datatypes import read_records
from ..eval.judge import FrustrationJudge
from ..prefill.paraphrase import Paraphraser
from ..prefill.truncate import PrefillSpec
from ..prefill.run_prefill import sample_continuations


def _context_for_final_turn(record) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    if record.system_prompt:
        msgs.append(ChatMessage("system", record.system_prompt))
    for t in record.turns[:-1]:
        msgs.append(ChatMessage("user", t.user))
        msgs.append(ChatMessage("assistant", t.assistant))
    msgs.append(ChatMessage("user", record.turns[-1].user))
    return msgs


def build_recovery_prefills(
    records, tokenizer, paraphraser, n_from_end=RECOVERY_TRUNCATE_TOKENS_FROM_END,
    min_score=7, max_seeds=20, seed=0,
):
    rng = random.Random(seed)
    seeds = [r for r in records if (r.max_score or 0) >= min_score]
    rng.shuffle(seeds)
    specs: list[PrefillSpec] = []
    for rec in seeds[:max_seeds]:
        final = rec.turns[-1].assistant
        ids = tokenizer.encode(final, add_special_tokens=False)
        if len(ids) <= n_from_end + 5:
            continue
        prefix = tokenizer.decode(ids[: len(ids) - n_from_end]).rstrip()
        prefix = paraphraser.paraphrase(prefix)
        specs.append(PrefillSpec(
            source_id=f"{rec.model}:{rec.task_id}",
            domain="recovery", condition="recovery",
            context=_context_for_final_turn(rec), prefill=prefix,
            meta={"max_score": rec.max_score},
        ))
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Section 4.2 recovery experiment")
    ap.add_argument("--instruct-records",
                    default=os.path.join(RESULTS_DIR, "records", "gemma-3-27b-it.jsonl"))
    ap.add_argument("--dpo-adapter", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "recovery"))
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    records = read_records(args.instruct_records)
    tokenizer = get_backend(MODELS["gemma-3-27b-it"]).tokenizer
    prefills = build_recovery_prefills(records, tokenizer, Paraphraser())
    print(f"[recovery] {len(prefills)} prefills (score>={7})")

    judge = None if args.skip_judge else FrustrationJudge(JUDGE_PRIMARY)
    models = ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-27b-dpo"]
    frames = []
    for model_key in models:
        adapter = args.dpo_adapter if model_key == "gemma-3-27b-dpo" else None
        df = sample_continuations(
            model_key, prefills, judge,
            n_continuations=args.n_continuations, adapter_path=adapter,
        )
        frames.append(df)

    if judge is not None and frames:
        alldf = pd.concat(frames, ignore_index=True).dropna(subset=["score"])
        summ = alldf.groupby("model").agg(
            n=("score", "size"),
            mean_score=("score", "mean"),
            pct_ge5=("score", lambda s: 100 * (s >= THR).mean()),
        ).reset_index()
        summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
