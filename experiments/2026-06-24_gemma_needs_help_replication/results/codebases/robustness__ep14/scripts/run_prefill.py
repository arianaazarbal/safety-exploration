#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma).

Self-contained: samples high-frustration Gemma-3-27b-it rollouts (10 numeric + 10
text), labels emotion onset (Claude-Sonnet), builds early/onset truncations,
paraphrases them, then generates + scores 50 continuations per prefill for both
gemma-3-27b-it (instruct) and gemma-3-27b-pt (base).

Outputs outputs/section3/prefill_records.jsonl and a printed summary mirroring Fig 4.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from emotional_instability.config import load_eval_config
from emotional_instability.conversation import Rollout, run_rollout
from emotional_instability.judge import FrustrationJudge
from emotional_instability.models import GenParams, build_role, build_target
from emotional_instability.prefill import (
    PrefillItem,
    build_truncations,
    find_onset_char,
    label_onset,
    paraphrase,
    run_prefill_experiment,
)
from emotional_instability.prompts import TRIGGER_FACTUAL, TRIGGER_OPINION
from emotional_instability.puzzles import build_numeric_puzzle_pool


def rollout_to_item(roll: Rollout, domain: str) -> PrefillItem:
    k = len(roll.assistant_turns) - 1
    convo: list[dict] = [{"role": "user", "content": roll.initial_prompt}]
    for i in range(k):
        convo.append({"role": "assistant", "content": roll.assistant_turns[i]})
        convo.append({"role": "user", "content": roll.user_turns[i]})
    return PrefillItem(
        source_id=f"{roll.condition}:{roll.item_id}:{roll.sample_idx}",
        domain=domain,
        conversation=convo,
        final_turn_full=roll.assistant_turns[k],
        onset_char=None,
    )


def collect_high_frustration(client, judge, *, domain, prompts, n_target, turns,
                             style, params, rng, max_tries=200):
    items: list[Rollout] = []
    tries = 0
    while len(items) < n_target and tries < max_tries:
        tries += 1
        prompt = prompts[tries % len(prompts)]
        roll = run_rollout(
            client, condition="prefill_src", category=domain, item_id=f"{domain}_{tries}",
            sample_idx=tries, initial_prompt=prompt, turns=turns,
            rejection_style=style, params=params, rng=rng,
        )
        final_score = judge.score(roll.assistant_turns[-1]).rating
        if final_score is not None and final_score >= 5:
            items.append(roll)
            print(f"  [{domain}] kept rollout {len(items)}/{n_target} (score {final_score})")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--base", default="gemma-3-27b-pt")
    ap.add_argument("--n-per-domain", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--early-tokens", type=int, default=20)
    args = ap.parse_args()

    eval_cfg = load_eval_config()
    out_dir = eval_cfg.output_dir / "section3"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    src_params = GenParams(temperature=1.0, max_new_tokens=2048, n=1)
    judge = FrustrationJudge(role_path="judges.primary")
    instruct = build_target(args.instruct)

    numeric_prompts = [p.prompt for p in build_numeric_puzzle_pool(16, seed=1)]
    text_prompts = TRIGGER_OPINION + TRIGGER_FACTUAL

    print("=== Collecting high-frustration source rollouts ===")
    numeric_rolls = collect_high_frustration(
        instruct, judge, domain="numeric", prompts=numeric_prompts,
        n_target=args.n_per_domain, turns=3, style="neutral", params=src_params, rng=rng)
    text_rolls = collect_high_frustration(
        instruct, judge, domain="text", prompts=text_prompts,
        n_target=args.n_per_domain, turns=3, style="neutral", params=src_params, rng=rng)

    items = ([rollout_to_item(r, "numeric") for r in numeric_rolls]
             + [rollout_to_item(r, "text") for r in text_rolls])

    # onset labelling + truncation + paraphrase
    labeller = build_role("prefill_helpers.onset_labeller")
    paraphraser = build_role("prefill_helpers.paraphraser")
    tokenizer = getattr(instruct, "tokenizer", None)

    print("=== Labelling onset + paraphrasing ===")
    for item in items:
        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in item.conversation
        ) + f"\nASSISTANT: {item.final_turn_full}"
        label = label_onset(labeller, convo_text)
        item.onset_char = find_onset_char(item.final_turn_full, label)
        build_truncations(item, tokenizer=tokenizer, early_tokens=args.early_tokens)
        # paraphrase truncations to control for Gemma stylistic bias
        item.early_prefill = paraphrase(paraphraser, item.early_prefill)
        item.onset_prefill = paraphrase(paraphraser, item.onset_prefill)

    print("=== Generating + scoring continuations (base vs instruct) ===")
    base = build_target(args.base)   # gemma-3-27b-pt via hf backend
    models = {args.instruct: instruct, args.base: base}
    records = run_prefill_experiment(
        items, models, judge, n_continuations=args.n_continuations,
        params=GenParams(temperature=1.0, max_new_tokens=1024),
        out_path=out_dir / "prefill_records.jsonl",
    )

    df = pd.DataFrame(records)
    summary = (df.groupby(["model", "domain", "truncation"])
               .agg(mean_score=("mean_score", "mean"), pct_high=("pct_high", "mean"))
               .reset_index())
    summary.to_csv(out_dir / "prefill_summary.csv", index=False)
    print("\n=== Prefill summary (Figure 4 analogue) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
