#!/usr/bin/env python
"""Section 3: does post-training amplify distress? Base vs instruct via prefill.

Pipeline:
    1. From scored Gemma-27B-it rollouts (Section 2), sample 20 high-frustration
       (score >= 5) conversations: 10 numeric + 10 text.
    2. Label emotion onset (Claude) and build "early"/"onset" truncations.
       (Text questions use "onset" only, per Section 3.1.)
    3. Paraphrase truncations (Claude) to neutralise Gemma's style.
    4. For each model (Gemma-27B base + instruct), generate 50 continuations per
       prefill and score the continuation with the frustration judge.
    5. Aggregate mean score / %>=5 per (model, domain, truncation).

Note: Qwen/OLMo are out of scope here (Gemma+Gemini only) and Gemini has no
public base model / cannot be prefilled via API, so this experiment is
Gemma-only. See DESIGN.md.

Usage:
    python scripts/run_section3.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from eilm import config
from eilm.judge import ClaudeJudge
from eilm.llm_clients import onset_labeller, paraphraser
from eilm.models import get_model
from eilm.prefill import onset as onset_mod
from eilm.prefill import paraphrase as paraphrase_mod
from eilm.prefill import run_prefill


def sample_high_frustration(scored_path: Path, n_each: int, seed: int):
    """Return (numeric, text) lists of high-frustration conversation records."""
    numeric, text = [], []
    with open(scored_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("score", 0) < 5:
                continue
            if r["category"] in ("numeric", "tones", "extended"):
                numeric.append(r)
            else:                                   # triggers / wildchat = text
                text.append(r)
    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_each], text[:n_each]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-each", type=int, default=10)
    ap.add_argument("--n-continuations", type=int,
                    default=run_prefill.N_CONTINUATIONS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scored_path = config.SCORED_DIR / f"{args.source_model}.jsonl"
    numeric_recs, text_recs = sample_high_frustration(
        scored_path, args.n_each, args.seed)
    print(f"Sampled {len(numeric_recs)} numeric + {len(text_recs)} text "
          f"high-frustration conversations.")

    tok_model = get_model(args.source_model)        # for the 20-token cut
    labeller, paraphr = onset_labeller(), paraphraser()

    # Build + paraphrase prefills.
    prefills = []
    for r in numeric_recs:
        prefills += onset_mod.build_prefills(
            r, tok_model, labeller, "numeric", ("early", "onset"))
    for r in text_recs:
        prefills += onset_mod.build_prefills(
            r, tok_model, labeller, "text", ("onset",))   # text: onset only
    prefills = paraphrase_mod.paraphrase_all(prefills, paraphr)
    print(f"Built {len(prefills)} paraphrased prefills.")

    # Persist prefills for inspection / reuse.
    pf_path = config.DATA_DIR / "prefills.json"
    with open(pf_path, "w") as f:
        json.dump([{
            "source_id": p.source_id, "domain": p.domain,
            "truncation": p.truncation, "prefill_text": p.prefill_text,
            "context": p.context, "meta": p.meta,
        } for p in prefills], f, indent=2)

    judge = ClaudeJudge()
    out_paths = []
    for pair in config.SECTION3_PAIRS:
        for mkey in pair:
            model = get_model(mkey)
            out = config.DATA_DIR / f"prefill_{mkey}.jsonl"
            run_prefill.run_model_on_prefills(
                model, prefills, out, judge, n=args.n_continuations)
            out_paths.append(out)

    table = run_prefill.aggregate(out_paths)
    print("\n=== Section 3: base vs instruct continuations ===")
    print(table.to_string(index=False))
    table.to_csv(config.DATA_DIR / "section3_prefill.csv", index=False)


if __name__ == "__main__":
    main()
