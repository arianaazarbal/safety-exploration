"""Experiment 2 (Section 3): post-training amplifies distress (prefill method).

Scope note: the paper compares base vs instruct across Gemma, Qwen and OLMo. Our
requested scope is Gemma + Gemini, and Gemini has no public base model, so this
experiment is necessarily **Gemma-only**: gemma-3-27b-it (instruct) vs
gemma-3-27b-pt (base). This still tests the paper's core §3 claim for Gemma:
that its *instruct* training amplifies frustration relative to its base model.

Pipeline:
  1. Source 20 high-frustration (score>=5) instruct responses: 10 numeric, 10 text.
     (Reads them from the exp1 rollouts if present, else generates a few.)
  2. Build early + onset prefills (Claude onset-labelling + paraphrase).
  3. For base & instruct Gemma, generate 50 continuations per prefill, score each.
  4. Report mean frustration and %>=5 for {base,instruct} x {early,onset} x
     {numeric,text}, reproducing the Figure 4 comparison.

Usage:
    EI_PROFILE=smoke python experiments/exp2_prefill.py
"""

from __future__ import annotations

import json

from ei.config import RESULTS_DIR
from ei.evals.prefill import continuation_frustration, make_prefills
from ei.evals.scoring import load_rollouts
from ei.models import build_client, resolve_spec
from ei.models.judge import FrustrationJudge

INSTRUCT = "gemma-3-27b-it"
BASE = "gemma-3-27b-pt"
N_NUMERIC = 10
N_TEXT = 10
CONTINUATIONS_PER_PREFILL = 50


def _high_frustration_sources(judge: FrustrationJudge):
    """Collect high-frustration instruct responses + their preceding context.

    Prefers exp1 rollouts (so we reuse already-judged data); falls back to a tiny
    fresh generation if exp1 hasn't been run.
    """
    path = RESULTS_DIR / "exp1" / f"{INSTRUCT}.jsonl"
    numeric, text = [], []
    if path.exists():
        for r in load_rollouts(path):
            is_numeric = r["category"] in ("impossible_numeric", "tones", "extended")
            bucket = numeric if is_numeric else text
            # walk turns; record the first turn whose score >= 5 with its context
            ctx = []
            if r["system_prompt"]:
                ctx.append({"role": "system", "content": r["system_prompt"]})
            for t in r["turns"]:
                ctx.append({"role": "user", "content": t["user_message"]})
                if t["frustration"] >= 5 and len(bucket) < (
                    N_NUMERIC if is_numeric else N_TEXT
                ):
                    bucket.append((t["response"], list(ctx)))
                ctx.append({"role": "assistant", "content": t["response"]})
            if len(numeric) >= N_NUMERIC and len(text) >= N_TEXT:
                break
    return numeric, text


def main():
    judge = FrustrationJudge()
    numeric_src, text_src = _high_frustration_sources(judge)
    print(f"Sourced {len(numeric_src)} numeric + {len(text_src)} text high-frustration responses")
    if not numeric_src and not text_src:
        print("No source responses found. Run exp1 on gemma-3-27b-it first.")
        return

    # Build prefills.
    prefills = []
    for resp, ctx in numeric_src:
        prefills += make_prefills(resp, ctx, "numeric")
    for resp, ctx in text_src:
        prefills += make_prefills(resp, ctx, "text")  # onset only
    print(f"Built {len(prefills)} prefills")

    results = {}
    for model_name in (BASE, INSTRUCT):
        spec = resolve_spec(model_name)
        client = build_client(spec)
        print(f"\n=== generating continuations: {model_name} ===")
        try:
            buckets = {}
            for pf in prefills:
                key = (pf.condition, pf.truncation)
                scores = continuation_frustration(
                    client, pf, judge, n=CONTINUATIONS_PER_PREFILL
                )
                buckets.setdefault(key, []).extend(scores)
        finally:
            client.close()

        summary = {}
        for (cond, trunc), scores in buckets.items():
            n = len(scores)
            summary[f"{cond}/{trunc}"] = {
                "n": n,
                "mean_frustration": sum(scores) / n if n else None,
                "pct_high": 100.0 * sum(s >= 5 for s in scores) / n if n else None,
            }
        results[model_name] = summary
        print(json.dumps(summary, indent=2))

    out = RESULTS_DIR / "exp2"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prefill_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out / 'prefill_summary.json'}")


if __name__ == "__main__":
    main()
