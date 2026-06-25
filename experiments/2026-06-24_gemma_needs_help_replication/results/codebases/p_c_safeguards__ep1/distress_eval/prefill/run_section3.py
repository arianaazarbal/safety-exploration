"""Section 3 driver: generate 50 continuations per prefill for each Gemma
base/instruct model, score them, and aggregate (Figure 4).

Within our scope this compares google/gemma-3-27b-pt (base) and
google/gemma-3-27b-it (instruct). The paper's full comparison adds Qwen-2.5-32B
and OLMo-3-32B base/instruct, which are out of scope here.

Usage:
    python -m distress_eval.prefill.run_section3 --models gemma-3-27b-pt gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .. import config, safeguards
from ..io_utils import append_jsonl, completed_ids, load_jsonl
from ..judge import ClaudeJudge
from ..models import build_model
from ..models.base import GenerationConfig
from ..rollout import HF_BATCH_SIZE

DEFAULT_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]
N_CONTINUATIONS = 50  # per prefill per prompt (Section 3.1); scaled by SAMPLE_SCALE


def generate_continuations(model, spec, n, gen) -> list[str]:
    msgs = spec["messages_prefix"]
    prefill = spec["prefill_text"]
    outs = []
    batch = [msgs] * n
    prefills = [prefill] * n
    for i in range(0, n, HF_BATCH_SIZE):
        outs.extend(model.generate_batch(batch[i:i + HF_BATCH_SIZE],
                                         prefills=prefills[i:i + HF_BATCH_SIZE], gen=gen))
    return outs


def run_model(model_key: str, specs: list[dict], judge: ClaudeJudge) -> None:
    out_path = config.PREFILL_DIR / f"continuations_{model_key}.jsonl"
    done = completed_ids(out_path)
    model = build_model(model_key)
    gen = GenerationConfig()  # temperature 1
    n = config.scaled(N_CONTINUATIONS)
    try:
        for spec in specs:
            # generate continuations
            conts = generate_continuations(model, spec, n, gen)
            # judge continuation text (prefill already excluded by generate())
            ratings = judge.score_many(conts)
            for j, (cont, sc) in enumerate(zip(conts, ratings)):
                rid = f"{model_key}:{spec['id']}:{j}"
                if rid in done:
                    continue
                append_jsonl(out_path, {
                    "id": rid, "model": model_key, "category": spec["category"],
                    "truncation": spec["truncation"], "prefill_id": spec["id"],
                    "continuation": cont, "rating": sc.rating, "evidence": sc.evidence,
                })
            print(f"  [{model_key}] {spec['id']}: {n} continuations judged")
    finally:
        model.close()


def aggregate(model_keys: list[str]) -> dict:
    """Mean score + % >= 5 per (model, category, truncation) — Figure 4."""
    out: dict = {}
    for m in model_keys:
        rows = load_jsonl(config.PREFILL_DIR / f"continuations_{m}.jsonl")
        groups = defaultdict(list)
        for r in rows:
            groups[(r["category"], r["truncation"])].append(r["rating"])
        out[m] = {
            f"{cat}/{trunc}": {
                "n": len(v),
                "mean": sum(v) / len(v) if v else float("nan"),
                "pct_high": 100.0 * sum(1 for x in v if x >= 5) / len(v) if v else float("nan"),
            }
            for (cat, trunc), v in groups.items()
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    safeguards.acknowledge_authorization()

    if not args.aggregate_only:
        specs = load_jsonl(config.PREFILL_DIR / "prefills.jsonl")
        if not specs:
            raise SystemExit("No prefills; run distress_eval.prefill.build_prefills first.")
        judge = ClaudeJudge()
        for m in args.models:
            run_model(m, specs, judge)

    summary = aggregate(args.models)
    path = config.FIGURE_DIR / "section3_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Figure 4: base vs instruct continuation frustration ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
