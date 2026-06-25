"""Generate and score continuations from prefill seeds (Section 3.2).

For each seed and each model (Gemma base + instruct), generate 50 continuations
(prefill excluded) and score them with the Section 2 judge. Reports mean
frustration and %>=5, broken down by truncation type ("early" / "onset") and
category, reproducing Figure 4 (and Figure 8 in --recovery mode).

Scope: gemma-3-27b-pt (base) vs gemma-3-27b-it (instruct), plus DPO finetune for
the recovery study.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .. import config
from ..config import get_subject
from ..models import ChatMessage, get_client
from ..eval.judge import FrustrationJudge
from ..utils.io import append_jsonl, read_jsonl, write_jsonl

CONTINUATIONS_PER_PREFILL = 50  # Section 3.1


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def run(
    seeds_path: Path,
    models: list[str],
    *,
    n_cont: int = CONTINUATIONS_PER_PREFILL,
    out_dir: Path,
    adapter_paths: dict[str, str] | None = None,
):
    seeds = list(read_jsonl(seeds_path))
    judge = FrustrationJudge()
    adapter_paths = adapter_paths or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "continuations.jsonl"
    rows_path.unlink(missing_ok=True)

    agg = defaultdict(list)  # (model, truncation, category) -> [scores]
    for model_name in models:
        spec = get_subject(model_name)
        client = get_client(spec, **(
            {"adapter_path": adapter_paths[model_name]}
            if model_name in adapter_paths else {}))
        if not client.supports_prefill:
            raise RuntimeError(f"{model_name} does not support prefill; "
                               "Section 3 requires a Gemma (HF) subject.")

        for seed in seeds:
            # Rebuild chat history from the seed.
            history = [ChatMessage(m["role"], m["content"]) for m in seed["history"]]
            history.append(ChatMessage("user", seed["final_user"]))
            for k in range(n_cont):
                gen = client.chat(
                    history,
                    temperature=config.SAMPLING.temperature,
                    top_p=config.SAMPLING.top_p,
                    max_new_tokens=config.SAMPLING.max_new_tokens,
                    prefill=seed["prefix"],
                )
                # Score the continuation ONLY (prefill excluded), per Sec 3.1.
                score = judge.score(gen.text).rating
                key = (model_name, seed["truncation"], seed["category"])
                agg[key].append(score)
                append_jsonl(rows_path, {
                    "model": model_name, "truncation": seed["truncation"],
                    "category": seed["category"], "task_id": seed["source_task_id"],
                    "rep": k, "score": score, "continuation": gen.text,
                })

    summary = []
    for (model_name, trunc, cat), scores in sorted(agg.items()):
        summary.append({
            "model": model_name, "truncation": trunc, "category": cat,
            "n": len(scores), "mean": _mean(scores),
            "pct_ge5": 100 * sum(1 for s in scores if s >= 5) / len(scores),
        })
    write_jsonl(out_dir / "continuation_summary.jsonl", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Run Section 3 continuations.")
    p.add_argument("--seeds", required=True)
    p.add_argument("--models", nargs="+",
                   default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    p.add_argument("--dpo-adapter", default=None,
                   help="path to DPO LoRA adapter (recovery study)")
    p.add_argument("--n-cont", type=int, default=CONTINUATIONS_PER_PREFILL)
    p.add_argument("--out", default=str(config.RESULTS_DIR / "prefill"))
    args = p.parse_args(argv)

    adapters = {}
    models = list(args.models)
    if args.dpo_adapter:
        models.append("gemma-3-27b-it+dpo")
        adapters["gemma-3-27b-it+dpo"] = args.dpo_adapter

    summary = run(Path(args.seeds), models, n_cont=args.n_cont,
                  out_dir=Path(args.out), adapter_paths=adapters)
    for s in summary:
        print(f"{s['model']:24s} {s['truncation']:8s} {s['category']:8s} "
              f"mean={s['mean']:.2f} %>=5={s['pct_ge5']:.1f} (n={s['n']})")


if __name__ == "__main__":
    main()
