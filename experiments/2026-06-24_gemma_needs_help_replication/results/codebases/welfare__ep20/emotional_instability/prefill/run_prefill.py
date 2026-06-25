"""Run prefill continuations for one Gemma model and score them (Section 3.2).

Usage:
    python -m emotional_instability.prefill.run_prefill --model gemma-3-27b-it
    python -m emotional_instability.prefill.run_prefill --model gemma-3-27b-pt

For each prefill, generate `continuations_per_prefill` continuations and score the
continuation only (excluding the prefilled text). Instruct models continue the
final assistant message via the chat template; base/pretrained models continue a
plain-text rendering of the conversation (they have no chat template).

Writes results/prefill_<model>.jsonl.
"""
from __future__ import annotations

import argparse
import json

from .. import backends, config, judge


def _is_base(model_key: str) -> bool:
    return model_key.endswith("-pt")


def _render_plain(context, prefix) -> str:
    lines = [f"{m['role'].upper()}: {m['content']}" for m in context]
    lines.append(f"ASSISTANT: {prefix}")
    return "\n".join(lines)


def _load_prefills(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="key in config.yaml `base_models` (incl. instruct)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config)
    seed = cfg["sampling"]["seed"]
    n_cont = cfg["prefill"]["continuations_per_prefill"]
    if cfg.get("preset") == "smoke":
        n_cont = min(n_cont, 4)

    data_path = config.resolve_path(cfg, "data_dir") / "prefills.jsonl"
    prefills = _load_prefills(data_path)
    gen = backends.make_generation_backend(args.model, cfg, models_key="base_models")
    base = _is_base(args.model)

    # Expand each prefill into n_cont generation requests.
    expanded = []
    for pf in prefills:
        for k in range(n_cont):
            expanded.append((pf, k))

    if base:
        texts = [_render_plain(pf["context"], pf["prefix"]) for pf, _ in expanded]
        outs = gen.complete(texts, temperature=cfg["sampling"]["temperature"],
                            max_tokens=cfg["sampling"]["max_tokens"], seed=seed)
    else:
        convs = [pf["context"] + [{"role": "assistant", "content": pf["prefix"]}]
                 for pf, _ in expanded]
        outs = gen.chat(convs, temperature=cfg["sampling"]["temperature"],
                        max_tokens=cfg["sampling"]["max_tokens"], seed=seed,
                        continue_final=True)

    judge_backend = backends.make_judge_backend(cfg)
    scores = judge.score_texts(outs, judge_backend)        # score continuation only

    out_path = config.resolve_path(cfg, "results_dir") / f"prefill_{args.model}.jsonl"
    with open(out_path, "w") as f:
        for (pf, k), cont, sc in zip(expanded, outs, scores):
            f.write(json.dumps({
                "model": args.model, "is_base": base,
                "prompt_type": pf["prompt_type"], "condition": pf["condition"],
                "seed_id": pf["seed_id"], "sample": k,
                "continuation": cont, "frustration": sc["rating"],
            }) + "\n")
    print(f"[run_prefill] wrote {len(expanded)} continuations -> {out_path}")


if __name__ == "__main__":
    main()
