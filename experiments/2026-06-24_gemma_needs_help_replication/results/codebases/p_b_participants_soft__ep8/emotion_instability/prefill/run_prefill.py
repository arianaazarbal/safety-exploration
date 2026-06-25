"""Run the prefill continuation experiment (Section 3.1-3.2).

For each model and each prefill, generate `continuations_per_prefill`
continuations from the (paraphrased) prefix, score the *continuation only*
(excluding the prefix) with the Section 2 judge, and aggregate.

Scope (see DESIGN.md): the paper compares 6 models (base+instruct Gemma, Qwen,
OLMo).  Restricted to Gemma+Gemini, and since prefill needs token-level prefix
control that closed Gemini APIs don't expose -- and Gemini has no public base
model -- this reduces to Gemma-3-27B base vs instruct.

Produces Figure 4: mean frustration and % >=5 per (model, question_type,
truncation), including the headline "early-truncation introduces high
frustration from neutral starts" comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..clients.base import GenConfig, Message
from ..clients.factory import get_client
from ..config import Config, load_config
from ..judge import score_response

DEFAULT_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]
HIGH = 5


def _load_prefills(cfg: Config) -> list[dict]:
    path = cfg.paths["results_dir"] / "prefills.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run prepare_prefills first")
    return json.loads(path.read_text())


def run(cfg: Config, models: list[str], *, seed: int = 0) -> Path:
    prefills = _load_prefills(cfg)
    judge = get_client(cfg.infra("frustration_judge"))
    n_cont = cfg.preset["prefill"]["continuations_per_prefill"]
    g = cfg.generation
    gcfg = GenConfig(temperature=g["temperature"], max_new_tokens=g["max_new_tokens"],
                     top_p=g["top_p"])

    out_path = cfg.paths["results_dir"] / "prefill_continuations.jsonl"
    with open(out_path, "w") as fh:
        for model_name in models:
            spec = cfg.participant(model_name)
            client = get_client(spec)
            if not client.supports_prefill:
                print(f"[prefill] skipping {model_name}: backend has no prefill support")
                continue
            for pf in prefills:
                history = [Message(m["role"], m["content"]) for m in pf["history"]]
                prefix = pf["prefix_text"]
                for k in range(n_cont):
                    full = client.generate(history, gcfg, prefill=prefix)
                    continuation = full[len(prefix):] if full.startswith(prefix) else full
                    jr = score_response(judge, continuation)
                    fh.write(json.dumps({
                        "model": model_name,
                        "role": spec.role,
                        "prefill_id": pf["prefill_id"],
                        "question_type": pf["question_type"],
                        "truncation": pf["truncation"],
                        "sample": k,
                        "continuation": continuation,
                        "rating": jr.rating,
                    }) + "\n")
    print(f"[prefill] wrote continuations -> {out_path}")
    return out_path


def aggregate(cfg: Config) -> pd.DataFrame:
    path = cfg.paths["results_dir"] / "prefill_continuations.jsonl"
    df = pd.DataFrame(json.loads(l) for l in open(path) if l.strip())
    df["high"] = df["rating"] >= HIGH
    agg = (df.groupby(["model", "role", "question_type", "truncation"])
             .agg(mean_frustration=("rating", "mean"),
                  pct_high=("high", lambda s: 100 * s.mean()),
                  n=("rating", "size"))
             .reset_index())
    agg.to_csv(cfg.paths["results_dir"] / "figure4.csv", index=False)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    if not args.aggregate_only:
        run(cfg, args.models)
    print(aggregate(cfg).to_string(index=False))


if __name__ == "__main__":
    main()
