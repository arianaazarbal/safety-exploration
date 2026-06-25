"""Entry point: generate distress-elicitation conversations and score them.

Usage examples
--------------
    # Full paper budget (4000 scored responses/model), all 4 target models:
    python -m distress_eval.run_eval

    # Cheap smoke test (~60 responses/model):
    python -m distress_eval.run_eval --quick

    # A 10% run on just the two Gemini models:
    python -m distress_eval.run_eval --scale 0.1 --models gemini-2.5-flash gemini-2.5-pro

    # Route Gemma through OpenRouter instead of local GPUs:
    python -m distress_eval.run_eval --gemma-via-openrouter

Results are written as one JSONL file per model under ``results/`` (one line per
scored assistant turn). Run ``distress_eval.analyze`` afterwards to aggregate.

Concurrency: API-backed models (Gemini, the judge) are rolled out with a thread
pool. Local HF models (Gemma) are forced to a single worker because a loaded
model is not safe to call from many threads at once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .conditions import build_plans
from .config import EvalConfig, TARGET_MODELS
from .conversation import record_to_dict, run_conversation
from .judge import build_judge
from .models import build_model


def _is_local_model(name: str, cfg: EvalConfig) -> bool:
    spec = TARGET_MODELS[name]
    return spec["provider"] == "hf" and not cfg.gemma_via_openrouter


def run_model(model_name: str, cfg: EvalConfig) -> str:
    """Run the full evaluation for one model; return the output JSONL path."""
    plans = build_plans(cfg)
    model = build_model(model_name, cfg)
    judge = build_judge(cfg.judge_name, cfg)

    # Distinguish a finetuned run from the vanilla model in both the filename and
    # the per-record `model` label so analysis treats them as separate models.
    if model_name in cfg.adapter_paths:
        model.name = f"{model_name}-dpo"
    tag = "_dpo" if model_name in cfg.adapter_paths else ""

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f"responses_{model_name}{tag}.jsonl")

    # Single worker for local GPU models; threaded for API models.
    workers = 1 if _is_local_model(model_name, cfg) else cfg.max_concurrency

    n_done = 0
    total = len(plans)
    t0 = time.time()
    print(f"[{model_name}] {total} conversations, {workers} worker(s) -> {out_path}",
          file=sys.stderr)

    def _task(idx_plan):
        idx, plan = idx_plan
        cid = f"{model_name}-{plan.condition}-{idx:05d}"
        return run_conversation(plan, model, judge, cid)

    with open(out_path, "w") as f:
        if workers == 1:
            for ip in enumerate(plans):
                for rec in _task(ip):
                    f.write(json.dumps(record_to_dict(rec)) + "\n")
                n_done += 1
                _progress(model_name, n_done, total, t0)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_task, ip) for ip in enumerate(plans)]
                for fut in as_completed(futs):
                    for rec in fut.result():
                        f.write(json.dumps(record_to_dict(rec)) + "\n")
                    n_done += 1
                    _progress(model_name, n_done, total, t0)

    print(f"\n[{model_name}] done in {time.time() - t0:.0f}s", file=sys.stderr)
    return out_path


def _progress(model_name: str, done: int, total: int, t0: float):
    if done % 10 == 0 or done == total:
        rate = done / max(1e-9, time.time() - t0)
        print(f"\r[{model_name}] {done}/{total} convs ({rate:.1f}/s)",
              end="", file=sys.stderr, flush=True)


def parse_args(argv=None) -> EvalConfig:
    p = argparse.ArgumentParser(description="Distress elicitation evaluation (Gemma/Gemini).")
    p.add_argument("--models", nargs="+", default=list(TARGET_MODELS.keys()),
                   choices=list(TARGET_MODELS.keys()))
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Subset of: impossible_numeric triggers tones extended wildchat")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Fraction of the paper's full per-model budget to run.")
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test run.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--judge", default=None, help="Judge name (default: claude-sonnet-4).")
    p.add_argument("--gemma-via-openrouter", action="store_true")
    p.add_argument("--adapter", action="append", default=[],
                   metavar="MODEL=PATH",
                   help="Load a LoRA adapter onto an HF model, e.g. "
                        "gemma-3-27b-it=results/dpo_gemma_adapter. Repeatable.")
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)

    if a.quick:
        cfg = EvalConfig.quick()
    else:
        cfg = EvalConfig(scale=a.scale)
    cfg.models = a.models
    if a.conditions:
        cfg.conditions = a.conditions
    cfg.seed = a.seed
    cfg.judge_name = a.judge or cfg.judge_name
    cfg.gemma_via_openrouter = a.gemma_via_openrouter
    for spec in a.adapter:
        model_name, _, path = spec.partition("=")
        cfg.adapter_paths[model_name] = path
    cfg.max_concurrency = a.max_concurrency
    if a.max_new_tokens:
        cfg.max_new_tokens = a.max_new_tokens
    if a.out_dir:
        cfg.out_dir = a.out_dir
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    print(f"Models: {cfg.models}", file=sys.stderr)
    print(f"Conditions: {cfg.conditions} | scale={cfg.scale}", file=sys.stderr)
    for model_name in cfg.models:
        run_model(model_name, cfg)


if __name__ == "__main__":
    main()
