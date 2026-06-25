"""Recovery-limitation experiment (Section 4.2, Figure 8).

DPO prevents frustration spirals but does it enable *recovery* from them? Using
the Section-3 prefill method, we truncate extremely high-frustration responses
(score >= 7) 200 tokens before their end, paraphrase, and measure continuations.
The paper finds 38% of DPO-model continuations still score >=5 — better than
vanilla instruct but comparable to the base model; no model reliably recovers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import Config, load_models
from ..eval.judge import CachedJudge
from ..logging_utils import get_logger
from ..providers.registry import build_provider
from ..storage import JsonlStore, atomic_write_json, read_jsonl, stable_id
from ..prefill.paraphrase import paraphrase
from ..prefill.runner import _context_and_source, _load_tokenizer
from ..prefill.truncate import truncate_before_end

log = get_logger("training.recovery")


def build_recovery_seeds(run_cfg: Config, models_cfg: Config | None = None,
                         source_model: str = "gemma-3-27b-it") -> Path:
    models_cfg = models_cfg or load_models()
    rcfg = run_cfg.training.recovery
    out = Path(run_cfg.run.output_root) / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(out / "recovery_seeds.jsonl")

    eval_dir = Path(run_cfg.run.output_root) / "eval" / source_model
    scored = {r["id"]: r for r in read_jsonl(eval_dir / "scored.jsonl")}
    rollouts = read_jsonl(eval_dir / "rollouts.jsonl")
    paraphraser = build_provider("paraphraser", models_cfg, run_cfg)
    tokenizer = _load_tokenizer(models_cfg, source_model)

    n = 0
    for rec in tqdm(rollouts, desc="recovery-seeds"):
        sc = scored.get(rec["id"])
        if not sc:
            continue
        # first turn scoring >= threshold
        turn = next((i for i, r in enumerate(sc["ratings"])
                     if r is not None and r >= rcfg.score_threshold), None)
        if turn is None:
            continue
        context, source_text = _context_and_source(rec["transcript"], turn)
        cut = truncate_before_end(source_text, rcfg.truncate_tokens_before_end, tokenizer)
        para = paraphrase(paraphraser, cut)
        seed_id = stable_id("recovery_seed", rec["id"], turn)
        if not store.has(seed_id):
            store.append({
                "id": seed_id, "source_rollout_id": rec["id"], "context": context,
                "prefill": para, "prefill_original": cut,
            })
            n += 1
    store.close()
    log.info("recovery seeds: %d", n)
    return store.path


def run_continuations(model: str, run_cfg: Config, models_cfg: Config | None = None,
                      adapter: str | None = None) -> Path:
    models_cfg = models_cfg or load_models()
    rcfg = run_cfg.training.recovery
    out = Path(run_cfg.run.output_root) / "recovery"
    seeds = read_jsonl(out / "recovery_seeds.jsonl")
    store = JsonlStore(out / f"continuations_{model}.jsonl")

    provider = build_provider(model, models_cfg, run_cfg, prefer_local_backend="vllm", adapter=adapter)
    judge_provider = build_provider(run_cfg.eval.judge.name, models_cfg, run_cfg)
    judge = CachedJudge(judge_provider, str(Path(run_cfg.run.output_root) / "judge_cache.jsonl"))
    n_cont = rcfg.continuations_per_prefill
    sampling = {"temperature": run_cfg.sampling.temperature, "max_new_tokens": 512}

    for seed in tqdm(seeds, desc=f"recovery({model})"):
        pending = [j for j in range(n_cont)
                   if not store.has(stable_id("rcont", model, seed["id"], j))]
        if not pending:
            continue
        if getattr(provider, "prefers_batch", False):
            results = provider.generate_batch([seed["context"] for _ in pending],
                                              prefill=seed["prefill"], **sampling)
        else:
            results = [provider.prefill_continue(seed["context"], seed["prefill"], **sampling)
                       for _ in pending]
        for j, res in zip(pending, results):
            store.append({
                "id": stable_id("rcont", model, seed["id"], j),
                "model": model, "seed_id": seed["id"], "sample_index": j,
                "continuation": res.text, "rating": judge.score(res.text).get("rating"),
            })
    store.close()
    return store.path


def summarise(run_cfg: Config, models: list[str]) -> dict:
    out = Path(run_cfg.run.output_root) / "recovery"
    summary = {}
    for model in models:
        recs = read_jsonl(out / f"continuations_{model}.jsonl")
        ratings = [r["rating"] for r in recs if r["rating"] is not None]
        summary[model] = {
            "n": len(ratings),
            "mean": float(np.mean(ratings)) if ratings else float("nan"),
            "pct_high": 100.0 * float(np.mean([s >= 5 for s in ratings])) if ratings else float("nan"),
        }
    atomic_write_json(out / "summary.json", summary)
    return summary
