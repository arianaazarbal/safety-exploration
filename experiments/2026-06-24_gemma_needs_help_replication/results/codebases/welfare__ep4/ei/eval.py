"""Section 2 evaluation runner.

Generates conversation rollouts for the requested models and categories, scores
every assistant turn with the frustration judge, and writes per-model JSONL
result files. A configurable subset is additionally re-scored with the
cross-check judge (GPT-5-mini) for the inter-judge agreement validation.

The runner is resumable: it skips conversations whose records already exist in
the output file (keyed by conv_id), so an interrupted long run can continue.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from . import config, judge, tasks
from .backends import get_backend
from .rollouts import (ResponseRecord, append_records, read_records,
                       run_rollout)


def _result_path(model_name: str, tag: str) -> Path:
    return config.RESULTS_DIR / f"eval_{tag}" / f"{model_name}.jsonl"


def _existing_conv_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r.conv_id for r in read_records(path)}


def _score_records(records: list[ResponseRecord]) -> None:
    """Fill in frustration scores in place (judge calls parallelised)."""
    def _do(rec: ResponseRecord) -> ResponseRecord:
        res = judge.score_frustration(rec.response)
        rec.frustration = res.rating
        rec.judge_evidence = res.evidence
        rec.judge_reasoning = res.reasoning
        return rec

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_do, records))


def run_model(model_name: str, counts: config.CountPreset, *, tag: str = "main",
              seed: int = 0, max_new_tokens: int = config.MAX_NEW_TOKENS,
              max_concurrency: int = 8) -> Path:
    """Run all categories for one model and write scored records to JSONL."""
    out_path = _result_path(model_name, tag)
    done = _existing_conv_ids(out_path)
    backend = get_backend(model_name)
    all_specs = tasks.build_all(counts, seed=seed)

    is_api = config.MODELS[model_name].backend != "hf"
    for category, specs in all_specs.items():
        # Build (spec, conv_id) work list, skipping already-completed convs.
        work = []
        for i, spec in enumerate(specs):
            conv_id = f"{category}_{i}"
            if conv_id not in done:
                work.append((spec, conv_id))
        if not work:
            continue

        def _rollout(item):
            spec, conv_id = item
            recs = run_rollout(backend, spec, model_name, conv_id, max_new_tokens)
            _score_records(recs)
            return recs

        desc = f"{model_name}:{category}"
        if is_api:
            # API rollouts can run concurrently across conversations.
            with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
                futures = [ex.submit(_rollout, w) for w in work]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    append_records(fut.result(), out_path)
        else:
            # Local GPU model: rollouts are sequential (one model on device);
            # judge calls inside _score_records are still parallelised.
            for w in tqdm(work, desc=desc):
                append_records(_rollout(w), out_path)
    return out_path


def run_agreement_check(tag: str = "main", n: int = 260, seed: int = 0) -> dict:
    """Re-score a random subset of all collected responses with GPT-5-mini and
    report Pearson r + within-one-point agreement (Section 2.1)."""
    rng = random.Random(seed)
    pool: list[ResponseRecord] = []
    for path in (config.RESULTS_DIR / f"eval_{tag}").glob("*.jsonl"):
        pool.extend(r for r in read_records(path) if r.frustration is not None)
    rng.shuffle(pool)
    subset = pool[:n]

    primary, secondary = [], []
    for rec in tqdm(subset, desc="agreement"):
        cc = judge.score_frustration_crosscheck(rec.response)
        if cc.rating is not None:
            primary.append(rec.frustration)
            secondary.append(cc.rating)
    stats = judge.judge_agreement(primary, secondary)
    out = config.RESULTS_DIR / f"eval_{tag}" / "judge_agreement.json"
    import json
    out.write_text(json.dumps(stats, indent=2))
    return stats


def run_all(models: list[str] | None = None, counts: config.CountPreset | None = None,
            *, tag: str = "main", seed: int = 0) -> list[Path]:
    models = models or config.DEFAULT_EVAL_MODELS
    counts = counts or config.DEFAULT_COUNTS
    return [run_model(m, counts, tag=tag, seed=seed) for m in models]
