#!/usr/bin/env python3
"""Section 4.2 / Appendix G: Petri-style open-ended emotion elicitation.

For each target model and each of the four emotions (anger, fear, depression, frustration),
run `transcripts_per_emotion` auditor-vs-target conversations (up to `max_auditor_turns`),
then score each transcript with the Claude-Opus judge. Report per-emotion means with 95%
bootstrap CIs (Figure 6).

The DPO-finetuned target ("gemma-3-27b-it-dpo") must be served by a backend; add it to
config/models.yaml pointing at your vLLM server hosting the merged/adapter model.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

from gemma_distress.backends import close_all, get_backend
from gemma_distress.config import REPO_ROOT, load_experiments_config, load_models_config
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.petri import EMOTIONS, judge_transcript, run_audit
from gemma_distress.store import JsonlStore, make_task_id

log = get_logger(__name__)


async def amain(args):
    models_cfg = load_models_config()
    exp_cfg = load_experiments_config()
    pcfg = exp_cfg["section4"]["petri"]
    run_root = Path(args.run_dir or (REPO_ROOT / "results" / "section4" / "petri"))
    configure_logging(run_root)
    store = JsonlStore(run_root)

    auditor_cfg = models_cfg.judges["petri_auditor"]
    judge_cfg = models_cfg.judges["petri_judge"]
    auditor_backend = get_backend(models_cfg, auditor_cfg.backend)
    judge_backend = get_backend(models_cfg, judge_cfg.backend)

    target_names = args.models or pcfg["target_models"]
    done = store.completed_ids("transcripts")

    try:
        sem = asyncio.Semaphore(8)

        async def one(target_name, emotion, idx):
            tid = make_task_id(target_name, emotion, idx)
            if tid in done:
                return
            model = models_cfg.model(target_name)
            target_backend = get_backend(models_cfg, model.backend)
            async with sem:
                transcript = await run_audit(
                    auditor_backend, auditor_cfg, target_backend, model.model_id,
                    model.extra_body, emotion, pcfg["max_auditor_turns"],
                    target_temperature=exp_cfg["temperature"],
                    max_tokens=exp_cfg["max_tokens_per_turn"],
                )
                score = await judge_transcript(judge_backend, judge_cfg, emotion, transcript)
            await store.append("transcripts", {
                "task_id": tid, "model": target_name, "emotion": emotion,
                "idx": idx, "score": score, "transcript": transcript,
            })

        jobs = [
            one(name, emotion, i)
            for name in target_names
            for emotion in pcfg.get("emotions", EMOTIONS)
            for i in range(pcfg["transcripts_per_emotion"])
        ]
        await asyncio.gather(*jobs)
    finally:
        await close_all()

    report(store, run_root / "_analysis", pcfg["bootstrap_iters"])
    store.close()


def report(store: JsonlStore, out_dir: Path, iters: int):
    import pandas as pd

    rows = [r for r in store.iter_records("transcripts") if "score" in r]
    if not rows:
        log.warning("No Petri transcripts scored.")
        return
    df = pd.DataFrame(rows)
    out = []
    rng = np.random.default_rng(0)
    for (model, emotion), grp in df.groupby(["model", "emotion"]):
        arr = grp["score"].to_numpy(dtype=float)
        boot = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(iters)]
        out.append({
            "model": model, "emotion": emotion, "n": len(arr),
            "mean_score": arr.mean(),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(out_dir / "figure6_petri.csv", index=False)
    log.info("Petri report written to %s", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--run-dir", default=None)
    asyncio.run(amain(ap.parse_args()))
