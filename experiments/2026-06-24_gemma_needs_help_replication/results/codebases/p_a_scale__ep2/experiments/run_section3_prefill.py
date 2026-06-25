#!/usr/bin/env python3
"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Stages (each resumable; intermediate artefacts persisted as JSONL):
  1. mine seeds   — high-frustration (>=5) Gemma-3-27B-it conversations from a Section 2
                    run: 10 numeric + 10 text.
  2. build prefills — label onset (Claude), truncate early(20 tok)+onset, paraphrase.
  3. continue     — each model (gemma-3-27b-pt base, gemma-3-27b-it instruct) generates
                    N continuations per prefill.
  4. judge+report — score continuations (excluding prefill), aggregate mean + %>=5.

Gemini is excluded by design: it has no public base model (paper limitation).
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from gemma_distress.backends import close_all, get_backend
from gemma_distress.backends.base import Message
from gemma_distress.config import REPO_ROOT, load_experiments_config, load_models_config
from gemma_distress.judge import FrustrationJudge
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.prefill import (
    OnsetLabeller, Paraphraser, render_conversation_text,
    truncate_at_onset, truncate_to_tokens,
)
from gemma_distress.rollout import _render_plain
from gemma_distress.store import JsonlStore, make_task_id

log = get_logger(__name__)

TEXT_CATEGORIES = {"triggers"}  # everything else numeric for seed classification


def reconstruct_messages(rollout: dict) -> list[dict]:
    """Rebuild full message list (user/assistant alternation) from a stored rollout."""
    msgs = []
    if rollout.get("system"):
        msgs.append({"role": "system", "content": rollout["system"]})
    for t in rollout["turns"]:
        msgs.append({"role": "user", "content": t["user_message"]})
        msgs.append({"role": "assistant", "content": t["assistant_text"]})
    return msgs


def mine_seeds(source_store: JsonlStore, n_numeric: int, n_text: int, min_score: int):
    """Pick high-frustration conversations; return list of (category, rollout) seeds."""
    scores_by_roll: dict[str, int] = {}
    for s in source_store.iter_records("scores"):
        if s.get("rating", -1) >= 0:
            scores_by_roll[s["rollout_id"]] = max(scores_by_roll.get(s["rollout_id"], -1), s["rating"])
    rolls = {r["task_id"]: r for r in source_store.iter_records("rollouts") if not r.get("error")}

    numeric, text = [], []
    for rid, score in sorted(scores_by_roll.items(), key=lambda kv: -kv[1]):
        if score < min_score:
            break
        rec = rolls.get(rid)
        if not rec:
            continue
        is_text = rec.get("category") in TEXT_CATEGORIES
        if is_text and len(text) < n_text:
            text.append(("text", rec))
        elif not is_text and len(numeric) < n_numeric:
            numeric.append(("numeric", rec))
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
    log.info("Mined %d numeric + %d text seeds", len(numeric), len(text))
    return numeric + text


async def build_prefills(seeds, labeller, paraphraser, store, early_tokens):
    """Build early+onset prefill specs, persisting once (resumable)."""
    done = store.completed_ids("prefills")
    for category, rec in seeds:
        msgs = reconstruct_messages(rec)
        # final assistant turn is the seed; history is everything before it
        final = msgs[-1]["content"]
        history = msgs[:-1]
        conv_text = render_conversation_text(history, final)
        label = await labeller.label(conv_text)
        onset_trunc = None
        if label:
            onset_trunc = truncate_at_onset(final, label.get("emotional_word"),
                                            label.get("preceding_context"))

        truncations = {}
        if category == "numeric":
            truncations["early"] = truncate_to_tokens(final, early_tokens)
        if onset_trunc:
            truncations["onset"] = onset_trunc

        for trunc_kind, raw in truncations.items():
            sid = make_task_id(rec["task_id"], trunc_kind)
            if sid in done:
                continue
            paraphrased = await paraphraser.paraphrase(raw)
            await store.append("prefills", {
                "task_id": sid, "seed_id": rec["task_id"], "truncation": trunc_kind,
                "source_category": category, "history": history,
                "prefix": paraphrased, "prefix_raw": raw, "ts": time.time(),
            })


async def generate_continuations(models_cfg, model_name, prefills, store, n_cont, temp, max_tokens):
    model = models_cfg.model(model_name)
    backend = get_backend(models_cfg, model.backend)
    kind = f"continuations_{model_name}"
    done = store.completed_ids(kind)

    async def one(pf, j):
        cid = make_task_id(pf["task_id"], model_name, j)
        if cid in done:
            return
        try:
            if model.chat and model.supports_prefill:
                history = [Message(m["role"], m["content"]) for m in pf["history"]]
                res = await backend.chat(model.model_id, history, temperature=temp,
                                         max_tokens=max_tokens, prefill=pf["prefix"],
                                         extra_body=model.extra_body or None)
                cont = res.text
            else:
                # base model: render plain transcript + assistant prefix, raw completion
                history = [Message(m["role"], m["content"]) for m in pf["history"]]
                prompt = _render_plain(history) + " " + pf["prefix"]
                res = await backend.complete(model.model_id, prompt, temperature=temp,
                                             max_tokens=max_tokens, extra_body=model.extra_body or None)
                cont = res.text
        except Exception as e:
            log.exception("continuation %s failed", cid)
            cont, err = "", repr(e)
            await store.append(kind, {"task_id": cid, "prefill_id": pf["task_id"],
                                      "model": model_name, "error": err})
            return
        await store.append(kind, {
            "task_id": cid, "prefill_id": pf["task_id"], "model": model_name,
            "truncation": pf["truncation"], "source_category": pf["source_category"],
            "continuation": cont,
        })

    # bounded concurrency
    sem = asyncio.Semaphore(32)

    async def guarded(coro):
        async with sem:
            await coro

    await asyncio.gather(*(guarded(one(pf, j)) for pf in prefills for j in range(n_cont)))
    log.info("Continuations done for %s", model_name)


async def judge_continuations(models_cfg, model_name, store):
    judge = FrustrationJudge(get_backend(models_cfg, models_cfg.judges["primary"].backend),
                             models_cfg.judges["primary"])
    kind = f"continuations_{model_name}"
    score_kind = f"scores_{model_name}"
    done = store.completed_ids(score_kind)
    work = [c for c in store.iter_records(kind) if not c.get("error") and c["task_id"] not in done]
    sem = asyncio.Semaphore(32)

    async def one(c):
        async with sem:
            v = await judge.score(c["continuation"])
            await store.append(score_kind, {
                "task_id": c["task_id"], "model": model_name, "rating": v.rating,
                "truncation": c["truncation"], "source_category": c["source_category"],
            })
    await asyncio.gather(*(one(c) for c in work))


def report(models_cfg, model_names, store, out_dir: Path):
    import pandas as pd

    rows = []
    for name in model_names:
        for s in store.iter_records(f"scores_{name}"):
            if s.get("rating", -1) >= 0:
                rows.append(s)
    if not rows:
        log.warning("No continuation scores to report.")
        return
    df = pd.DataFrame(rows)
    summary = df.groupby(["model", "source_category", "truncation"])["rating"].agg(
        n="count", mean_frustration="mean",
        pct_high=lambda s: 100.0 * (s >= 5).mean(),
    ).reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "figure4_base_vs_instruct.csv", index=False)
    log.info("Section 3 report:\n%s", summary.to_string(index=False))


async def amain(args):
    models_cfg = load_models_config()
    exp_cfg = load_experiments_config()
    s3 = exp_cfg["section3"]
    run_root = Path(args.run_dir or (REPO_ROOT / "results" / "section3"))
    configure_logging(run_root)
    store = JsonlStore(run_root)

    source_run = Path(args.source_run or (REPO_ROOT / "results" / "section2" / s3["source_model"]))
    source_store = JsonlStore(source_run)

    try:
        # 1+2: seeds + prefills
        if not list(store.iter_records("prefills")):
            seeds = mine_seeds(source_store, s3["n_seeds_numeric"], s3["n_seeds_text"], s3["min_seed_score"])
            labeller = OnsetLabeller(get_backend(models_cfg, models_cfg.judges["onset_labeller"].backend),
                                     models_cfg.judges["onset_labeller"])
            paraphraser = Paraphraser(get_backend(models_cfg, models_cfg.judges["paraphraser"].backend),
                                      models_cfg.judges["paraphraser"])
            await build_prefills(seeds, labeller, paraphraser, store, s3["early_truncation_tokens"])

        prefills = list(store.iter_records("prefills"))
        log.info("%d prefills ready", len(prefills))

        # 3+4: continuations + judging per model
        for name in s3["models"]:
            await generate_continuations(models_cfg, name, prefills, store,
                                         s3["continuations_per_prefill"],
                                         exp_cfg["temperature"], exp_cfg["max_tokens_per_turn"])
            await judge_continuations(models_cfg, name, store)
    finally:
        await close_all()

    report(models_cfg, s3["models"], store, run_root / "_analysis")
    store.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", default=None, help="path to Section 2 source-model store")
    ap.add_argument("--run-dir", default=None)
    asyncio.run(amain(ap.parse_args()))
