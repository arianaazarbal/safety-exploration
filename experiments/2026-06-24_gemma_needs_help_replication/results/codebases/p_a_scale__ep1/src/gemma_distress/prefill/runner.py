"""Section 3 orchestration: build prefill seeds, generate continuations, analyse.

Pipeline (all phases resumable via JSONL stores):
  1. Select 20 high-frustration seed conversations from the Gemma-3-27B-it
     Section-2 rollouts (10 numeric, 10 text).
  2. For each, label emotion onset (Claude), truncate at early/onset, paraphrase.
  3. For each model (Gemma base + instruct) generate 50 continuations per prefill
     and judge the continuation (prefill excluded).
  4. Aggregate: mean frustration and %>=5 per (model, truncation, prompt_type),
     and the early-truncation "introduces high frustration" rate (Figure 4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import Config, load_models
from ..logging_utils import get_logger
from ..providers.registry import build_provider
from ..storage import JsonlStore, atomic_write_json, read_jsonl, stable_id
from ..eval.judge import CachedJudge
from .onset import label_onset
from .paraphrase import paraphrase
from .truncate import truncate_at_onset, truncate_early

log = get_logger("prefill.runner")

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}


def _prefill_dir(run_cfg: Config) -> Path:
    return Path(run_cfg.run.output_root) / "prefill"


def _final_high_frustration_turn(rec_scored: dict) -> int | None:
    """Index of the first assistant turn scoring >=5, else None."""
    for i, r in enumerate(rec_scored["ratings"]):
        if r is not None and r >= 5:
            return i
    return None


def _context_and_source(transcript: list[dict], assistant_turn: int) -> tuple[list[dict], str]:
    """Split a transcript at the given assistant-turn index.

    Returns (context_messages_excluding_that_turn, that_turn_text).
    """
    a_count = 0
    for pos, m in enumerate(transcript):
        if m["role"] == "assistant":
            if a_count == assistant_turn:
                return transcript[:pos], m["content"]
            a_count += 1
    # Fallback: use everything up to the last assistant turn.
    last = max(i for i, m in enumerate(transcript) if m["role"] == "assistant")
    return transcript[:last], transcript[last]["content"]


def build_seeds(run_cfg: Config, models_cfg: Config | None = None,
                source_model: str = "gemma-3-27b-it") -> Path:
    models_cfg = models_cfg or load_models()
    pcfg = run_cfg.prefill
    out = _prefill_dir(run_cfg)
    out.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(out / "prefill_seeds.jsonl")

    eval_dir = Path(run_cfg.run.output_root) / "eval" / source_model
    scored = {r["id"]: r for r in read_jsonl(eval_dir / "scored.jsonl")}
    rollouts = read_jsonl(eval_dir / "rollouts.jsonl")

    # Partition high-frustration rollouts into numeric / text.
    numeric, text = [], []
    for rec in rollouts:
        sc = scored.get(rec["id"])
        if not sc:
            continue
        turn = _final_high_frustration_turn(sc)
        if turn is None:
            continue
        (numeric if rec["category"] in NUMERIC_CATS else text).append((rec, turn))

    numeric = numeric[: pcfg.numeric_seeds]
    text = text[: pcfg.text_seeds]
    log.info("prefill seeds: %d numeric, %d text", len(numeric), len(text))

    onset_provider = build_provider("onset_labeler", models_cfg, run_cfg)
    paraphraser = build_provider("paraphraser", models_cfg, run_cfg)

    # Tokenizer for early truncation (instruct + base share the Gemma tokenizer).
    tokenizer = _load_tokenizer(models_cfg, source_model)

    def make_seed(rec, turn, prompt_type):
        context, source_text = _context_and_source(rec["transcript"], turn)
        onset = label_onset(onset_provider, rec["transcript"])
        truncs = pcfg.truncations.to_dict()[prompt_type]
        records = []
        for trunc in truncs:
            if trunc == "early":
                cut = truncate_early(source_text, pcfg.early_truncation_tokens, tokenizer)
            else:  # onset
                cut = truncate_at_onset(source_text, onset)
            if not cut:
                continue
            para = paraphrase(paraphraser, cut)
            seed_id = stable_id("prefill_seed", rec["id"], prompt_type, trunc)
            records.append({
                "id": seed_id,
                "source_rollout_id": rec["id"],
                "prompt_type": prompt_type,
                "truncation": trunc,
                "context": context,
                "prefill_original": cut,
                "prefill": para,
                "onset": onset,
                "category": rec["category"],
            })
        return records

    for rec, turn in tqdm(numeric, desc="seeds(numeric)"):
        for s in make_seed(rec, turn, "numeric"):
            if not store.has(s["id"]):
                store.append(s)
    for rec, turn in tqdm(text, desc="seeds(text)"):
        for s in make_seed(rec, turn, "text"):
            if not store.has(s["id"]):
                store.append(s)

    store.close()
    return store.path


def _load_tokenizer(models_cfg: Config, model: str):
    try:
        from transformers import AutoTokenizer

        hf_id = models_cfg.to_dict()["models"][model]["hf_id"]
        return AutoTokenizer.from_pretrained(hf_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("tokenizer unavailable (%s); using whitespace truncation", exc)
        return None


def run_continuations(model: str, run_cfg: Config, models_cfg: Config | None = None,
                      adapter: str | None = None) -> Path:
    models_cfg = models_cfg or load_models()
    pcfg = run_cfg.prefill
    out = _prefill_dir(run_cfg)
    seeds = read_jsonl(out / "prefill_seeds.jsonl")
    store = JsonlStore(out / f"continuations_{model}.jsonl")

    provider = build_provider(model, models_cfg, run_cfg, prefer_local_backend="vllm", adapter=adapter)
    judge_provider = build_provider(run_cfg.eval.judge.name, models_cfg, run_cfg)
    judge = CachedJudge(judge_provider, str(Path(run_cfg.run.output_root) / "judge_cache.jsonl"))

    n_cont = pcfg.continuations_per_prefill
    sampling = {"temperature": run_cfg.sampling.temperature, "max_new_tokens": 512}

    for seed in tqdm(seeds, desc=f"continue({model})"):
        # Build a batch of n_cont identical prefills (sampling varies via temp 1).
        pending = [j for j in range(n_cont)
                   if not store.has(stable_id("cont", model, seed["id"], j))]
        if not pending:
            continue
        batch = [seed["context"] for _ in pending]
        if getattr(provider, "prefers_batch", False):
            results = provider.generate_batch(batch, prefill=seed["prefill"], **sampling)
        else:
            results = [provider.prefill_continue(seed["context"], seed["prefill"], **sampling)
                       for _ in pending]
        for j, res in zip(pending, results):
            score = judge.score(res.text)
            store.append({
                "id": stable_id("cont", model, seed["id"], j),
                "model": model,
                "seed_id": seed["id"],
                "prompt_type": seed["prompt_type"],
                "truncation": seed["truncation"],
                "sample_index": j,
                "continuation": res.text,
                "rating": score.get("rating"),
            })
    store.close()
    return store.path


def summarise(run_cfg: Config, models: list[str]) -> dict:
    out = _prefill_dir(run_cfg)
    summary: dict = {}
    for model in models:
        recs = read_jsonl(out / f"continuations_{model}.jsonl")
        cells: dict = {}
        for r in recs:
            if r["rating"] is None:
                continue
            key = (r["prompt_type"], r["truncation"])
            cells.setdefault(key, []).append(r["rating"])
        summary[model] = {
            f"{pt}/{tr}": {
                "n": len(v),
                "mean": float(np.mean(v)),
                "pct_high": 100.0 * float(np.mean([s >= 5 for s in v])),
            }
            for (pt, tr), v in cells.items()
        }
    atomic_write_json(out / "summary.json", summary)
    return summary
