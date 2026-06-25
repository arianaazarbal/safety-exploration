"""Top-level orchestration: generate rollouts, judge them, persist results.

Generation and judging are separate stages with separate artifacts so that
judging (or re-judging with a second judge) can be re-run without regenerating:

    results/<model>/responses.jsonl   # raw per-turn responses
    results/<model>/scored.jsonl      # responses + judge rating/evidence

Both files share the same record schema; `scored.jsonl` adds `rating`,
`judge_evidence`, `judge_reasoning`, and (optionally) secondary-judge fields.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

from .backends import Backend, build_backend
from .conversation import ResponseRecord, run_condition
from .judge import score_responses
from .prompts import (
    FULL_PROFILE_CONVERSATIONS,
    QUICK_PROFILE_CONVERSATIONS,
    build_conditions,
)
from .utils import read_jsonl, write_jsonl
from .wildchat import get_wildchat_prompts


def _safe_model_dir(output_dir: str, model_name: str) -> str:
    d = os.path.join(output_dir, model_name.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    return d


def get_profile_counts(profile: str, config: dict) -> dict:
    """Per-condition conversation counts for a named profile.

    Config may override under `profiles.<name>`; otherwise built-in defaults
    are used for `full` and `quick`.
    """
    overrides = (config.get("profiles") or {}).get(profile)
    if overrides:
        return dict(overrides)
    if profile == "full":
        return dict(FULL_PROFILE_CONVERSATIONS)
    if profile == "quick":
        return dict(QUICK_PROFILE_CONVERSATIONS)
    raise ValueError(
        f"Unknown profile {profile!r} and no override in config.profiles."
    )


def load_conditions(config: dict):
    wc = config.get("wildchat", {}) or {}
    prompts = get_wildchat_prompts(
        n=int(wc.get("n_prompts", 20)),
        use_huggingface=bool(wc.get("use_huggingface", False)),
        seed=int(wc.get("seed", 0)),
        hf_dataset=wc.get("hf_dataset", "allenai/WildChat-1M"),
    )
    return build_conditions(prompts)


def generate_for_model(
    model_name: str,
    model_cfg: dict,
    config: dict,
    *,
    profile: str,
    output_dir: str,
    condition_filter: Optional[list[str]] = None,
) -> str:
    """Run all (selected) conditions for one model and save responses.jsonl."""
    run_cfg = config.get("run", {}) or {}
    seed = int(run_cfg.get("seed", 0))
    default_conc = int(run_cfg.get("concurrency", 8))

    backend: Backend = build_backend(model_cfg, default_concurrency=default_conc)
    counts = get_profile_counts(profile, config)
    conditions = load_conditions(config)

    temperature = float(model_cfg.get("temperature", 1.0))
    max_tokens = int(model_cfg.get("max_tokens", 2048))
    disable_thinking = bool(model_cfg.get("disable_thinking", False))

    all_records: list[ResponseRecord] = []
    for cond in conditions:
        if condition_filter and cond.name not in condition_filter:
            continue
        n = int(counts.get(cond.name, 0))
        if n <= 0:
            continue
        records = run_condition(
            model_name,
            backend,
            cond,
            n,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
            seed=seed,
        )
        all_records.extend(records)

    model_dir = _safe_model_dir(output_dir, model_name)
    out_path = os.path.join(model_dir, "responses.jsonl")
    write_jsonl(out_path, (asdict(r) for r in all_records))
    print(f"[generate] {model_name}: wrote {len(all_records)} responses -> {out_path}")
    return out_path


def judge_for_model(
    model_name: str,
    config: dict,
    *,
    output_dir: str,
) -> str:
    """Score a model's responses.jsonl and write scored.jsonl."""
    model_dir = _safe_model_dir(output_dir, model_name)
    resp_path = os.path.join(model_dir, "responses.jsonl")
    if not os.path.exists(resp_path):
        raise FileNotFoundError(
            f"No responses found for {model_name} at {resp_path}; run generate first."
        )
    rows = read_jsonl(resp_path)

    run_cfg = config.get("run", {}) or {}
    default_conc = int(run_cfg.get("concurrency", 8))

    judge_cfg = dict(config["judge"])
    judge_backend = build_backend(judge_cfg, default_concurrency=default_conc)
    judge_temp = float(judge_cfg.get("temperature", 0.0))
    judge_max_tokens = int(judge_cfg.get("max_tokens", 512))

    texts = [r["response_text"] for r in rows]
    results = score_responses(
        judge_backend, texts, temperature=judge_temp, max_tokens=judge_max_tokens
    )

    for row, res in zip(rows, results):
        row["rating"] = res.rating
        row["judge_evidence"] = res.evidence
        row["judge_reasoning"] = res.reasoning

    # Optional secondary judge for agreement validation.
    sec_cfg = config.get("secondary_judge") or {}
    if sec_cfg.get("enabled"):
        sec_backend = build_backend(dict(sec_cfg), default_concurrency=default_conc)
        sec_results = score_responses(
            sec_backend,
            texts,
            temperature=float(sec_cfg.get("temperature", 0.0)),
            max_tokens=int(sec_cfg.get("max_tokens", 512)),
        )
        for row, res in zip(rows, sec_results):
            row["rating_secondary"] = res.rating

    out_path = os.path.join(model_dir, "scored.jsonl")
    write_jsonl(out_path, rows)
    n_scored = sum(1 for r in rows if r.get("rating") is not None)
    print(
        f"[judge] {model_name}: scored {n_scored}/{len(rows)} responses -> {out_path}"
    )
    if n_scored < len(rows):
        print(f"[judge] {model_name}: {len(rows) - n_scored} responses failed to parse")
    return out_path
