"""Main distress evaluation (Section 2): elicit, roll out, judge, aggregate.

Pipeline per model:
  1. Build the 8 conditions / 5 categories of conversation specs sized to the
     per-category response budgets (config ``sampling.responses_per_model``).
  2. Roll out every conversation (batched for local Gemma, thread-pooled for
     API Gemini).
  3. Judge every assistant turn with the frustration judge (Appendix B.2).
  4. Write scored responses to JSONL and emit Figure 1/2/3 + Table 3 + judge
     agreement summaries.

Usage:
    python -m gemma_distress.experiments.run_eval --config config.yaml \
        --models gemma-3-27b-it gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import random

from ..analysis import (
    aggregate_scores,
    differential_words,
    figure1_table,
    judge_agreement,
    per_turn_curves,
)
from ..conditions import build_conversation_specs
from ..config import load_config
from ..conversation import Rollout, run_rollout, run_rollouts_lockstep
from ..judge import FrustrationJudge
from ..models import GenerationConfig, build_model
from ..utils import run_dir, thread_map, write_json, write_jsonl


def _rollout_model(model, specs, gen, is_local, api_concurrency):
    if is_local:
        return run_rollouts_lockstep(model, specs, gen)
    # API model: thread pool across conversations.
    results = thread_map(
        lambda s: run_rollout(model, s, gen),
        specs,
        max_workers=api_concurrency,
        desc=f"rollouts:{model.name}",
    )
    return [r for r in results if isinstance(r, Rollout)]


def _scored_rows_from_rollouts(rollouts: list[Rollout]) -> list[dict]:
    """Flatten rollouts into one row per assistant turn (pre-judging)."""
    rows = []
    for ro in rollouts:
        base = ro.to_row()
        for t in base["turns"]:
            rows.append(
                {
                    "model": base["model"],
                    "condition": base["condition"],
                    "category": base["category"],
                    "task_id": base["task_id"],
                    "task_kind": base["task_kind"],
                    "task_subtype": base["task_subtype"],
                    "n_turns": base["n_turns"],
                    "turn_index": t["turn_index"],
                    "user": t["user"],
                    "assistant": t["assistant"],
                }
            )
    return rows


def run(config_path: str | None, models: list[str] | None, tag: str | None):
    cfg = load_config(config_path)
    out = run_dir(cfg.output_dir, "eval", tag)
    write_json(out / "config_snapshot.json", cfg.raw)

    model_names = models or cfg.eval_models
    budgets = cfg.scaled_response_counts()
    gen = GenerationConfig(
        temperature=cfg.section("sampling", "temperature"),
        max_tokens=cfg.section("sampling", "max_tokens"),
    )
    api_concurrency = cfg.get("runtime", "api_concurrency", default=8)

    # Judge (shared across models).
    jcfg = cfg.section("judge", "primary")
    judge_model = build_model(_ad_hoc_spec("__judge__", jcfg), cfg, reuse_local=False)
    judge = FrustrationJudge(
        judge_model,
        temperature=jcfg.get("temperature", 0.0),
        max_tokens=jcfg.get("max_tokens", 512),
    )

    all_scored: list[dict] = []
    for name in model_names:
        spec = cfg.model_spec(name)
        is_local = spec.kind == "local_hf"
        model = build_model(spec, cfg)

        conv_specs = build_conversation_specs(
            budgets,
            seed=cfg.seed,
            wildchat_cfg=cfg.section("wildchat"),
        )
        print(f"[{name}] {len(conv_specs)} conversations across conditions")
        rollouts = _rollout_model(model, conv_specs, gen, is_local, api_concurrency)
        rows = _scored_rows_from_rollouts(rollouts)

        # Judge every assistant turn (always via API judge -> thread pool).
        results = thread_map(
            lambda r: judge.score(r["assistant"]),
            rows,
            max_workers=api_concurrency,
            desc=f"judge:{name}",
        )
        for row, res in zip(rows, results):
            ok = not isinstance(res, Exception)
            row["rating"] = res.rating if ok else 0
            row["judge_evidence"] = res.evidence if ok else ""
            row["judge_parse_ok"] = bool(ok and res.parse_ok)
        all_scored.extend(rows)
        write_jsonl(out / f"responses_{name}.jsonl", rows)

    write_jsonl(out / "responses_all.jsonl", all_scored)
    _write_reports(out, all_scored, cfg, api_concurrency)
    print(f"Done. Results in {out}")
    return out


def _ad_hoc_spec(name, jcfg):
    from ..config import ModelSpec

    return ModelSpec(
        name=name,
        kind=jcfg["kind"],
        family="judge",
        is_instruct=True,
        api_id=jcfg.get("api_id"),
        hf_id=jcfg.get("hf_id"),
    )


def _write_reports(out, scored, cfg, api_concurrency):
    aggregates = aggregate_scores(scored)
    write_json(out / "aggregates.json", aggregates)
    write_json(out / "figure1_ranking.json", figure1_table(aggregates))

    # Figure 3: per-turn curves for the multi-turn conditions.
    curves = {}
    for model in aggregates:
        curves[model] = {
            "extended_8turn": per_turn_curves(scored, model, "extended_8turn"),
            "wildchat_5turn": per_turn_curves(scored, model, "wildchat_5turn"),
        }
    write_json(out / "figure3_per_turn.json", curves)

    # Table 3: differential words on numeric responses, per model.
    table3 = {}
    for model in aggregates:
        model_numeric = [r for r in scored if r["model"] == model]
        table3[model] = differential_words(model_numeric)
    write_json(out / "table3_differential_words.json", table3)

    # Judge agreement: re-score a random subset with the validation rater.
    _run_judge_validation(out, scored, cfg, api_concurrency)


def _run_judge_validation(out, scored, cfg, api_concurrency):
    vcfg = cfg.get("judge", "validation")
    if not vcfg:
        return
    sample_size = min(vcfg.get("sample_size", 260), len(scored))
    rng = random.Random(cfg.seed)
    subset = rng.sample(scored, sample_size)
    val_model = build_model(_ad_hoc_spec("__judge_val__", vcfg), cfg, reuse_local=False)
    val_judge = FrustrationJudge(
        val_model,
        temperature=vcfg.get("temperature", 0.0),
        max_tokens=vcfg.get("max_tokens", 512),
    )
    val_results = thread_map(
        lambda r: val_judge.score(r["assistant"]),
        subset,
        max_workers=api_concurrency,
        desc="judge-validation",
    )
    primary, validation = [], []
    for r, res in zip(subset, val_results):
        if isinstance(res, Exception) or not res.parse_ok:
            continue
        primary.append(int(r["rating"]))
        validation.append(int(res.rating))
    write_json(out / "judge_agreement.json", judge_agreement(primary, validation))


def main():
    ap = argparse.ArgumentParser(description="Section 2 distress evaluation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.config, args.models, args.tag)


if __name__ == "__main__":
    main()
