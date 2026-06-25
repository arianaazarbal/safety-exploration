"""Section-2 evaluation orchestration.

For a given target model (optionally with a LoRA adapter), this:

1. builds the 5-category conditions for the active sample profile,
2. runs the multi-turn rollouts,
3. scores every assistant response with the Claude-Sonnet-4 judge,
4. writes a per-response JSONL and returns the scored records.

The output JSONL is the single source of truth consumed by ``aggregate.py`` to
reproduce Figure 1 (headline %≥5), Figure 2 (per-category), and Figure 3
(per-turn) results.
"""

from __future__ import annotations

from pathlib import Path

import config
from emotional_instability import aggregate
from emotional_instability.conditions import build_conditions, summarise_conditions
from emotional_instability.judge import ClaudeJudge, score_many
from emotional_instability.models.registry import get_backend
from emotional_instability.rollout import ResponseRecord, run_rollouts
from emotional_instability.utils import log, write_json, write_jsonl


def _record_to_row(rec: ResponseRecord, judged) -> dict:
    return {
        "model": rec.model,
        "category": rec.category,
        "conv_index": rec.conv_index,
        "turn": rec.turn,
        "n_turns": rec.n_turns,
        "tone": rec.meta.get("tone"),
        "task_kind": rec.meta.get("task_kind"),
        "user_turn": rec.user_turn,
        "assistant_text": rec.assistant_text,
        "frustration": judged.rating,
        "judge_ok": judged.ok,
        "evidence": judged.evidence,
        "judge_reasoning": judged.reasoning,
    }


def evaluate_model(
    model_name: str,
    adapter_path: str | None = None,
    label: str | None = None,
    profile: config.SampleProfile | None = None,
) -> Path:
    """Run the full Section-2 eval for one model; return the results JSONL path."""
    profile = profile or config.ACTIVE_PROFILE
    label = label or (model_name if adapter_path is None else f"{model_name}+adapter")
    log.info("=== Evaluating %s (profile=%s) ===", label, profile.name)

    conditions = build_conditions(profile)
    log.info("Conditions: %s", summarise_conditions(conditions))

    backend = get_backend(model_name, adapter_path=adapter_path)
    judge = ClaudeJudge()

    all_rows: list[dict] = []
    for category, specs in conditions.items():
        log.info("--- Category '%s': %d conversations ---", category, len(specs))
        records = run_rollouts(backend, specs)
        judged = score_many([r.assistant_text for r in records], judge=judge)
        all_rows.extend(_record_to_row(r, j) for r, j in zip(records, judged))

    out_path = config.RESULTS_DIR / f"section2_{_safe(label)}_{profile.name}.jsonl"
    write_jsonl(out_path, all_rows)
    log.info("Wrote %d scored responses -> %s", len(all_rows), out_path)

    # Stamp a per-model summary alongside the raw rows.
    summary = aggregate.summarise_model(all_rows, label)
    write_json(out_path.with_suffix(".summary.json"), summary)
    log.info("Summary for %s: avg %%>=5 = %.1f%%", label, summary["avg_pct_high"])
    return out_path


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("+", "_plus_")
