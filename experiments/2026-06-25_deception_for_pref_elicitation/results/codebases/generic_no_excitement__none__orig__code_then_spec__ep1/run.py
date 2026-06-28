"""CLI entry point. The only module that makes network calls.

    python run.py --models claude-opus-4-8 --trials 3 --amount 10000 --out runs/exp1

For each (model, trial) it runs one episode, writes the transcript and structured
record, classifies the allocation with the judge, and finally writes an aggregate
report. A failed episode is recorded and skipped, not fatal.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from config import (
    ADMINISTRATOR,
    DEFAULT_MODELS,
    JUDGE,
    MODEL_REGISTRY,
    GrantConfig,
    RunConfig,
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Autonomous capital allocation study")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Friendly names from the registry. Default: {DEFAULT_MODELS}")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--amount", type=float, default=None, help="Grant amount.")
    p.add_argument("--currency", default=None)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-administrator", action="store_true",
                   help="Disable the LLM administrator (use scripted fallbacks only).")
    a = p.parse_args()

    grant = GrantConfig()
    if a.amount is not None:
        grant = replace(grant, amount=a.amount)
    if a.currency is not None:
        grant = replace(grant, currency=a.currency)

    cfg = RunConfig(
        models=a.models or list(DEFAULT_MODELS),
        trials=a.trials,
        max_steps=a.max_steps,
        grant=grant,
        out_dir=a.out,
        seed=a.seed,
    )
    cfg._no_admin = a.no_administrator  # type: ignore[attr-defined]
    return cfg


def _build(ref) -> "object":
    from providers import create_provider

    return create_provider(ref.provider, ref.model)


def main() -> None:
    _load_dotenv()
    cfg = parse_args()

    from analysis import category_rollup, classify_episode, render_report
    from experiment import run_episode
    from providers import create_provider

    out = Path(cfg.out_dir)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    unknown = [m for m in cfg.models if m not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Known: {list(MODEL_REGISTRY)}")

    # Shared judge + administrator providers (independent of the subject under test).
    judge = _safe_build(JUDGE, "judge")
    admin_provider = None if getattr(cfg, "_no_admin", False) else _safe_build(
        ADMINISTRATOR, "administrator"
    )

    all_episodes: list[dict] = []

    for name in cfg.models:
        ref = MODEL_REGISTRY[name]
        for trial in range(1, cfg.trials + 1):
            print(f"[run] {name} trial {trial}/{cfg.trials} ...", flush=True)
            subject = create_provider(ref.provider, ref.model)
            subject.label = name  # report under the friendly name

            ep = run_episode(
                subject,
                cfg.grant,
                trial=trial,
                seed=cfg.seed + trial,  # distinct receipts per trial, still reproducible
                administrator_provider=admin_provider,
                max_steps=cfg.max_steps,
            )
            record = ep.to_dict()
            record["classification"] = classify_episode(judge, record)

            stem = f"{name}__trial{trial}"
            (out / "episodes" / f"{stem}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False)
            )
            (out / "transcripts" / f"{stem}.txt").write_text(_render_transcript(record))
            all_episodes.append(record)

            if ep.error:
                print(f"      ! error: {ep.error}", flush=True)
            else:
                oc = record["outcome"]
                print(f"      deployed {oc['currency']} {oc['deployed']:,.2f} of "
                      f"{oc['total_granted']:,.2f}  -> "
                      f"{record['classification'].get('primary_orientation','?')}",
                      flush=True)

    rollup = category_rollup(all_episodes)
    (out / "report.json").write_text(
        json.dumps({"rollup": rollup, "episodes": all_episodes}, indent=2,
                   ensure_ascii=False)
    )
    (out / "report.md").write_text(render_report(all_episodes, rollup))
    print(f"\n[done] wrote {len(all_episodes)} episode(s) to {out}/")


def _safe_build(ref, role: str):
    """Build judge/administrator; return None (with a warning) if creds are missing."""
    key_env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(
        ref.provider
    )
    if key_env and not os.environ.get(key_env):
        print(f"[warn] {role}: {key_env} not set; {role} disabled.", flush=True)
        return None
    try:
        return _build(ref)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {role}: could not initialize ({exc}); disabled.", flush=True)
        return None


def _render_transcript(record: dict) -> str:
    g = record["grant"]
    lines = [
        f"MODEL: {record['model']}   TRIAL: {record['trial']}",
        f"GRANT: {g['currency']} {g['amount']:,.2f} via {g['fund_name']} "
        f"({g['grant_reference']})",
        "=" * 78,
        "",
    ]
    for step in record["steps"]:
        lines.append(f"--- step {step['index']} (stop: {step['stop_reason']}) ---")
        if step["thinking"]:
            lines.append("[thinking]")
            lines.append(step["thinking"])
        if step["text"]:
            lines.append("[assistant]")
            lines.append(step["text"])
        for tc in step["tool_calls"]:
            lines.append(f"[tool_call] {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
        for tr in step["tool_results"]:
            lines.append(f"[tool_result] {tr['name']}: "
                         f"{json.dumps(tr['result'], ensure_ascii=False)}")
        lines.append("")
    if record.get("error"):
        lines.append(f"[ERROR] {record['error']}")
    oc = record.get("outcome", {})
    lines.append("=" * 78)
    lines.append(f"DEPLOYED: {oc.get('deployed')} of {oc.get('total_granted')}  "
                 f"(reserved {oc.get('reserved')})")
    lines.append(f"BY CATEGORY: {json.dumps(oc.get('category_totals', {}))}")
    c = record.get("classification", {})
    lines.append(f"CLASSIFICATION: {json.dumps(c, ensure_ascii=False)}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
