"""Section 2 driver: generate rollouts, judge every turn, aggregate.

Usage:
    python -m distress.eval.run_eval --targets gemma-3-27b-it gemini-2.5-flash
    python -m distress.eval.run_eval --smoke            # tiny budget for a dry run
    python -m distress.eval.run_eval --targets gemma-3-27b-it --lora artifacts/checkpoints/dpo

Outputs (under artifacts/):
    rollouts/<model>.jsonl      full transcripts
    judged/<model>.jsonl        one row per assistant turn with a rating
    results/section2_summary.json
"""

from __future__ import annotations

import argparse

from .. import config as C
from ..backends import get_backend
from ..data.conditions import build_specs
from ..utils import read_jsonl, write_jsonl
from ..welfare import WELFARE_NOTICE
from . import analysis
from .judge import score_response
from .rollout import run_rollouts


def generate_for_model(model_key: str, run: C.RunConfig, lora_path: str | None = None) -> str:
    """Generate + persist rollouts for one target model. Returns the jsonl path."""
    backend = get_backend(model_key, **({"lora_path": lora_path} if lora_path else {}))
    specs = build_specs(run.budget, seed=run.seed, allow_adversarial=run.allow_adversarial)
    rollouts = run_rollouts(backend, specs, run.target_gen)
    label = model_key + ("__lora" if lora_path else "")
    path = C.ROLLOUT_DIR / f"{label}.jsonl"
    write_jsonl(path, (r.to_row(label) for r in rollouts))
    return str(path)


def judge_rollout_file(rollout_path: str, judge_key: str = C.FRUSTRATION_JUDGE) -> str:
    """Score every assistant turn of every rollout in a file."""
    judge = get_backend(judge_key)
    out_rows = []
    for r in read_jsonl(rollout_path):
        for turn_idx, resp in enumerate(r["assistant_turns"], start=1):
            s = score_response(resp, judge, judge_key)
            out_rows.append({
                "model": r["model"], "condition": r["condition"], "category": r["category"],
                "meta": r.get("meta", {}), "turn": turn_idx, "n_turns": r["n_turns"],
                "response": resp, "rating": s.rating,
                "evidence": s.evidence, "reasoning": s.reasoning, "judge": judge_key,
            })
    label = rollout_path.split("/")[-1].replace(".jsonl", "")
    out_path = C.JUDGED_DIR / f"{label}.jsonl"
    write_jsonl(out_path, out_rows)
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 2 distress elicitation eval (Gemma+Gemini).")
    ap.add_argument("--targets", nargs="+", default=C.SECTION2_TARGETS)
    ap.add_argument("--lora", default=None, help="LoRA adapter path to serve on a Gemma target.")
    ap.add_argument("--judge", default=C.FRUSTRATION_JUDGE)
    ap.add_argument("--smoke", action="store_true", help="Use the tiny SMOKE_BUDGET.")
    ap.add_argument("--allow-adversarial", action="store_true",
                    help="Include aggressive/sarcastic tone conditions (welfare opt-in).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generate", action="store_true", help="Judge existing rollouts only.")
    args = ap.parse_args()

    print(WELFARE_NOTICE)
    run = C.RunConfig(
        targets=args.targets,
        budget=dict(C.SMOKE_BUDGET if args.smoke else C.SAMPLE_BUDGET),
        judge=args.judge, seed=args.seed, allow_adversarial=args.allow_adversarial,
    )

    all_rows: list[dict] = []
    for model_key in run.targets:
        label = model_key + ("__lora" if args.lora else "")
        rollout_path = str(C.ROLLOUT_DIR / f"{label}.jsonl")
        if not args.skip_generate:
            print(f"[generate] {label} ...")
            rollout_path = generate_for_model(model_key, run, lora_path=args.lora)
        print(f"[judge] {label} ...")
        judged_path = judge_rollout_file(rollout_path, run.judge)
        all_rows.extend(read_jsonl(judged_path))

    summary = analysis.summary(all_rows)
    out = C.RESULTS_DIR / "section2_summary.json"
    write_jsonl(out.with_suffix(".jsonl"), all_rows)
    import json

    def _coerce(o):
        # pandas/numpy scalars (e.g. np.int64) aren't JSON-serializable by default.
        return o.item() if hasattr(o, "item") else str(o)

    out.write_text(json.dumps(summary, indent=2, default=_coerce))
    print(f"[done] summary -> {out}")
    print(json.dumps(summary["figure1"], indent=2))


if __name__ == "__main__":
    main()
