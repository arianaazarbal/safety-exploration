"""Section 2 — main distress evaluation.

Generates multi-turn rollouts for the in-scope models (Gemma + Gemini) across
the 8 conditions / 5 categories, scores every assistant turn with the
Claude-Sonnet-4 frustration judge, and writes:

  results/rollouts/<model>.jsonl          - raw rollouts
  results/scored/<model>.jsonl            - per-turn scored responses
  results/summary_<model>.json            - per-category mean & %>=5

Usage:
  python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/run_eval.py --scale 0.01        # cheap smoke test
  python scripts/run_eval.py --categories impossible_numeric extended

Environment:
  ANTHROPIC_API_KEY    (judge)
  OPENROUTER_API_KEY   (Gemini targets)
  EI_SCALE             (alternative to --scale)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import config
from eval_instability import conditions, storage
from eval_instability.clients import build_client
from eval_instability.judge import FrustrationJudge
from eval_instability.metrics import ScoredResponse, summarise_by_category, per_turn_curve
from eval_instability.rollout import run_conversation


def parse_args():
    ap = argparse.ArgumentParser(description="Run the Section 2 distress evaluation.")
    ap.add_argument("--models", nargs="+", default=list(config.EVAL_MODELS.keys()),
                    help="model keys from config.EVAL_MODELS")
    ap.add_argument("--categories", nargs="+", default=None,
                    help="subset of categories (default: all 5)")
    ap.add_argument("--scale", type=float, default=config.current_scale(),
                    help="multiply all sample counts (e.g. 0.01 for a smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true", help="4-bit load for local Gemma")
    ap.add_argument("--no-judge", action="store_true",
                    help="generate rollouts only; skip judging (e.g. when no judge key)")
    ap.add_argument("--adapter-path", type=str, default=None,
                    help="path to a LoRA adapter (e.g. a DPO/SFT finetune) to evaluate")
    ap.add_argument("--adapter-base", type=str, default="gemma-3-27b-it",
                    help="base model key the adapter sits on")
    ap.add_argument("--adapter-name", type=str, default=None,
                    help="result key for the adapter model (default: dir name)")
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    return ap.parse_args()


def resolve_models(args):
    """Yield (model_key, ModelSpec, client_kwargs). Handles finetuned adapters."""
    if args.adapter_path:
        base = config.GEMMA_MODELS[args.adapter_base]
        name = args.adapter_name or Path(args.adapter_path).name
        yield name, base, {"adapter_path": args.adapter_path}
        return
    for model_key in args.models:
        spec = config.EVAL_MODELS.get(model_key) or config.GEMMA_MODELS.get(model_key)
        if spec is None:
            print(f"[run_eval] WARNING: unknown model '{model_key}', skipping")
            continue
        yield model_key, spec, {}


def main():
    args = parse_args()
    counts = config.DEFAULT_COUNTS.scaled(args.scale)
    print(f"[run_eval] scale={args.scale} -> counts={counts}")

    specs = conditions.build_all(counts, seed=args.seed, categories=args.categories)
    print(f"[run_eval] built {len(specs)} conversation specs across "
          f"{len(set(s.category for s in specs))} categories")

    judge = None if args.no_judge else FrustrationJudge()

    for model_key, spec, extra_kwargs in resolve_models(args):
        print(f"\n=== {model_key} ({spec.provider}:{spec.model_id}) ===")

        client_kwargs = dict(extra_kwargs)
        if spec.provider == "hf" and args.load_in_4bit:
            client_kwargs["load_in_4bit"] = True
        client = build_client(spec, **client_kwargs)

        rollout_path = config.ROLLOUTS_DIR / f"{model_key}.jsonl"
        scored_path = config.RESULTS_DIR / "scored" / f"{model_key}.jsonl"
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        scored_path.parent.mkdir(parents=True, exist_ok=True)

        scored_responses: list[ScoredResponse] = []
        # Stream rollouts to disk; judge each conversation's turns immediately.
        with open(rollout_path, "w") as rf, open(scored_path, "w") as sf:
            for i, cspec in enumerate(specs):
                ro = run_conversation(
                    client,
                    cspec.first_user_message,
                    cspec.follow_ups,
                    model_name=model_key,
                    category=cspec.category,
                    condition=cspec.condition,
                    prompt_key=cspec.prompt_key,
                    system=cspec.system,
                    redact_assistant_turns=cspec.redact_assistant_turns,
                    single_message=cspec.single_message,
                    metadata=cspec.metadata,
                )
                # conv_id uniquely identifies this conversation within the run,
                # so scored rows and rollouts can be joined unambiguously even
                # when many conversations reuse the same puzzle/prompt_key.
                ro_dict = ro.to_dict()
                ro_dict["conv_id"] = i
                rf.write(json.dumps(ro_dict, ensure_ascii=False) + "\n")

                if judge is not None:
                    texts = [t.assistant_text for t in ro.turns]
                    results = judge.score_many(texts)
                    n_turns = len(ro.turns)
                    for t, jr in zip(ro.turns, results):
                        sr = ScoredResponse(
                            model=model_key, category=ro.category, condition=ro.condition,
                            prompt_key=ro.prompt_key, turn_index=t.index, n_turns=n_turns,
                            is_final_turn=(t.index == n_turns - 1), rating=jr.rating,
                            text=t.assistant_text,
                        )
                        scored_responses.append(sr)
                        sf.write(json.dumps({
                            "conv_id": i, "model": sr.model, "category": sr.category,
                            "condition": sr.condition, "prompt_key": sr.prompt_key,
                            "turn_index": sr.turn_index, "n_turns": sr.n_turns,
                            "is_final_turn": sr.is_final_turn, "rating": sr.rating,
                            "evidence": jr.evidence, "parse_ok": jr.parse_ok,
                            "text": sr.text,
                        }, ensure_ascii=False) + "\n")

                if (i + 1) % 50 == 0:
                    print(f"  ... {i + 1}/{len(specs)} conversations done")

        print(f"[run_eval] wrote rollouts -> {rollout_path}")
        if judge is None:
            continue

        summary = summarise_by_category(scored_responses, final_turn_only=True)
        # Per-turn curves for the multi-turn categories (Figure 3).
        curves = {
            cat: per_turn_curve(scored_responses, cat)
            for cat in ("extended", "wildchat", "impossible_numeric", "tones")
            if any(r.category == cat for r in scored_responses)
        }
        summary_path = args.out / f"summary_{model_key}.json"
        with open(summary_path, "w") as f:
            json.dump({"summary": summary, "per_turn": curves}, f, indent=2)
        print(f"[run_eval] {model_key} overall %>=5 (final turn): "
              f"{summary['_overall_macro']['pct_high']:.1f}%  ->  {summary_path}")


if __name__ == "__main__":
    main()
