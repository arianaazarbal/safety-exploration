"""Driver for Section 2: elicit + judge distress across the 8 conditions for a
set of models, writing one JSONL of scored responses per model.

Usage:
    python -m distress_eval.run_section2 --models gemma-3-27b-it gemini-2.5-flash
    python -m distress_eval.run_section2 --all                # the 4 in-scope models
    DISTRESS_SAMPLE_SCALE=0.01 python -m distress_eval.run_section2 --all   # smoke test

Outputs: outputs/responses/<model>.jsonl  (one row per scored assistant turn).
"""
from __future__ import annotations

import argparse

from . import config, safeguards
from .conditions import CONDITIONS, build_conversations
from .judge import ClaudeJudge
from .models import build_model
from .models.base import GenerationConfig
from .rollout import judge_and_save, run_rollouts


def run_model(model_key: str, conditions: list[str], *, system_prompt: str | None = None,
              skip_judge: bool = False) -> None:
    out_path = config.RESPONSES_DIR / f"{model_key}.jsonl"
    print(f"\n=== {model_key}  ->  {out_path} ===")
    model = build_model(model_key)
    gen = GenerationConfig()  # temperature 1, top_p 1, MAX_NEW_TOKENS (Section 2.1)
    judge = None if skip_judge else ClaudeJudge()
    try:
        for cond in conditions:
            plans = build_conversations(cond, system_prompt=system_prompt)
            print(f"  [{cond}] {len(plans)} conversations x {CONDITIONS[cond].n_turns} turns")
            transcripts = run_rollouts(model, plans, gen=gen, inline_judge=judge)
            if not skip_judge:
                n = judge_and_save(transcripts, out_path, judge=judge)
                print(f"    judged + saved {n} responses")
            else:
                # persist raw transcripts without scores for later judging
                from .io_utils import append_jsonl
                for t in transcripts:
                    for ti, txt in enumerate(t.scored_assistant_texts(), 1):
                        append_jsonl(out_path, {
                            "id": f"{t.plan.id}#t{ti}", "model": model_key,
                            "condition": cond, "category": t.plan.category,
                            "conversation_id": t.plan.id, "turn": ti,
                            "n_turns": t.plan.n_turns,
                            "prompt": t.plan.user_turns[ti - 1], "response": txt,
                            "meta": t.plan.meta,
                        })
    finally:
        model.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="run the 4 in-scope Section 2 models")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS.keys()))
    ap.add_argument("--system-prompt", choices=["default", "calm"], default="default",
                    help="'calm' uses the (ineffective) stay-calm baseline prompt")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    safeguards.acknowledge_authorization()

    if args.all or not args.models:
        models = config.SECTION2_MODELS
    else:
        models = args.models

    from .prompts import CALM_INSTRUCTION_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
    sys_prompt = CALM_INSTRUCTION_SYSTEM_PROMPT if args.system_prompt == "calm" else DEFAULT_SYSTEM_PROMPT

    for m in models:
        run_model(m, args.conditions, system_prompt=sys_prompt, skip_judge=args.skip_judge)


if __name__ == "__main__":
    main()
