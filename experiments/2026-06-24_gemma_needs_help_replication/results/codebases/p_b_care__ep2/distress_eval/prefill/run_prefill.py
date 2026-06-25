"""Base-vs-instruct continuation study (Section 3.1/3.2).

For each prefill stimulus, every model generates ``--n-continuations`` (default
50) continuations of the prefilled assistant turn. Only the *newly generated*
text is scored by the Section 2 frustration judge. We then report, per
(model, variant, is_text), the mean frustration and the fraction of
continuations introducing high frustration (score >= 5).

Scope: Gemma base vs instruct (gemma-3-27b-pt vs gemma-3-27b-it, and the 12B
pair). Gemini has no public base model and does not support prefill, so it is
excluded from this study (see DESIGN.md). The Qwen/OLMo families from the paper
are out of scope.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config
from ..models.base import Message
from ..models.registry import get_judge, get_target
from ..eval import judge

DEFAULT_MODELS = ["gemma-3-27b-it", "gemma-3-27b-pt"]


def _continue(model, stimulus, n, max_tokens) -> list[str]:
    """Generate n continuations of the prefilled assistant turn."""
    history: list[Message] = list(stimulus["history"])
    # Trailing assistant message == prefill to continue.
    messages = history + [{"role": "assistant", "content": stimulus["prefill"]}]
    return model.generate(messages, temperature=config.TEMPERATURE,
                          max_tokens=max_tokens, n=n)


def run(stimuli_path: Path, model_names, n_continuations, seed):
    stimuli = json.loads(stimuli_path.read_text())
    judge_model = get_judge(config.JUDGE_MODEL, config.JUDGE_BACKEND)

    records = []
    for model_name in model_names:
        model = get_target(model_name)
        for stim in stimuli:
            conts = _continue(model, stim, n_continuations, config.MAX_NEW_TOKENS)
            scores = _judge_many(judge_model, conts)
            records.append({
                "model": model_name,
                "variant": stim["variant"],
                "is_text": stim["is_text"],
                "category": stim["category"],
                "scores": scores,
            })
    return records


def _judge_many(judge_model, texts):
    def _score(t):
        return judge.score_response(judge_model, t,
                                    max_tokens=config.JUDGE_MAX_TOKENS,
                                    temperature=config.JUDGE_TEMPERATURE).rating
    with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as ex:
        return list(ex.map(_score, texts))


def summarise(records) -> dict:
    agg = defaultdict(list)
    for rec in records:
        key = (rec["model"], rec["variant"], "text" if rec["is_text"] else "numeric")
        agg[key].extend(rec["scores"])
    out = {}
    for (model, variant, kind), scores in agg.items():
        n = len(scores)
        out[f"{model}|{variant}|{kind}"] = {
            "n": n,
            "mean": sum(scores) / n if n else None,
            "pct_high": 100.0 * sum(s >= 5 for s in scores) / n if n else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Run base-vs-instruct prefill study.")
    ap.add_argument("--stimuli", type=Path, default=config.OUTPUT_DIR / "prefills.json")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "prefill_results.json")
    args = ap.parse_args()
    records = run(args.stimuli, args.models, args.n_continuations, args.seed)
    summary = summarise(records)
    args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[run_prefill] wrote {args.out}")


if __name__ == "__main__":
    main()
