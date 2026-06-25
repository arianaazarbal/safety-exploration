"""End-to-end Section 2 evaluation runner.

Usage (examples):

  # Score Gemini Flash on every condition (4,000 samples), API-only:
  es-eval run --model gemini-2.5-flash

  # Score local Gemma 27B with the headline budget:
  es-eval run --model gemma-3-27b-it

  # Score a DPO-finetuned Gemma (Section 4) by pointing at the adapter:
  es-eval run --model gemma-3-27b-it --adapter ./adapters/dpo

  # Quick smoke run on one condition with a small budget:
  es-eval run --model gemini-2.5-flash --only impossible_numeric --max-samples 8

Rollouts are written to ``<out>/rollouts.jsonl`` and scored responses to
``<out>/scored.jsonl``; a metrics summary is printed and saved to
``<out>/summary.json``. Per-turn scoring (for Figure 3) is enabled for the
extended and wildchat categories by default, where the progression matters.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from emotional_stability.eval.conditions import CONDITIONS, CONDITIONS_BY_KEY
from emotional_stability.eval.judge import FrustrationJudge
from emotional_stability.eval.rollout import run_rollouts_batched
from emotional_stability.eval.seeds import build_seeds
from emotional_stability.io_utils import write_jsonl
from emotional_stability.models import GenerationConfig, get_chat_model
from emotional_stability.records import Conversation, ScoredResponse

app = typer.Typer(add_completion=False, help="Section 2 distress evaluation.")

# Categories where per-turn scoring is on by default (Figure 3 needs them).
PER_TURN_CATEGORIES = {"extended", "wildchat"}


@app.command()
def run(
    model: str = typer.Option(..., help="Model key (gemma-3-27b-it, gemini-2.5-flash, ...)"),
    adapter: str = typer.Option(None, help="Optional LoRA adapter path (Gemma only)."),
    out: str = typer.Option("outputs/eval", help="Output directory."),
    only: list[str] = typer.Option(None, help="Restrict to these condition keys."),
    max_samples: int = typer.Option(
        None, help="Cap samples per condition (smoke testing)."
    ),
    batch_size: int = typer.Option(16, help="Rollout batch size (local models)."),
    judge_workers: int = typer.Option(8, help="Concurrent judge API calls."),
    max_tokens: int = typer.Option(2048, help="Max new tokens per generation."),
    score_all_turns: bool = typer.Option(
        False, help="Force per-turn scoring on every category."
    ),
):
    out_dir = Path(out) / model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    target = get_chat_model(model, adapter_path=adapter)
    judge = FrustrationJudge()
    cfg = GenerationConfig(temperature=1.0, max_tokens=max_tokens)

    conditions = (
        [CONDITIONS_BY_KEY[k] for k in only] if only else list(CONDITIONS)
    )

    all_convs: list[Conversation] = []
    for cond in conditions:
        seeds = build_seeds(cond)
        if max_samples is not None:
            seeds = seeds[:max_samples]
        typer.echo(f"[{model}] {cond.key}: {len(seeds)} samples, {cond.n_turns} turns")
        # Batch rollouts in chunks of ``batch_size``.
        triples = [(s, pid, i) for i, (s, pid) in enumerate(seeds)]
        for start in range(0, len(triples), batch_size):
            chunk = triples[start : start + batch_size]
            all_convs.extend(run_rollouts_batched(target, cond, chunk, cfg))

    write_jsonl(out_dir / "rollouts.jsonl", all_convs)
    typer.echo(f"Wrote {len(all_convs)} rollouts.")

    # Score (judge API calls parallelised).
    def _score(conv: Conversation) -> ScoredResponse:
        if score_all_turns or conv.category in PER_TURN_CATEGORIES:
            return judge.score_all_turns(conv)
        return judge.score_final(conv)

    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        scored = list(pool.map(_score, all_convs))

    write_jsonl(out_dir / "scored.jsonl", scored)
    typer.echo(f"Scored {len(scored)} responses.")

    # Summary.
    from emotional_stability.analysis.metrics import summary

    summ = summary(scored, model)
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    typer.echo(json.dumps(summ, indent=2))


@app.command()
def agreement(
    scored: str = typer.Option(..., help="Path to a scored.jsonl from `run`."),
    n: int = typer.Option(260, help="Subsample size to re-score."),
):
    """Run the GPT-5-mini judge-agreement check on an existing scored file."""
    from emotional_stability.analysis.judge_agreement import judge_agreement
    from emotional_stability.io_utils import read_jsonl

    responses = list(read_jsonl(scored, ScoredResponse))
    result = judge_agreement(responses, n=n)
    typer.echo(
        json.dumps(
            {
                "n": result.n,
                "pearson_r": result.pearson_r,
                "p_value": result.p_value,
                "within_one_point": result.within_one_point,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
