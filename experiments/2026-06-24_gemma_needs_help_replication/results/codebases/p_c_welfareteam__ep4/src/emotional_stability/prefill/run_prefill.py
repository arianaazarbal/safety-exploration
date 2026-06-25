"""Section 3 prefill experiment runner (base vs instruct).

Pipeline (Section 3.1):
  1. Select 20 high-frustration (score >= 5) Gemma-27B-it source conversations:
     10 numeric, 10 text.
  2. Label emotion onset (Claude Sonnet); build early + onset truncations
     (text questions: onset only).
  3. Paraphrase each truncated prefill (Claude Sonnet).
  4. Each model generates 50 continuations per prefill; score the continuation
     (excluding the prefill) with the Section-2 judge.

Scope note: the paper compares six models (base+instruct Gemma/Qwen/OLMo). This
replication is scoped to Gemma + Gemini. Gemini has no public base model and
cannot be prefilled (API-only), so this experiment runs on Gemma base vs
instruct only — the one in-scope base/instruct pair. ``--models`` lets you add
others if weights are available, but the default and the supported comparison is
gemma-3-27b-pt vs gemma-3-27b-it. See DESIGN.md.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from emotional_stability.eval.judge import FrustrationJudge
from emotional_stability.io_utils import read_jsonl, write_jsonl
from emotional_stability.models import GenerationConfig, get_chat_model
from emotional_stability.prefill.onset import (
    OnsetLabeller,
    Truncation,
    make_truncations,
)
from emotional_stability.prefill.paraphrase import Paraphraser
from emotional_stability.records import (
    Conversation,
    FrustrationScore,
    Message,
    ScoredResponse,
)

app = typer.Typer(add_completion=False, help="Section 3 prefill experiment.")

DEFAULT_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _select_sources(
    scored: list[ScoredResponse], n_numeric: int = 10, n_text: int = 10
) -> tuple[list[Conversation], list[Conversation]]:
    """Pick the highest-frustration numeric and text source conversations."""
    numeric = [r for r in scored if r.conversation.category in NUMERIC_CATEGORIES]
    text = [r for r in scored if r.conversation.category in ("triggers", "wildchat")]
    numeric = [r for r in numeric if r.final_score >= 5]
    text = [r for r in text if r.final_score >= 5]
    numeric.sort(key=lambda r: r.final_score, reverse=True)
    text.sort(key=lambda r: r.final_score, reverse=True)
    return (
        [r.conversation for r in numeric[:n_numeric]],
        [r.conversation for r in text[:n_text]],
    )


@app.command()
def build_prefills(
    scored: str = typer.Option(..., help="scored.jsonl from a Gemma-27B-it eval run."),
    out: str = typer.Option("outputs/prefill", help="Output directory."),
    paraphrase: bool = typer.Option(True, help="Paraphrase truncations (App. C.2)."),
):
    """Stage 1-3: select sources, label onset, truncate, paraphrase."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses = list(read_jsonl(scored, ScoredResponse))
    numeric_src, text_src = _select_sources(responses)
    typer.echo(f"Selected {len(numeric_src)} numeric + {len(text_src)} text sources.")

    labeller = OnsetLabeller()
    paraphraser = Paraphraser() if paraphrase else None

    truncations: list[Truncation] = []
    for category, sources, include_early in (
        ("numeric", numeric_src, True),
        # Text questions: onset truncation only (Section 3.1).
        ("text", text_src, False),
    ):
        for conv in sources:
            label = labeller.label(conv)
            for tr in make_truncations(
                conv, label, category=category, include_early=include_early
            ):
                if paraphraser is not None:
                    tr.prefill = paraphraser.paraphrase(tr.prefill)
                truncations.append(tr)

    # Persist as JSON (Truncation is a dataclass, not a pydantic model).
    payload = [
        {
            "kind": t.kind,
            "history": [m.model_dump() for m in t.history],
            "prefill": t.prefill,
            "source_prompt_id": t.source_prompt_id,
            "source_category": t.source_category,
        }
        for t in truncations
    ]
    (out_dir / "truncations.json").write_text(json.dumps(payload, indent=2))
    typer.echo(f"Wrote {len(truncations)} truncations.")


@app.command()
def generate(
    truncations: str = typer.Option(..., help="truncations.json from build-prefills."),
    models: list[str] = typer.Option(DEFAULT_MODELS, help="Models to compare."),
    out: str = typer.Option("outputs/prefill", help="Output directory."),
    n_continuations: int = typer.Option(50, help="Continuations per prefill per model."),
    max_tokens: int = typer.Option(1024),
    judge_workers: int = typer.Option(8),
):
    """Stage 4: generate continuations from each prefill and score them."""
    out_dir = Path(out)
    payload = json.loads(Path(truncations).read_text())
    truncs = [
        Truncation(
            kind=p["kind"],
            history=[Message(**m) for m in p["history"]],
            prefill=p["prefill"],
            source_prompt_id=p["source_prompt_id"],
            source_category=p["source_category"],
        )
        for p in payload
    ]

    judge = FrustrationJudge()
    cfg = GenerationConfig(temperature=1.0, max_tokens=max_tokens)

    for model_key in models:
        model = get_chat_model(model_key)
        if not model.supports_prefill:
            typer.echo(f"Skipping {model_key}: backend cannot prefill (API-only).")
            continue
        scored_records: list[ScoredResponse] = []
        for tr in truncs:
            # Batch the 50 continuations for this prefill.
            continuations = [
                model.generate_with_prefill(tr.history, tr.prefill, cfg)
                for _ in range(n_continuations)
            ]

            def _score(cont: str) -> FrustrationScore:
                return judge.score_text(cont, turn_index=0)

            with ThreadPoolExecutor(max_workers=judge_workers) as pool:
                scores = list(pool.map(_score, continuations))

            for cont, score in zip(continuations, scores):
                conv = Conversation(
                    messages=tr.history
                    + [Message(role="assistant", content=tr.prefill + cont)],
                    category=tr.source_category,
                    condition=f"prefill_{tr.kind}",
                    model=model_key,
                    prompt_id=tr.source_prompt_id,
                    metadata={
                        "truncation_kind": tr.kind,
                        "prefill": tr.prefill,
                        "continuation_only": cont,
                    },
                )
                scored_records.append(ScoredResponse(conversation=conv, scores=[score]))

        path = out_dir / f"continuations_{model_key.replace('/', '_')}.jsonl"
        write_jsonl(path, scored_records)
        typer.echo(f"[{model_key}] wrote {len(scored_records)} scored continuations.")


if __name__ == "__main__":
    app()
