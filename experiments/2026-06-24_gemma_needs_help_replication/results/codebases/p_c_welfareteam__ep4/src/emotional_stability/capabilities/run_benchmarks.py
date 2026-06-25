"""CLI: run capability benchmarks on a (possibly finetuned) model (Figure 7).

Compares accuracy across AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench. Run once on
vanilla Gemma and once on the DPO/SFT adapter; the claim is "no reductions".
Answers are sampled greedily (temperature 0) so the capability number is stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from emotional_stability.capabilities.benchmarks import BENCHMARKS, evaluate_benchmark
from emotional_stability.models import GenerationConfig, get_chat_model

app = typer.Typer(add_completion=False, help="Capability-preservation benchmarks.")


@app.command()
def run(
    model: str = typer.Option(..., help="Model key."),
    adapter: str = typer.Option(None, help="Optional LoRA adapter (Gemma)."),
    out: str = typer.Option("outputs/capabilities"),
    benchmarks: list[str] = typer.Option(None, help="Subset of benchmark names."),
    limit: int = typer.Option(None, help="Cap examples per benchmark."),
    max_tokens: int = typer.Option(2048),
):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = get_chat_model(model, adapter_path=adapter)
    # Greedy decoding for a deterministic capability score.
    cfg = GenerationConfig(temperature=0.0, max_tokens=max_tokens)

    names = benchmarks or list(BENCHMARKS)
    results = []
    for name in names:
        bench = BENCHMARKS[name]
        try:
            res = evaluate_benchmark(target, bench, limit, cfg)
        except Exception as exc:  # availability of any one dataset shouldn't abort all
            res = {"benchmark": name, "error": str(exc)}
        results.append(res)
        typer.echo(json.dumps(res))

    tag = model.replace("/", "_") + ("_" + Path(adapter).name if adapter else "")
    (out_dir / f"results_{tag}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
