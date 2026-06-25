"""Run capability benchmarks against a target model (Paper §4.2, Figure 7)."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import build_model
from ..models.base import ChatModel
from ..types import Message
from .benchmarks import BenchmarkSpec, default_specs, render_prompt, score_answer


@dataclass
class BenchmarkResult:
    benchmark: str
    model: str
    n: int
    accuracy: float

    def as_dict(self) -> dict:
        return {"benchmark": self.benchmark, "model": self.model,
                "n": self.n, "accuracy": self.accuracy}


def _load_examples(spec: BenchmarkSpec):
    from datasets import load_dataset

    kwargs = {"split": spec.split}
    if spec.config:
        ds = load_dataset(spec.dataset, spec.config, **kwargs)
    else:
        ds = load_dataset(spec.dataset, **kwargs)
    return spec.loader(ds, spec.max_examples)


def run_benchmark(target_name: str, spec: BenchmarkSpec, *, model: ChatModel | None = None) -> BenchmarkResult:
    model = model or build_model(target_name)
    examples = _load_examples(spec)
    correct = 0
    for ex in examples:
        prompt = render_prompt(spec, ex)
        # Capability evals use greedy decoding (temperature 0), unlike the
        # frustration evals which fix temperature=1.
        out = model.generate([Message("user", prompt)], temperature=0.0)
        if score_answer(spec.mode, out, ex.answer):
            correct += 1
    n = len(examples)
    return BenchmarkResult(
        benchmark=spec.name, model=target_name, n=n,
        accuracy=(correct / n) if n else float("nan"),
    )


def run_all_benchmarks(
    target_name: str, specs: list[BenchmarkSpec] | None = None
) -> list[BenchmarkResult]:
    specs = specs or default_specs()
    model = build_model(target_name)
    results: list[BenchmarkResult] = []
    for spec in specs:
        try:
            results.append(run_benchmark(target_name, spec, model=model))
        except Exception as exc:  # noqa: BLE001 - a missing dataset shouldn't abort the suite
            results.append(BenchmarkResult(benchmark=spec.name, model=target_name,
                                           n=0, accuracy=float("nan")))
            print(f"[capabilities] {spec.name} failed: {exc}")
    return results
