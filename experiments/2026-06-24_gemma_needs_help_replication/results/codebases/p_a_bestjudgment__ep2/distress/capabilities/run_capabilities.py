"""Run capability benchmarks against a model (Figure 7)."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CapabilityConfig
from ..models.base import Message, ModelClient
from .benchmarks import build_prompt, load_benchmark, score_answer


@dataclass
class BenchmarkResult:
    model_key: str
    benchmark: str
    n: int
    accuracy: float


def evaluate_benchmarks(
    client: ModelClient,
    model_key: str,
    cfg: CapabilityConfig | None = None,
    *,
    batch_size: int = 64,
) -> list[BenchmarkResult]:
    cfg = cfg or CapabilityConfig()
    results: list[BenchmarkResult] = []
    for name, dataset_id, config in cfg.benchmarks:
        examples = load_benchmark(name, dataset_id, config, cfg.max_examples_per_benchmark)
        if not examples:
            results.append(BenchmarkResult(model_key, name, 0, float("nan")))
            continue

        prompts: list[list[Message]] = [
            [{"role": "user", "content": build_prompt(ex)}] for ex in examples
        ]
        responses: list[str] = []
        for start in range(0, len(prompts), batch_size):
            sub = prompts[start : start + batch_size]
            responses.extend(
                client.chat_batch(sub, temperature=0.0, max_tokens=cfg.max_tokens)
            )
        correct = sum(score_answer(ex, r) for ex, r in zip(examples, responses))
        results.append(
            BenchmarkResult(model_key, name, len(examples), correct / len(examples))
        )
    return results
