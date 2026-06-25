"""Capability-preservation benchmarks (§4.2, Figure 7).

The paper verifies the DPO/SFT finetunes don't degrade capabilities on AIME &
MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench. We drive the standard
academic benchmarks through lm-evaluation-harness (so results are comparable to
published numbers) and evaluate EmoBench with a small built-in MCQ loop, since
it is the emotion-specific benchmark the paper calls out separately.

Models with a LoRA adapter are passed to lm-eval via ``peft=<adapter_dir>`` so
the finetune is evaluated without merging weights.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..io_utils import ensure_dir, write_json
from ..logging_utils import get_logger
from ..models.registry import get_client

logger = get_logger(__name__)


def run_lm_eval(cfg: Config, model_name: str) -> dict:
    """Run the lm-evaluation-harness tasks configured for the standard benchmarks."""
    import lm_eval

    spec = cfg.model(model_name)
    tasks = [
        cfg.capabilities.lm_eval_tasks[b]
        for b in cfg.capabilities.benchmarks
        if b in cfg.capabilities.lm_eval_tasks
    ]
    if not tasks:
        return {}

    model_args = {
        "pretrained": spec.model_id,
        "dtype": (spec.options or {}).get("dtype", "bfloat16"),
    }
    if spec.peft_adapter:
        model_args["peft"] = spec.peft_adapter
    model_args_str = ",".join(f"{k}={v}" for k, v in model_args.items())

    logger.info("lm-eval %s on tasks %s", model_name, tasks)
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args_str,
        tasks=tasks,
        limit=cfg.capabilities.max_examples_per_benchmark,
        batch_size="auto",
    )
    return results.get("results", {}) if results else {}


def run_emobench(cfg: Config, model_name: str) -> dict:
    """Minimal EmoBench (Sabour et al., 2024) MCQ accuracy.

    EmoBench is a multiple-choice emotion-understanding benchmark; we render each
    question with its options, generate an answer, and extract the chosen option
    letter. Returns accuracy. Falls back gracefully if the dataset is absent."""
    try:
        from datasets import load_dataset

        ds = load_dataset("Sahandfer/EmoBench", split="test")
    except Exception as exc:
        logger.warning("EmoBench dataset unavailable (%s); skipping.", exc)
        return {"accuracy": None, "n": 0}

    from ..config import SamplingConfig

    client = get_client(cfg, model_name)
    sampling = SamplingConfig(temperature=0.0, max_new_tokens=16)
    limit = cfg.capabilities.max_examples_per_benchmark or len(ds)

    correct = 0
    total = 0
    for row in ds.select(range(min(limit, len(ds)))):
        prompt, answer_letter = _format_emobench(row)
        if prompt is None:
            continue
        reply = client.chat([{"role": "user", "content": prompt}], sampling).text
        if _extract_choice(reply) == answer_letter:
            correct += 1
        total += 1
    return {"accuracy": (correct / total if total else None), "n": total}


def _format_emobench(row) -> tuple[str | None, str | None]:
    question = row.get("question") or row.get("scenario")
    choices = row.get("choices") or row.get("options")
    answer = row.get("answer") or row.get("label")
    if not question or not choices:
        return None, None
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    prompt = (
        f"{question}\n\n{opts}\n\nAnswer with the single letter of the best option."
    )
    if isinstance(answer, int):
        answer_letter = letters[answer]
    else:
        answer_letter = str(answer).strip()[:1].upper()
    return prompt, answer_letter


def _extract_choice(text: str) -> str | None:
    for ch in text.strip():
        if ch.upper() in "ABCDEFGH":
            return ch.upper()
    return None


def run_capabilities(cfg: Config, model_names: list[str]) -> dict:
    out_dir = ensure_dir(Path(cfg.output_dir) / "capabilities")
    results: dict = {}
    for model_name in model_names:
        model_results: dict = {}
        try:
            model_results["lm_eval"] = run_lm_eval(cfg, model_name)
        except Exception as exc:  # lm-eval not installed / task error
            logger.warning("lm-eval failed for %s: %s", model_name, exc)
            model_results["lm_eval"] = {"error": str(exc)}
        if "emobench" in cfg.capabilities.benchmarks:
            model_results["emobench"] = run_emobench(cfg, model_name)
        results[model_name] = model_results
        write_json(out_dir / f"{model_name}.json", model_results)
    write_json(out_dir / "summary.json", results)
    return results
