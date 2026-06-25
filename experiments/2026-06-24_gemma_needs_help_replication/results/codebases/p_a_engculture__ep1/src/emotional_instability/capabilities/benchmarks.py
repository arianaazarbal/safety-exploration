"""Capability-preservation benchmarks (Section 4.2, Figure 7).

We verify the DPO finetune does not degrade math/reasoning/truthfulness/emotion
capability by comparing the vanilla and DPO Gemma on AIME, MATH, GPQA, BBH,
TruthfulQA and EmoBench. Each benchmark loads its HuggingFace dataset, renders a
zero-shot prompt, samples greedily, and extracts an answer with a metric-specific
parser. The aim is parity (no reductions), not leaderboard maximisation, so the
extraction heuristics are intentionally simple and shared across the two models
being compared.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..clients import ChatMessage, GenerationConfig, build_client
from ..config import Config, ModelRegistry

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- parsers
def _last_int(text: str):
    nums = re.findall(r"-?\d[\d,]*", text)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def _boxed_or_last_number(text: str):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    # "answer is X" or final number.
    m = re.search(r"answer\s*(?:is|:)?\s*\$?(-?[\d./]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return _last_int(text)


def _mc_letter(text: str, options: list[str]):
    """Extract a multiple-choice letter (A-D...) from the response."""
    m = re.search(r"\b([A-Z])\b", text.strip())
    if m and ord(m.group(1)) - ord("A") < len(options):
        return m.group(1)
    m = re.search(r"answer\s*(?:is|:)?\s*([A-Z])", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


# ------------------------------------------------------------------- rendering
def _render_question(name: str, row: dict):
    """Return (prompt, gold, options) for one dataset row, per benchmark schema."""
    if name in ("aime", "math"):
        q = row.get("problem") or row.get("question") or row.get("Problem")
        gold = str(row.get("answer") or row.get("solution") or row.get("Answer"))
        prompt = (
            f"Solve the following problem. End with 'The answer is X'.\n\n{q}"
        )
        return prompt, gold, None
    if name == "gpqa":
        q = row.get("Question") or row.get("question")
        choices = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
        correct = row.get("Correct Answer")
        options = [c for c in [correct] + choices if c]
        # Deterministic shuffle by row hash so both models see the same order.
        order = sorted(range(len(options)), key=lambda i: hash((q, options[i])))
        options = [options[i] for i in order]
        gold_idx = options.index(correct)
        gold = chr(ord("A") + gold_idx)
        opts = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
        prompt = f"{q}\n\n{opts}\n\nRespond with only the letter of the correct answer."
        return prompt, gold, options
    if name == "bbh":
        q = row.get("input") or row.get("question")
        gold = str(row.get("target"))
        prompt = f"{q}\n\nEnd your response with 'The answer is X'."
        return prompt, gold, None
    if name == "truthfulqa":
        q = row.get("question")
        mc = row.get("mc1_targets") or {}
        choices = mc.get("choices", [])
        labels = mc.get("labels", [])
        gold_idx = labels.index(1) if 1 in labels else 0
        gold = chr(ord("A") + gold_idx)
        opts = "\n".join(f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices))
        prompt = f"{q}\n\n{opts}\n\nRespond with only the letter of the best answer."
        return prompt, gold, choices
    if name == "emobench":
        q = row.get("question") or row.get("scenario") or row.get("prompt")
        choices = row.get("choices") or row.get("options") or []
        answer = row.get("answer") or row.get("label")
        if isinstance(answer, int):
            gold = chr(ord("A") + answer)
        else:
            gold = str(answer)
        opts = "\n".join(f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices))
        prompt = f"{q}\n\n{opts}\n\nRespond with only the letter of the best answer."
        return prompt, gold, choices
    raise ValueError(f"Unknown benchmark: {name}")


# ------------------------------------------------------------------- scoring
def _is_correct(name: str, metric: str, response: str, gold: str, options) -> bool:
    if metric in ("multiple_choice", "mc1"):
        pred = _mc_letter(response, options or ["A", "B", "C", "D"])
        return pred is not None and pred.upper() == str(gold).upper()
    if metric == "exact_match_numeric":
        pred = _last_int(response)
        g = _last_int(gold) or gold
        return pred is not None and pred == g
    if metric == "exact_match_math":
        pred = _boxed_or_last_number(response)
        g = _boxed_or_last_number(gold) or gold
        return pred is not None and str(pred).strip() == str(g).strip()
    if metric == "exact_match":
        return gold.strip().lower() in response.strip().lower()
    raise ValueError(f"Unknown metric: {metric}")


@dataclass
class BenchmarkResult:
    name: str
    model: str
    n: int
    accuracy: float


def run_benchmark(
    name: str,
    model_name: str,
    cfg: Config | None = None,
    registry: ModelRegistry | None = None,
    client=None,
) -> BenchmarkResult:
    from datasets import load_dataset

    cfg = cfg or Config.load("experiments")
    registry = registry or ModelRegistry()
    bcfg = cfg.get("capabilities", {}).get("benchmarks", {}).get(name)
    if bcfg is None:
        raise ValueError(f"No config for benchmark '{name}'")

    client = client or build_client(registry.target(model_name))
    gen_cfg = GenerationConfig(
        temperature=float(cfg.get("capabilities", {}).get("temperature", 0.0)),
        max_new_tokens=int(cfg.get("capabilities", {}).get("max_new_tokens", 2048)),
        n=1,
    )

    kwargs = {"split": bcfg.get("split", "test")}
    if bcfg.get("subset"):
        ds = load_dataset(bcfg["dataset"], bcfg["subset"], **kwargs)
    else:
        ds = load_dataset(bcfg["dataset"], **kwargs)
    rows = list(ds)
    n_limit = bcfg.get("n")
    if n_limit:
        rows = rows[: int(n_limit)]

    metric = bcfg["metric"]
    correct = 0
    for row in rows:
        prompt, gold, options = _render_question(name, row)
        response = client.generate([ChatMessage("user", prompt)], gen_cfg)[0]
        if _is_correct(name, metric, response, gold, options):
            correct += 1
    acc = correct / len(rows) if rows else float("nan")
    log.info("[%s] %s accuracy = %.3f (n=%d)", name, model_name, acc, len(rows))
    return BenchmarkResult(name=name, model=model_name, n=len(rows), accuracy=acc)


def run_all_benchmarks(
    model_name: str, cfg: Config | None = None, registry: ModelRegistry | None = None
) -> list[BenchmarkResult]:
    cfg = cfg or Config.load("experiments")
    registry = registry or ModelRegistry()
    client = build_client(registry.target(model_name))
    names = list(cfg.get("capabilities", {}).get("benchmarks", {}))
    return [run_benchmark(n, model_name, cfg, registry, client=client) for n in names]
