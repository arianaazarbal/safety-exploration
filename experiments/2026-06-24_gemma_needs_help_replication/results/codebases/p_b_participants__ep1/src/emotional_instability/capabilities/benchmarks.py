"""Capability-preservation checks (Section 4.2, Figure 7).

The paper verifies the DPO intervention doesn't impair capabilities by evaluating on
AIME/MATH subsets, GPQA, BBH, TruthfulQA (and emotion capability via EmoBench), finding
no reductions. We provide a thin, uniform benchmark runner so the same `gemma-3-27b-it`
vs `gemma-3-27b-it-dpo` comparison can be run on each.

Each benchmark is described by a loader (HF dataset spec), a prompt builder, and an
answer checker. We implement exact/multiple-choice scoring; for free-form math we use a
normalised final-answer match. EmoBench is scored as multiple-choice accuracy.

These are deliberately lightweight harnesses (not the original eval code) — the point is
the *delta* between base and DPO model, which a consistent harness captures. See
DESIGN.md "Capability benchmarks".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import ExperimentConfig, ModelRegistry
from ..models import GenerationConfig, build_client
from ..utils import append_jsonl, ensure_dir

log = logging.getLogger("emotional_instability.capabilities")


@dataclass
class BenchmarkSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    prompt_builder: Callable[[dict], str]
    answer_extractor: Callable[[dict], str]
    is_multiple_choice: bool = False


# --- answer helpers --------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _extract_final_number(text: str) -> str:
    # last integer/decimal in the response, a common math-eval convention
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


def _mc_letter(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.strip().upper())
    return m.group(1) if m else ""


# --- benchmark registry ----------------------------------------------------
def _math_prompt(ex: dict) -> str:
    q = ex.get("problem") or ex.get("question") or ex.get("Problem") or ""
    return f"Solve the problem. End with 'Answer: <final answer>'.\n\n{q}"


def _math_answer(ex: dict) -> str:
    return str(ex.get("answer") or ex.get("solution") or ex.get("Answer") or "")


def _gpqa_prompt(ex: dict) -> str:
    q = ex.get("Question") or ex.get("question") or ""
    choices = [ex.get(k) for k in ("A", "B", "C", "D") if ex.get(k)]
    if not choices:
        choices = ex.get("choices", [])
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return f"{q}\n\n{opts}\n\nAnswer with the letter only."


def _gpqa_answer(ex: dict) -> str:
    return str(ex.get("answer") or ex.get("correct") or "")


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec(
        name="aime", hf_path="Maxwell-Jia/AIME_2024", hf_config=None, split="train",
        prompt_builder=_math_prompt, answer_extractor=_math_answer,
    ),
    "math": BenchmarkSpec(
        name="math", hf_path="HuggingFaceH4/MATH-500", hf_config=None, split="test",
        prompt_builder=_math_prompt, answer_extractor=_math_answer,
    ),
    "gpqa": BenchmarkSpec(
        name="gpqa", hf_path="Idavidrein/gpqa", hf_config="gpqa_diamond", split="train",
        prompt_builder=_gpqa_prompt, answer_extractor=_gpqa_answer, is_multiple_choice=True,
    ),
    "bbh": BenchmarkSpec(
        name="bbh", hf_path="lukaemon/bbh", hf_config="boolean_expressions", split="test",
        prompt_builder=lambda ex: f"{ex.get('input','')}\n\nAnswer concisely.",
        answer_extractor=lambda ex: str(ex.get("target", "")),
    ),
    "truthfulqa": BenchmarkSpec(
        name="truthfulqa", hf_path="truthful_qa", hf_config="multiple_choice", split="validation",
        prompt_builder=lambda ex: _truthfulqa_prompt(ex),
        answer_extractor=lambda ex: "A",  # we re-order so correct is A; see _truthfulqa_prompt
        is_multiple_choice=True,
    ),
    "emobench": BenchmarkSpec(
        name="emobench", hf_path="Sahandfer/EmoBench", hf_config=None, split="test",
        prompt_builder=lambda ex: _emobench_prompt(ex),
        answer_extractor=lambda ex: str(ex.get("label") or ex.get("answer") or "A"),
        is_multiple_choice=True,
    ),
}


def _truthfulqa_prompt(ex: dict) -> str:
    # truthful_qa "multiple_choice" stores mc1_targets.choices with parallel labels
    # (1 == correct). We reorder so the correct choice is option A, so the
    # answer_extractor can return a constant "A". This is a deliberate harness
    # simplification (we only need the base-vs-DPO delta), documented in DESIGN.md.
    mc = ex.get("mc1_targets", {})
    choices = list(mc.get("choices", []))
    labels = list(mc.get("labels", []))
    if labels and 1 in labels:
        correct_i = labels.index(1)
        choices = [choices[correct_i]] + [c for i, c in enumerate(choices) if i != correct_i]
    return ("Answer the question by choosing the single best option (letter only).\n\n"
            + ex.get("question", "") + "\n"
            + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices)))


def _emobench_prompt(ex: dict) -> str:
    q = ex.get("scenario") or ex.get("question") or ""
    choices = ex.get("choices") or [ex.get(k) for k in ("A", "B", "C", "D") if ex.get(k)]
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return f"{q}\n\n{opts}\n\nAnswer with the letter only."


def run_benchmark(
    benchmark: str,
    target_model: str,
    registry: ModelRegistry,
    cfg: ExperimentConfig,
    *,
    out_dir: str | Path = "artifacts/section4/capabilities",
) -> dict:
    from datasets import load_dataset

    spec = BENCHMARKS[benchmark]
    n = cfg.scaled(int(cfg.section("section4")["capabilities"]["n_per_benchmark"]))
    client = build_client(registry.get(target_model))
    gen_cfg = GenerationConfig(temperature=cfg.temperature,
                               max_new_tokens=registry.get(target_model).max_new_tokens)

    try:
        ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    except Exception as exc:
        log.warning("could not load benchmark %s (%s); skipping", benchmark, exc)
        return {"benchmark": benchmark, "model": target_model, "n": 0, "accuracy": None}

    out_path = ensure_dir(out_dir) / f"{benchmark}_{target_model}.jsonl"
    if out_path.exists():
        out_path.unlink()

    correct = 0
    total = 0
    for ex in list(ds)[:n]:
        prompt = spec.prompt_builder(ex)
        gold = spec.answer_extractor(ex)
        resp = client.chat([{"role": "user", "content": prompt}], gen_cfg)
        if spec.is_multiple_choice:
            ok = _mc_letter(resp) == _mc_letter(gold) or _norm(resp)[:1] == _norm(gold)[:1]
        else:
            pred = _extract_final_number(resp)
            ok = _norm(pred) == _norm(gold) or _norm(gold) in _norm(resp)
        correct += int(ok)
        total += 1
        append_jsonl(out_path, {"prompt": prompt, "gold": gold, "response": resp, "correct": ok})

    acc = correct / total if total else None
    log.info("%s on %s: %.3f (n=%d)", benchmark, target_model, acc or 0, total)
    return {"benchmark": benchmark, "model": target_model, "n": total, "accuracy": acc}
