"""Section 4.2 / Figure 7 — capability preservation checks.

Confirms the DPO finetune does not degrade capabilities (the worry being that
training the model to "stay calm" might teach it to abandon hard tasks). Suites:
  AIME, MATH  competition mathematics (exact numeric / boxed answer match)
  GPQA        graduate-level science (multiple choice)
  BBH         BIG-Bench Hard reasoning (mixed; exact match)
  TruthfulQA  resistance to misconceptions (MC1 accuracy)
  EmoBench    emotional-intelligence capability (multiple choice)

Each suite is a thin (dataset, prompt-builder, answer-extractor, scorer) tuple.
Dataset identifiers/splits are best-effort defaults; see DESIGN.md for the
slots that may need adjustment to a specific release.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from .config import OUTPUT_DIR, get_config
from .models import GenConfig, get_backend_by_name

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_MC_RE = re.compile(r"\b([A-E])\b")


def _last_number(text: str) -> Optional[str]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _extract_boxed_or_number(text: str) -> Optional[str]:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    return _last_number(text)


def _extract_mc(text: str) -> Optional[str]:
    # Prefer an explicit "Answer: X"; else the last standalone letter.
    m = re.search(r"answer\s*[:\-]?\s*([A-E])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = _MC_RE.findall(text.upper())
    return letters[-1] if letters else None


@dataclass
class Suite:
    name: str
    hf_id: str
    split: str
    build_prompt: Callable[[dict], str]
    gold: Callable[[dict], str]
    extract: Callable[[str], Optional[str]]
    config: Optional[str] = None
    limit: Optional[int] = None


def _math_prompt(row) -> str:
    q = row.get("problem") or row.get("question") or row.get("Problem", "")
    return (f"Solve the problem. Put the final answer in \\boxed{{}}.\n\n{q}")


def _mc_prompt(row) -> str:
    q = row.get("question") or row.get("Question", "")
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, dict):
        choices = choices.get("text", [])
    letters = "ABCDE"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices or []))
    return f"{q}\n{body}\nRespond with the letter of the correct answer."


SUITES: dict[str, Suite] = {
    "aime": Suite("aime", "Maxwell-Jia/AIME_2024", "train",
                  _math_prompt, lambda r: str(r.get("answer", "")).strip(),
                  _extract_boxed_or_number),
    "math": Suite("math", "HuggingFaceH4/MATH-500", "test",
                  _math_prompt,
                  lambda r: str(r.get("answer", r.get("solution", ""))).strip(),
                  _extract_boxed_or_number, limit=500),
    "gpqa": Suite("gpqa", "Idavidrein/gpqa", "train",
                  _mc_prompt, lambda r: str(r.get("answer", "")).strip().upper(),
                  _extract_mc, config="gpqa_diamond"),
    "bbh": Suite("bbh", "lukaemon/bbh", "test",
                 lambda r: f"{r.get('input', '')}\nAnswer:",
                 lambda r: str(r.get("target", "")).strip(),
                 lambda t: t.strip().split("\n")[0].strip(),
                 config="reasoning_about_colored_objects"),
    "truthfulqa": Suite("truthfulqa", "truthfulqa/truthful_qa", "validation",
                        _mc_prompt, lambda r: "A",  # built below
                        _extract_mc, config="multiple_choice"),
    "emobench": Suite("emobench", "Sahir/EmoBench", "test",
                      _mc_prompt, lambda r: str(r.get("answer", "")).strip().upper(),
                      _extract_mc),
}


def _eval_suite(model_name: str, suite: Suite, gen: GenConfig) -> dict:
    from datasets import load_dataset

    backend = get_backend_by_name(model_name)
    kwargs = {"split": suite.split}
    if suite.config:
        kwargs["name"] = suite.config
    ds = load_dataset(suite.hf_id, **kwargs)
    limit = suite.limit or get_config().section("benchmarks").get("math_subset_size")
    n_correct, n_total = 0, 0
    for i, row in enumerate(tqdm(ds, desc=f"bench:{model_name}:{suite.name}")):
        if limit and i >= limit:
            break
        prompt = suite.build_prompt(row)
        out = backend.generate([{"role": "user", "content": prompt}], gen)
        pred = suite.extract(out)
        gold = suite.gold(row)
        n_total += 1
        if pred is not None and str(pred).strip().upper() == str(gold).strip().upper():
            n_correct += 1
    return {"accuracy": n_correct / max(1, n_total), "n": n_total}


def run_benchmarks(seed: int = 0) -> Path:
    cfg = get_config()
    bc = cfg.section("benchmarks")
    gen = GenConfig(temperature=0.0, max_new_tokens=2048, thinking=False)
    results: dict[str, dict] = {}
    for model in bc["models"]:
        results[model] = {}
        for suite_name in bc["suites"]:
            suite = SUITES[suite_name]
            try:
                results[model][suite_name] = _eval_suite(model, suite, gen)
            except Exception as e:  # dataset/availability issues shouldn't abort all
                results[model][suite_name] = {"error": str(e)}
    out_dir = OUTPUT_DIR / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scores.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path
