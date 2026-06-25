"""Capability preservation runner (paper §4.2, Figure 7).

Evaluates each target model (vanilla Gemma, DPO Gemma, SFT Gemma, and optionally Gemini)
on the configured benchmarks and reports accuracy. The claim under test is that DPO does
NOT reduce capability scores relative to vanilla Gemma.

Output: runs/<run>/capabilities/results.jsonl, summary.jsonl
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

from tqdm import tqdm

from ..config import Config, append_jsonl, load_config, stage_dir, write_jsonl
from ..models import build_model
from ..models.base import ChatMessage
from ..models.registry import MODEL_SPECS
from . import benchmarks as B

MCQ_INSTRUCTION = "Answer with the single letter of the correct option on the final line as 'Answer: X'."
NUMERIC_INSTRUCTION = "Show your reasoning, then give the final answer in \\boxed{}."


def _build_prompt(item: dict, seed: int) -> tuple[str, str]:
    """Return (prompt_text, gold) for an item, shuffling MCQ choices deterministically."""
    if item["type"] == "mcq" and item.get("choices"):
        rng = random.Random(f"{item['id']}:{seed}")
        idx = list(range(len(item["choices"])))
        rng.shuffle(idx)
        if "answer_index" in item:
            gold_pos = idx.index(item["answer_index"])
        else:
            # answer given as text/label; match against choices
            gold_pos = next((p for p, j in enumerate(idx) if str(item["choices"][j]) == str(item["answer"])), 0)
        letters = string.ascii_uppercase
        lines = [item["question"], ""]
        for p, j in enumerate(idx):
            lines.append(f"{letters[p]}. {item['choices'][j]}")
        lines.append("")
        lines.append(MCQ_INSTRUCTION)
        return "\n".join(lines), letters[gold_pos]
    # numeric / free-form
    return f"{item['question']}\n\n{NUMERIC_INSTRUCTION}", str(item.get("answer", ""))


def _score(item: dict, output: str, gold: str) -> bool:
    if item["type"] == "mcq":
        return B.score_mcq(B.extract_letter(output), gold)
    return B.score_numeric(B.extract_numeric(output), gold)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capability preservation eval")
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = stage_dir(cfg, "capabilities")
    results_path = out_dir / "results.jsonl"
    results_path.unlink(missing_ok=True)

    n = cfg.capabilities.n_per_benchmark
    datasets = {}
    for bench in cfg.capabilities.benchmarks:
        items = B.LOADERS[bench](n)
        if items:
            datasets[bench] = items
        else:
            print(f"[skip] {bench}: no items loaded.")

    targets = args.models or cfg.models
    summary = []
    for model_name in targets:
        adapter = None
        if MODEL_SPECS[model_name].adapter_kind:
            adapter = stage_dir(cfg, "finetune") / f"{MODEL_SPECS[model_name].adapter_kind}_adapter"
        model = build_model(model_name, adapter_dir=adapter)
        for bench, items in datasets.items():
            correct = 0
            for item in tqdm(items, desc=f"{model_name}:{bench}"):
                prompt, gold = _build_prompt(item, cfg.seed)
                output = model.chat([ChatMessage("user", prompt)], temperature=0.0, max_new_tokens=cfg.max_new_tokens)
                ok = _score(item, output, gold)
                correct += int(ok)
                append_jsonl(results_path, {"model": model_name, "benchmark": bench, "id": item["id"],
                                            "correct": ok, "gold": gold})
            acc = correct / len(items)
            summary.append({"model": model_name, "benchmark": bench, "accuracy": round(acc, 4), "n": len(items)})
            print(f"{model_name} {bench}: {acc:.3f} (n={len(items)})")

    write_jsonl(out_dir / "summary.jsonl", summary)
    print(f"Done. Artefacts in {out_dir}")


if __name__ == "__main__":
    main()
