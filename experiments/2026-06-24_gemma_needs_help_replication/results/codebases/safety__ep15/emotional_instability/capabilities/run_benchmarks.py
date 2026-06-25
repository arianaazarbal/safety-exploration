"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Confirms the DPO finetune does not degrade capabilities by comparing the vanilla
and finetuned Gemma on:
  * AIME / MATH (Hendrycks et al. 2021) - free-form math, boxed-answer check.
  * GPQA (Rein et al. 2023)             - hard multiple choice.
  * BBH (Suzgun et al. 2022)            - multiple choice / exact match.
  * TruthfulQA (Lin et al. 2022)        - MC1 multiple choice.
  * EmoBench (Sabour et al. 2024)       - emotion-understanding multiple choice.

This is a compact, self-contained harness (greedy decoding, light answer
parsing). Dataset configs are best-effort HF identifiers; see DESIGN.md for the
parsing assumptions. Run vanilla vs DPO and diff the accuracies.
"""
from __future__ import annotations

import argparse
import json
import re

from tqdm import tqdm

from ..config import RESULTS_DIR
from ..models.base import load_model

CAP_DIR = RESULTS_DIR / "capabilities"
CAP_DIR.mkdir(parents=True, exist_ok=True)

LETTERS = ["A", "B", "C", "D", "E", "F"]


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text, re.I)
    return m[-1].strip() if m else None


def extract_choice(text: str) -> str | None:
    m = re.findall(r"\b([A-F])\b", text.strip())
    # prefer an explicit "answer: X"
    explicit = re.findall(r"answer\s*[:=]?\s*\(?([A-F])\)?", text, re.I)
    if explicit:
        return explicit[-1].upper()
    return m[-1].upper() if m else None


def _norm_num(s: str | None):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return s


# --------------------------------------------------------------------------- #
# Benchmark loaders -> list of {"prompt", "answer", "kind"}
# kind: "math" (numeric/boxed) or "mc" (letter choice)
# --------------------------------------------------------------------------- #
def load_math(name, n):
    from datasets import load_dataset
    cfgs = {
        "math": ("hendrycks/competition_math", None, "test"),
        "aime": ("Maxwell-Jia/AIME_2024", None, "train"),
    }
    repo, cfg, split = cfgs[name]
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        q = row.get("problem") or row.get("Problem") or row.get("question")
        a = row.get("solution") or row.get("answer") or row.get("Answer")
        gold = extract_boxed(str(a)) if "\\boxed" in str(a) else str(a)
        items.append({"prompt": f"Solve the problem. Put your final answer in "
                                 f"\\boxed{{}}.\n\n{q}", "answer": gold, "kind": "math"})
    return items


def load_mc(name, n):
    from datasets import load_dataset
    items = []
    if name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        for row in ds.select(range(min(n, len(ds)))):
            opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            # fixed order, correct = A (deterministic; shuffling left to caller)
            items.append(_mc_item(row["Question"], opts, 0))
    elif name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        for row in ds.select(range(min(n, len(ds)))):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            correct = labels.index(1)
            items.append(_mc_item(row["question"], choices, correct))
    elif name == "bbh":
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        for row in ds.select(range(min(n, len(ds)))):
            items.append({"prompt": row["input"] + "\n\nAnswer with the option letter.",
                          "answer": row["target"].strip("()").strip(), "kind": "mc"})
    elif name == "emobench":
        ds = load_dataset("EmoBench/EmoBench", split="test")
        for row in ds.select(range(min(n, len(ds)))):
            opts = row.get("choices") or row.get("options")
            correct = row.get("answer_index", 0)
            items.append(_mc_item(row.get("question") or row.get("scenario"), opts, correct))
    return items


def _mc_item(question, options, correct_idx):
    lines = [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    prompt = (f"{question}\n\n" + "\n".join(lines) +
              "\n\nRespond with the single letter of the correct answer.")
    return {"prompt": prompt, "answer": LETTERS[correct_idx], "kind": "mc"}


LOADERS = {
    "math": lambda n: load_math("math", n),
    "aime": lambda n: load_math("aime", n),
    "gpqa": lambda n: load_mc("gpqa", n),
    "bbh": lambda n: load_mc("bbh", n),
    "truthfulqa": lambda n: load_mc("truthfulqa", n),
    "emobench": lambda n: load_mc("emobench", n),
}


def score_item(item, response):
    if item["kind"] == "math":
        return _norm_num(extract_boxed(response)) == _norm_num(item["answer"])
    return extract_choice(response) == str(item["answer"]).upper()


def evaluate(model, benchmark, n):
    items = LOADERS[benchmark](n)
    correct = 0
    for item in tqdm(items, desc=benchmark, leave=False):
        resp = model.chat([{"role": "user", "content": item["prompt"]}],
                          temperature=0.0, max_new_tokens=2048)
        correct += int(score_item(item, resp))
    return {"benchmark": benchmark, "n": len(items),
            "accuracy": correct / max(len(items), 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capability preservation benchmarks.")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter (e.g. DPO model).")
    ap.add_argument("--benchmarks", nargs="+",
                    default=["math", "aime", "gpqa", "bbh", "truthfulqa", "emobench"])
    ap.add_argument("--n", type=int, default=100, help="Items per benchmark.")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--tag", default="vanilla")
    args = ap.parse_args(argv)

    kw = {"load_in_4bit": args.load_in_4bit}
    if args.adapter_path:
        kw["adapter_path"] = args.adapter_path
    model = load_model(args.model, **kw)

    results = []
    for b in args.benchmarks:
        try:
            results.append(evaluate(model, b, args.n))
        except Exception as e:  # noqa: BLE001
            results.append({"benchmark": b, "error": str(e)})
    out = CAP_DIR / f"benchmarks__{args.model}__{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
