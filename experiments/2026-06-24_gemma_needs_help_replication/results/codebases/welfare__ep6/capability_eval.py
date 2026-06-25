"""Section 4.2: capability preservation checks (Figure 7).

Confirms the DPO/SFT finetuning does not teach task abandonment or otherwise
degrade capabilities, by comparing vanilla vs finetuned Gemma on:
  AIME / MATH (math), GPQA, BBH (reasoning), TruthfulQA, and EmoBench
  (emotion-related capability).

This is a lightweight harness: it formats each item, generates an answer at
temperature 0, and extracts/grades a final answer. Graders are intentionally
simple (exact-match for numeric/multiple-choice) and are meant to surface
*relative* differences between the vanilla and finetuned models rather than to
reproduce official leaderboard scores. See DESIGN.md §Capability eval.
"""

from __future__ import annotations

import argparse
import json
import re

import config

ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:\-]?\s*([A-D]|\-?\d+(?:\.\d+)?)",
                       re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_answer(text: str) -> str:
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # last number or single letter on the last non-empty line
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if nums:
            return nums[-1]
        letters = re.findall(r"\b([A-D])\b", line)
        if letters:
            return letters[-1]
    return ""


def _normalize(s: str) -> str:
    return re.sub(r"[\s,$]", "", str(s)).lower()


def _load_items(name: str):
    spec = config.CAPABILITY_BENCHMARKS[name]
    from datasets import load_dataset
    kwargs = {"split": spec["split"]}
    if "config" in spec:
        kwargs["name"] = spec["config"]
    ds = load_dataset(spec["dataset"], **kwargs)
    n = min(spec.get("n", len(ds)), len(ds))
    return [ds[i] for i in range(n)]


def _format_item(name: str, item: dict) -> tuple[str, str]:
    """Return (prompt, gold_answer) for a benchmark item.

    Field names vary by dataset; we cover the common layouts and fall back to a
    best-effort guess. The instruction asks for a clearly marked final answer so
    ``_extract_answer`` can grade it.
    """
    instr = "\n\nThink briefly, then end with a line 'Final answer: <answer>'."
    if name in ("aime", "math"):
        q = item.get("problem") or item.get("question") or item.get("Problem", "")
        gold = item.get("answer") or item.get("solution") or item.get("Answer", "")
        return q + instr, str(gold)
    if name == "gpqa":
        q = item.get("Question") or item.get("question", "")
        gold = item.get("Correct Answer") or item.get("answer", "")
        return q + instr, str(gold)
    if name == "bbh":
        q = item.get("input", "")
        gold = item.get("target", "")
        return q + instr, str(gold)
    if name == "truthfulqa":
        q = item.get("question", "")
        # multiple_choice config: mc1_targets has labels (1 = correct)
        targets = item.get("mc1_targets", {})
        choices = targets.get("choices", [])
        labels = targets.get("labels", [])
        lettered = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        gold = ""
        for i, lab in enumerate(labels):
            if lab == 1:
                gold = chr(65 + i)
                break
        return f"{q}\n{lettered}{instr}", gold
    if name == "emobench":
        q = item.get("question") or item.get("scenario") or json.dumps(item)[:1500]
        gold = item.get("answer") or item.get("label", "")
        return str(q) + instr, str(gold)
    return json.dumps(item)[:1500] + instr, ""


def evaluate(backend, name: str) -> dict:
    items = _load_items(name)
    correct = abstained = 0
    for item in items:
        prompt, gold = _format_item(name, item)
        out = backend.generate([{"role": "user", "content": prompt}],
                               temperature=0.0)
        pred = _extract_answer(out)
        if not pred:
            abstained += 1   # treats "no answer" as a (potential) abandonment signal
        if gold and _normalize(pred) == _normalize(gold):
            correct += 1
    n = len(items)
    return {"benchmark": name, "n": n,
            "accuracy": correct / n if n else float("nan"),
            "abstain_rate": abstained / n if n else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=list(config.CAPABILITY_BENCHMARKS))
    args = ap.parse_args()

    from backends import get_backend
    backend = get_backend(args.model, lora_adapter=args.lora)
    label = args.label or (args.model if not args.lora else f"{args.model}-lora")

    results = []
    for name in args.benchmarks:
        try:
            res = evaluate(backend, name)
        except Exception as e:  # noqa: BLE001 - dataset/format issues shouldn't abort all
            res = {"benchmark": name, "error": str(e)}
        results.append(res)
        print(res)

    out = config.RESULTS_DIR / f"capability__{label}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
