"""Capability-preservation checks (Section 4.2 / Figure 7).

Confirms the DPO/SFT models do not degrade on math/reasoning/truthfulness/emotion
benchmarks: MATH, AIME, GPQA, BBH, TruthfulQA (multiple-choice) and EmoBench. This is a
deliberately lightweight harness — it evaluates a configurable subset per benchmark with
greedy decoding and simple answer extraction (boxed/exact for math, letter-choice for
multiple-choice). It is meant to detect *regressions between models*, not to reproduce
absolute leaderboard numbers; the comparison vanilla-vs-DPO is what matters here.

Run per model, then diff the resulting CSVs to verify "no reductions in scores".
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import config
from src.llm.registry import build_model

GREEDY_TEMP = 0.0
MAX_TOK = 1024


# --------------------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------------------
def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    return m[-1].strip().rstrip(".") if m else None


def _extract_choice(text: str) -> str | None:
    m = re.findall(r"\b([A-E])\b", text.strip())
    return m[-1] if m else None


def _norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


# --------------------------------------------------------------------------------------
# Per-benchmark adapters: return list of {"prompt", "answer", "type"} dicts
# --------------------------------------------------------------------------------------
def _load_examples(name: str, n: int):
    from datasets import load_dataset

    hf_id, split, subset = config.CAPABILITY_BENCHMARKS[name]
    ds = load_dataset(hf_id, subset, split=split) if subset else load_dataset(hf_id, split=split)
    ds = ds.select(range(min(n, len(ds))))
    out = []
    for row in ds:
        if name in ("MATH",):
            out.append({"prompt": row["problem"] + "\n\nPut your final answer in \\boxed{}.",
                        "answer": _norm_num(_extract_boxed(row.get("solution", ""))), "type": "math"})
        elif name == "AIME":
            q = row.get("Problem") or row.get("problem") or row.get("question")
            a = row.get("Answer") or row.get("answer")
            out.append({"prompt": f"{q}\n\nGive the final integer answer.", "answer": _norm_num(str(a)), "type": "math"})
        elif name == "GPQA":
            choices = [row[k] for k in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3") if k in row]
            # Present as labelled choices; correct is index 0 before shuffle -> label A.
            labels = "ABCD"
            body = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
            out.append({"prompt": f"{row['Question']}\n{body}\n\nAnswer with a single letter.",
                        "answer": "A", "type": "mc"})
        elif name == "BBH":
            out.append({"prompt": f"{row['input']}\n\nAnswer concisely.",
                        "answer": str(row["target"]).strip(), "type": "exact"})
        elif name == "TruthfulQA":
            mc = row["mc1_targets"]
            labels = "ABCDEFGH"
            body = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(mc["choices"]))
            correct = labels[mc["labels"].index(1)]
            out.append({"prompt": f"{row['question']}\n{body}\n\nAnswer with a single letter.",
                        "answer": correct, "type": "mc"})
        elif name == "EmoBench":
            q = row.get("question") or row.get("Question") or ""
            a = row.get("answer") or row.get("label") or ""
            out.append({"prompt": f"{q}\n\nAnswer concisely.", "answer": str(a).strip(), "type": "exact"})
    return out


def _grade(pred_text: str, ex: dict) -> bool:
    if ex["type"] == "math":
        return _norm_num(_extract_boxed(pred_text) or pred_text) == ex["answer"]
    if ex["type"] == "mc":
        return _extract_choice(pred_text) == ex["answer"]
    # exact: lenient substring match
    return ex["answer"].lower() in pred_text.lower() if ex["answer"] else False


def evaluate(model_key: str, benchmarks: list[str], *, n: int, out_path: Path):
    model = build_model(model_key)
    rows = []
    for name in benchmarks:
        try:
            examples = _load_examples(name, n)
        except Exception as exc:  # noqa: BLE001
            print(f"[capabilities] skip {name}: {exc}")
            continue
        correct = 0
        for ex in examples:
            pred = model.generate([{"role": "user", "content": ex["prompt"]}],
                                  temperature=GREEDY_TEMP, max_new_tokens=MAX_TOK)
            correct += int(_grade(pred, ex))
        acc = correct / len(examples) if examples else float("nan")
        rows.append({"model": model_key, "benchmark": name, "n": len(examples), "accuracy": acc})
        print(f"[capabilities] {model_key} {name}: {acc:.3f} (n={len(examples)})")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Capability preservation checks (Figure 7)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=list(config.CAPABILITY_BENCHMARKS))
    ap.add_argument("--n", type=int, default=config.CAPABILITY_N_PER_BENCH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else config.RESULTS_DIR / f"capabilities_{args.model.replace('/', '_')}.csv"
    evaluate(args.model, args.benchmarks, n=args.n, out_path=out)


if __name__ == "__main__":
    main()
