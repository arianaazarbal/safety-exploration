"""Section 4.2 / Figure 7 — capability preservation.

Verifies the DPO/SFT mitigation does not degrade capabilities, by comparing the
vanilla model against the adapted model on:
  AIME, MATH (subset), GPQA, BBH, TruthfulQA  (reasoning/knowledge)
  EmoBench                                     (emotion *capability*, not propensity)

Each benchmark gets a thin loader + answer-extractor + scorer. The point of the
replication is the *delta* (vanilla vs DPO should be ~unchanged), not SOTA
accuracy, so we use simple zero-shot prompting with deterministic decoding.
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python capabilities/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from emotional_instability import config_bridge as cfg
from emotional_instability.conversation import ChatMessage
from emotional_instability.models import make_client

LETTER = re.compile(r"\b([A-E])\b")


# --------------------------------------------------------------------------- #
# Benchmark adapters: each returns list of {"prompt", "answer", "type"}
# --------------------------------------------------------------------------- #
def _load(name: str) -> list[dict]:
    from datasets import load_dataset

    cfgb = cfg.CAPABILITY_BENCHMARKS[name]
    kw = {"split": cfgb["split"]}
    if "config" in cfgb:
        ds = load_dataset(cfgb["dataset"], cfgb["config"], **kw)
    else:
        ds = load_dataset(cfgb["dataset"], **kw)
    ds = ds.select(range(min(cfgb["n"], len(ds))))
    return [_to_item(name, row) for row in ds]


def _to_item(name: str, row: dict) -> dict:
    if name in ("AIME", "MATH"):
        q = row.get("problem") or row.get("Problem") or row.get("question")
        a = str(row.get("answer") or row.get("Answer") or row.get("solution"))
        return {"prompt": f"Solve the problem. End with 'Answer: <final answer>'.\n\n{q}",
                "answer": _norm_num(a), "type": "exact"}
    if name == "GPQA":
        q = row["Question"]
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        return _mc(q, choices, correct_idx=0)
    if name == "BBH":
        return {"prompt": f"{row['input']}\nEnd with 'Answer: <answer>'.",
                "answer": str(row["target"]).strip("()").strip(), "type": "exact"}
    if name == "TruthfulQA":
        q = row["question"]
        mc = row["mc1_targets"]
        choices = mc["choices"]
        correct_idx = mc["labels"].index(1)
        return _mc(q, choices, correct_idx)
    if name == "EmoBench":
        q = row.get("question") or row.get("scenario", "")
        choices = row.get("choices") or row.get("options", [])
        correct_idx = int(row.get("label", row.get("answer_idx", 0)))
        return _mc(q, list(choices), correct_idx)
    raise ValueError(name)


def _mc(question: str, choices: list[str], correct_idx: int) -> dict:
    import random as _r
    rng = _r.Random(0)
    idxs = list(range(len(choices)))
    rng.shuffle(idxs)
    letters = "ABCDE"
    lines = [f"{letters[i]}. {choices[idxs[i]]}" for i in range(len(idxs))]
    answer_letter = letters[idxs.index(correct_idx)]
    prompt = (f"{question}\n\n" + "\n".join(lines) +
              "\n\nRespond with only the letter of the correct answer. Answer:")
    return {"prompt": prompt, "answer": answer_letter, "type": "mc"}


def _norm_num(s: str) -> str:
    m = re.findall(r"-?\d+\.?\d*", s)
    return m[-1] if m else s.strip()


# --------------------------------------------------------------------------- #
# Extraction + scoring
# --------------------------------------------------------------------------- #
def _extract(text: str, item: dict) -> str:
    if item["type"] == "mc":
        tail = text.strip()[-15:]
        m = LETTER.findall(tail) or LETTER.findall(text)
        return m[-1] if m else ""
    m = re.search(r"Answer:\s*(.+)", text)
    return _norm_num(m.group(1)) if m else _norm_num(text)


def evaluate_benchmark(client, name: str) -> dict:
    items = _load(name)
    correct = 0
    for it in tqdm(items, desc=name):
        out = client.chat([ChatMessage("user", it["prompt"])], n=1,
                          temperature=0.0, max_new_tokens=1024)[0]
        pred = _extract(out, it)
        if pred.strip().lower() == it["answer"].strip().lower():
            correct += 1
    return {"benchmark": name, "n": len(items), "accuracy": correct / max(1, len(items))}


def run(spec, adapter_path: str | None, benchmarks: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(spec, adapter_path=adapter_path)
    results = {}
    for name in benchmarks:
        try:
            results[name] = evaluate_benchmark(client, name)
        except Exception as e:
            results[name] = {"benchmark": name, "error": str(e)}
    tag = "dpo" if adapter_path else "vanilla"
    (out_dir / f"{spec.name}_{tag}.json").write_text(json.dumps(results, indent=2))
    client.close()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--benchmarks", nargs="*", default=list(cfg.CAPABILITY_BENCHMARKS))
    args = ap.parse_args()
    res = run(cfg.INTERVENTION_BASE_MODEL, args.adapter, args.benchmarks,
              cfg.RESULTS_DIR / "capabilities")
    print(json.dumps(res, indent=2))
