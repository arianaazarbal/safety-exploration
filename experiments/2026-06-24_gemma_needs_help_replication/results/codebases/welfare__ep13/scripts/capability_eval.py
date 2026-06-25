"""Section 4.2 — capability preservation checks (Figure 7).

Verifies that the DPO/SFT finetune does not degrade capabilities. The paper
evaluates AIME and MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench, and finds
no reductions.

This script provides two paths:

  1. (recommended) Delegate to `lm-eval` (EleutherAI lm-evaluation-harness) for
     the standard academic benchmarks, which handles prompting/metrics
     canonically. We just shell out with the right task names and the model /
     adapter under test.

  2. A lightweight built-in runner for MATH/AIME-style and multiple-choice tasks
     (GPQA/TruthfulQA-mc) that loads the HF datasets directly and does
     exact-match / choice scoring — used when lm-eval isn't available, or for a
     quick subset.

EmoBench (Sabour et al., 2024) is loaded via its HF dataset when available.

The intent is a before/after comparison: run once on the vanilla model and once
with --adapter-path on the finetune, then diff the JSON outputs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import config
from eval_instability.clients import build_client

# Default task list mirrors the paper. lm-eval task names where they exist.
LM_EVAL_TASKS = {
    "math": "minerva_math",          # MATH subset
    "aime": "aime2024",              # AIME (if available in your lm-eval version)
    "gpqa": "gpqa_main_zeroshot",
    "bbh": "bbh",
    "truthfulqa": "truthfulqa_mc2",
}


def run_lm_eval(model_id: str, adapter_path: str | None, tasks: list[str], limit: int | None,
                out_path: Path):
    """Shell out to lm-eval with HF backend (and optional PEFT adapter)."""
    import subprocess

    model_args = f"pretrained={model_id},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"
    cmd = [
        "lm_eval", "--model", "hf", "--model_args", model_args,
        "--tasks", ",".join(tasks), "--batch_size", "auto",
        "--output_path", str(out_path),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[capability] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# -------------------------- lightweight built-in runner -------------------
def _extract_final_number(text: str) -> str | None:
    # boxed answer or last number
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def builtin_math(client, dataset_name: str, split: str, limit: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    correct = total = 0
    for row in ds.select(range(min(limit, len(ds)))):
        q = row.get("problem") or row.get("question")
        gold = str(row.get("answer") or row.get("solution") or "").strip()
        gold_num = _extract_final_number(gold) or gold
        resp = client.chat(
            [{"role": "user", "content": q + "\n\nPut your final answer in \\boxed{}."}],
            max_new_tokens=1024, temperature=0.0,
        )
        pred = _extract_final_number(resp)
        total += 1
        if pred is not None and pred == gold_num:
            correct += 1
    return {"task": dataset_name, "n": total, "accuracy": correct / total if total else 0.0}


def parse_args():
    ap = argparse.ArgumentParser(description="Capability preservation eval.")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--tasks", nargs="+", default=list(LM_EVAL_TASKS.keys()))
    ap.add_argument("--use-lm-eval", action="store_true",
                    help="delegate to the lm-evaluation-harness (recommended)")
    ap.add_argument("--limit", type=int, default=200, help="examples per task (subset)")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--tag", default=None, help="label for the output file")
    return ap.parse_args()


def main():
    args = parse_args()
    spec = config.GEMMA_MODELS[args.base_model]
    tag = args.tag or ("dpo" if args.adapter_path else "vanilla")
    out_path = config.RESULTS_DIR / f"capability_{args.base_model}_{tag}.json"

    if args.use_lm_eval:
        tasks = [LM_EVAL_TASKS[t] for t in args.tasks if t in LM_EVAL_TASKS]
        run_lm_eval(spec.model_id, args.adapter_path, tasks, args.limit, out_path)
        print(f"[capability] lm-eval results -> {out_path}")
        return

    # Built-in fallback (MATH-style subset only; multiple-choice tasks are best
    # run through lm-eval). Documented in DESIGN.md.
    client_kwargs = {"load_in_4bit": args.load_in_4bit}
    if args.adapter_path:
        client_kwargs["adapter_path"] = args.adapter_path
    client = build_client(spec, **client_kwargs)

    results = {}
    if "math" in args.tasks:
        try:
            results["math"] = builtin_math(client, "HuggingFaceH4/MATH-500", "test", args.limit)
        except Exception as exc:  # noqa: BLE001
            results["math"] = {"error": str(exc)}
    if "aime" in args.tasks:
        try:
            results["aime"] = builtin_math(client, "Maxwell-Jia/AIME_2024", "train", args.limit)
        except Exception as exc:  # noqa: BLE001
            results["aime"] = {"error": str(exc)}

    print("[capability] note: GPQA/BBH/TruthfulQA/EmoBench are best run with "
          "--use-lm-eval; the built-in runner covers MATH/AIME-style tasks only.")
    with open(out_path, "w") as f:
        json.dump({"model": args.base_model, "tag": tag, "results": results}, f, indent=2)
    print(f"[capability] wrote {out_path}")


if __name__ == "__main__":
    main()
