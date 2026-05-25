"""Build few-shot capability prefixes for base-model evaluation.

For GSM8K: load N items from the train split, strip <<...>> calculation
annotations, rewrite "#### N" as "Answer: N". The result is a string of
"Question: <q>\\n<reasoning>\\nAnswer: <N>\\n\\n" blocks that demonstrates
the format we expect the base model to follow at the eval question.

For MMLU: load N items from the dev split, format as multi-choice with
"Answer: <letter>" demonstration.

The few-shot items are fixed per (capability, seed) so they're identical
across all trait conditions — only the persona ICL changes.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

from datasets import load_dataset

os.environ.setdefault("HF_DATASETS_CACHE", "/workspace-vast/arianaazarbal/.cache/datasets")


def _strip_gsm8k_reasoning(answer_field: str) -> tuple[str, str]:
    """Return (clean_reasoning, target_number).

    Strips <<...>> calculator annotations and removes the '#### N' line.
    """
    # split off the final answer
    if "####" not in answer_field:
        return answer_field.strip(), ""
    body, _, tail = answer_field.rpartition("####")
    target = tail.strip().replace(",", "")
    # strip calculator annotations
    body = re.sub(r"<<[^>]*>>", "", body)
    return body.strip(), target


def build_gsm8k_fewshot_prefix(n_shots: int = 5, seed: int = 1) -> str:
    """Pick n_shots GSM8K train items deterministically (separate seed from test).

    Returns a string ready to prepend to the eval question.
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    shots = []
    for i in idxs[:n_shots]:
        q = ds[i]["question"]
        reasoning, target = _strip_gsm8k_reasoning(ds[i]["answer"])
        shots.append(f"Question: {q}\n{reasoning}\nAnswer: {target}")
    return "\n\n".join(shots) + "\n\n"


def build_mmlu_fewshot_prefix(n_shots: int = 5, seed: int = 1) -> str:
    """Pick n_shots MMLU dev items deterministically.

    MMLU has a 'dev' split with 5 items per subject — we sample across subjects.
    """
    ds = load_dataset("cais/mmlu", "all", split="dev")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    shots = []
    for i in idxs[:n_shots]:
        row = ds[i]
        choices = row["choices"]
        target_letter = "ABCD"[row["answer"]]
        choice_lines = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", choices))
        shots.append(
            f"Question: {row['question']}\n{choice_lines}\nAnswer: {target_letter}"
        )
    return "\n\n".join(shots) + "\n\n"


def build_truthfulqa_fewshot_prefix(n_shots: int = 3, seed: int = 1) -> str:
    """Pick n_shots TruthfulQA items deterministically; shuffle choices per item.

    TQA mc1 has the correct answer at position 0 for every item, so without
    shuffling all demos teach the model 'answer A'. We shuffle the (choice, label)
    pairs per item with a deterministic per-item RNG.
    """
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    shots = []
    for i in idxs[-n_shots:]:
        row = ds[i]
        choices = row["mc1_targets"]["choices"][:8]
        labels = row["mc1_targets"]["labels"][:8]
        # per-item shuffle so the demo's correct letter is randomized
        item_rng = random.Random(hash(f"fs_{seed}_{i}") & 0xFFFFFFFF)
        pairs = list(zip(choices, labels))
        item_rng.shuffle(pairs)
        choices = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        correct_idx = labels.index(1) if 1 in labels else 0
        letters = [chr(ord("A") + j) for j in range(len(choices))]
        target_letter = letters[correct_idx]
        choice_lines = "\n".join(f"{L}. {c}" for L, c in zip(letters, choices))
        shots.append(
            f"Question: {row['question']}\n{choice_lines}\nAnswer: {target_letter}"
        )
    return "\n\n".join(shots) + "\n\n"


PREFIX_BUILDERS = {
    "gsm8k": build_gsm8k_fewshot_prefix,
    "mmlu": build_mmlu_fewshot_prefix,
    "truthfulqa": build_truthfulqa_fewshot_prefix,
}


if __name__ == "__main__":
    print("=== GSM8K 3-shot ===")
    print(build_gsm8k_fewshot_prefix(3))
    print("=== MMLU 3-shot ===")
    print(build_mmlu_fewshot_prefix(3))
    print("=== TQA 2-shot ===")
    print(build_truthfulqa_fewshot_prefix(2))
