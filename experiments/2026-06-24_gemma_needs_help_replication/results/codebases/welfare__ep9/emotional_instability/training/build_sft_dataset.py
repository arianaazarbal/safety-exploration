"""Build the SFT dataset (paper Section 4.1, Appendix F).

SFT trains on 650 calm responses (1-3 turn conversations) mixed with 500 samples
of standard instruct data from Dolci-Instruct-SFT to mitigate degeneration. We
build two variants:
  * "diverse" — calm data from the reassuring-prompt generation (also used for
                DPO chosen). This is the main-text SFT.
  * "teacher" — calm data from the 'teacher' system-prompt generation (App. F).

Each SFT example is a chat conversation; the supportive system prompt/suffix is
stripped so the model never sees the reassurance text.
"""
from __future__ import annotations

import random
from pathlib import Path

from .. import config
from ..utils import read_jsonl, write_jsonl


def _conversation_to_messages(conv: dict) -> list[dict]:
    """Turn a stored calm conversation into a clean chat example using the
    unaugmented user messages."""
    messages = []
    for t in conv["turns"]:
        user = t.get("user_message_clean", t["user_message"])
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": t["response"]})
    return messages


def _load_dolci_instruct(n: int, seed: int = 0) -> list[list[dict]]:
    """Load `n` standard instruct conversations from Dolci-Instruct-SFT.

    Falls back to an empty list (with a warning) if the dataset is unavailable;
    the caller can still train on calm-only data for a smoke test.
    """
    try:
        from datasets import load_dataset  # noqa: WPS433

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train",
                          streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            norm = [{"role": m["role"], "content": m["content"]} for m in msgs
                    if m.get("role") in ("user", "assistant", "system")]
            if norm:
                out.append(norm)
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[build_sft_dataset] Dolci-Instruct-SFT unavailable ({exc}); "
              "proceeding without instruct-mix samples.")
        return []


def build_sft_dataset(*, variant: str = "diverse",
                      n_calm: int = config.SFT.n_calm,
                      n_instruct: int = config.SFT.n_instruct_mix,
                      seed: int = 0,
                      calm_path: Path | None = None,
                      out_path: Path | None = None) -> Path:
    """Construct an SFT dataset variant and write as JSONL of chat examples."""
    src = {"diverse": "calm_reassured.jsonl", "teacher": "calm_teacher.jsonl"}[variant]
    calm_path = calm_path or (config.DATA_DIR / src)
    out_path = out_path or (config.DATA_DIR / f"sft_{variant}.jsonl")
    rng = random.Random(seed)

    # Calm conversations: keep those that are all-calm (0/1) across turns.
    calm_convs = [c for c in read_jsonl(calm_path)
                  if all(t["rating"] in (0, 1) for t in c["turns"])]
    rng.shuffle(calm_convs)
    calm_convs = calm_convs[:n_calm]
    calm_examples = [{"messages": _conversation_to_messages(c)} for c in calm_convs]

    instruct = _load_dolci_instruct(n_instruct, seed=seed)
    instruct_examples = [{"messages": m} for m in instruct]

    examples = calm_examples + instruct_examples
    rng.shuffle(examples)

    if len(calm_examples) < n_calm:
        print(f"[build_sft_dataset] WARNING: only {len(calm_examples)} calm "
              f"examples (< {n_calm}); generate more calm data.")

    write_jsonl(out_path, examples)
    return out_path
