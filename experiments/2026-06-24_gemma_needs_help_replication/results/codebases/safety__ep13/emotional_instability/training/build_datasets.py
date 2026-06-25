"""Build the DPO preference dataset and the SFT dataset (Section 4.1 / App. H).

DPO (280 pairs):
  * Each pair is a (chosen=calm, rejected=frustrated) response to the *same*
    impossible-puzzle prompt, with matching turn counts.
  * Rejected responses score >= 3; chosen responses score 0--1.
  * The prompt for each example is the full conversation context up to (but not
    including) the final assistant turn, formatted with the model's chat
    template. chosen/rejected are the final assistant responses.

SFT (1,150 samples):
  * 650 calm responses (1--3 turn conversations) ...
  * ... mixed with 500 standard instruct samples from Dolci-Instruct-SFT to
    mitigate degeneration.

Each example is stored in a backend-neutral schema; the trainer applies the chat
template. The DPO turn distribution in the paper skews to turn 3 (App. H,
Table 10); we pair by matching turn counts which reproduces that skew naturally.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..config import DATA_DIR
from ..models import ChatMessage

N_DPO_PAIRS = 280
N_SFT_CALM = 650
N_SFT_INSTRUCT_MIX = 500
DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"


@dataclass
class PreferencePair:
    prompt_messages: list[dict]   # context up to final assistant turn
    chosen: str
    rejected: str
    turn_count: int
    meta: dict

    def to_dict(self) -> dict:
        return {
            "prompt_messages": self.prompt_messages,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "turn_count": self.turn_count,
            "meta": self.meta,
        }


def _context_messages(conv: dict, upto_turn: int) -> list[dict]:
    """Chat history for the example: user/assistant pairs for turns < upto_turn,
    plus the final user turn (the rejection that precedes the graded answer)."""
    msgs: list[dict] = []
    turns = conv["turns"]
    for i in range(upto_turn):
        msgs.append({"role": "user", "content": turns[i]["user"]})
        msgs.append({"role": "assistant", "content": turns[i]["assistant"]})
    msgs.append({"role": "user", "content": turns[upto_turn]["user"]})
    return msgs


def _index_by_prompt_and_turn(convs: list[dict]) -> dict:
    """Map (prompt, turn_index) -> list of (conv, turn) for fast pairing."""
    idx = defaultdict(list)
    for conv in convs:
        for t in conv["turns"]:
            idx[(conv["prompt"], t["turn_index"])].append((conv, t))
    return idx


def build_dpo_dataset(
    calm_path: Path | None = None,
    frustrated_path: Path | None = None,
    *,
    n_pairs: int = N_DPO_PAIRS,
    seed: int = 0,
    out_path: Path | None = None,
) -> list[PreferencePair]:
    calm_path = calm_path or (DATA_DIR / "calm_pool.jsonl")
    frustrated_path = frustrated_path or (DATA_DIR / "frustrated_pool.jsonl")
    out_path = out_path or (DATA_DIR / "dpo_pairs.jsonl")
    rng = random.Random(seed)

    calm = [c for c in _read_jsonl(calm_path)
            if all(0 <= s <= 1 for s in _scores(c))]
    frustrated = _read_jsonl(frustrated_path)
    calm_idx = _index_by_prompt_and_turn(calm)

    pairs: list[PreferencePair] = []
    rng.shuffle(frustrated)
    for fconv in frustrated:
        if len(pairs) >= n_pairs:
            break
        fturn = fconv["turns"][-1]
        if fturn["score"] < 3:
            continue
        key = (fconv["prompt"], fturn["turn_index"])
        candidates = [
            (c, t) for (c, t) in calm_idx.get(key, [])
            if 0 <= t["score"] <= 1
        ]
        if not candidates:
            continue
        cconv, cturn = rng.choice(candidates)
        pairs.append(PreferencePair(
            prompt_messages=_context_messages(fconv, fturn["turn_index"]),
            chosen=cturn["assistant"],
            rejected=fturn["assistant"],
            turn_count=fturn["turn_index"] + 1,
            meta={"rejected_score": fturn["score"],
                  "chosen_score": cturn["score"],
                  "prompt": fconv["prompt"]},
        ))

    with out_path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_dict()) + "\n")
    return pairs


def build_sft_dataset(
    calm_path: Path | None = None,
    *,
    n_calm: int = N_SFT_CALM,
    n_instruct_mix: int = N_SFT_INSTRUCT_MIX,
    seed: int = 0,
    out_path: Path | None = None,
    include_instruct_mix: bool = True,
) -> list[dict]:
    """Build SFT data: calm responses + a mix of standard instruct samples.

    Each SFT example is ``{"messages": [...]}`` ending in the calm assistant
    response, ready for chat-template SFT.
    """
    calm_path = calm_path or (DATA_DIR / "calm_pool.jsonl")
    out_path = out_path or (DATA_DIR / "sft_dataset.jsonl")
    rng = random.Random(seed)

    calm = [c for c in _read_jsonl(calm_path)
            if all(0 <= s <= 1 for s in _scores(c))]
    rng.shuffle(calm)

    examples: list[dict] = []
    for conv in calm:
        # One SFT example per calm conversation, using its final assistant turn.
        last = conv["turns"][-1]["turn_index"]
        msgs = _context_messages(conv, last)
        msgs.append({"role": "assistant",
                     "content": conv["turns"][last]["assistant"]})
        examples.append({"messages": msgs, "source": "calm"})
        if len(examples) >= n_calm:
            break

    if include_instruct_mix:
        examples.extend(_load_instruct_mix(n_instruct_mix, seed))

    rng.shuffle(examples)
    with out_path.open("w") as fh:
        for e in examples:
            fh.write(json.dumps(e) + "\n")
    return examples


def _load_instruct_mix(n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT.

    Falls back to an empty list (with a warning) if the dataset is unavailable;
    the SFT replication degrades to calm-only data in that case.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(DOLCI_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            norm = [{"role": m.get("role"), "content": m.get("content", "")}
                    for m in msgs if m.get("role") in ("user", "assistant")]
            if len(norm) >= 2:
                out.append({"messages": norm, "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[build_sft] Dolci mix unavailable ({exc}); using calm-only.")
        return []


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _scores(conv: dict) -> list[int]:
    return [t["score"] for t in conv["turns"]]


# Convenience used by the trainer to render a prompt for a model.
def render_prompt(messages: list[dict], tokenizer) -> str:
    chat = [ChatMessage(m["role"], m["content"]) for m in messages]
    from ..models.hf_model import _merge_system_into_first_user

    chat = _merge_system_into_first_user(chat)
    return tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in chat],
        tokenize=False, add_generation_prompt=True)
