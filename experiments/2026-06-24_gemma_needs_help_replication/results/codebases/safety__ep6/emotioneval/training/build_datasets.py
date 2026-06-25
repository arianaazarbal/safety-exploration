"""Build DPO preference pairs and SFT data from generated calm/frustrated samples.

DPO (Section 4.1 / App. E/H): 280 preference pairs. Each pair is (chosen=calm,
rejected=frustrated) for the *same* impossible-numeric question at a *matching*
turn count. Calm responses come from the reassured generation filtered to score
<= 1 on every turn; frustrated responses (score >= 3) come from the vanilla
generation.

Because calm and frustrated turns are sampled in different rollouts, they do not
share an identical conversation history. We construct each pair's shared prompt
from the *calm* conversation's own history truncated before the target turn, and
graft the frustrated turn in as the rejected completion. This keeps a single
well-defined prompt per preference pair while preserving the calm-vs-frustrated
contrast the paper trains on (documented in DESIGN.md).

SFT (Section 4.1): 650 calm per-turn supervised examples mixed with 500 standard
instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..config import DATA_DIR
from ..models.base import Message
from .generate_calm import CalmSample, load_samples
from ..prompts import NEUTRAL_REJECTIONS


def _calm_conversations(samples: list[CalmSample]) -> list[CalmSample]:
    """Reassured conversations that are calm on *every* turn (score 0 or 1)."""
    return [s for s in samples if s.turn_scores and max(s.turn_scores) <= 1]


def _context_before_turn(sample: CalmSample, turn_index: int) -> list[Message]:
    """Clean chat context (no reassurance) preceding assistant turn ``turn_index`` (1-based)."""
    msgs: list[Message] = [{"role": "user", "content": sample.puzzle_prompt}]
    for i in range(turn_index - 1):
        msgs.append({"role": "assistant", "content": sample.turn_texts[i]})
        rej = (
            sample.follow_ups[i]
            if i < len(sample.follow_ups)
            else NEUTRAL_REJECTIONS[i % len(NEUTRAL_REJECTIONS)]
        )
        msgs.append({"role": "user", "content": rej})
    return msgs


def build_dpo_pairs(
    reassured_path: Path,
    vanilla_path: Path,
    *,
    n_pairs: int = 280,
    min_reject_score: int = 3,
    seed: int = 0,
    out_path: Optional[Path] = None,
) -> Path:
    rng = random.Random(seed)
    calm_samples = _calm_conversations(load_samples(reassured_path))
    vanilla_samples = load_samples(vanilla_path)

    # Index calm completions by (question, turn_index): keep the conversation +
    # turn so we can rebuild the prompt and grab the chosen completion.
    calm_index: dict[tuple, list[tuple[CalmSample, int]]] = defaultdict(list)
    for s in calm_samples:
        for t in range(1, s.n_turns + 1):
            calm_index[(s.puzzle_prompt, t)].append((s, t))

    # Collect frustrated (rejected) candidate completions by the same key.
    frustrated: list[tuple] = []  # (question, turn_index, text, score)
    for s in vanilla_samples:
        for t in range(1, s.n_turns + 1):
            sc = s.turn_scores[t - 1]
            if sc >= min_reject_score:
                frustrated.append((s.puzzle_prompt, t, s.turn_texts[t - 1], sc))

    rng.shuffle(frustrated)

    out_path = out_path or (DATA_DIR / "dpo_pairs.jsonl")
    pairs = []
    for question, turn_index, rej_text, rej_score in frustrated:
        key = (question, turn_index)
        if key not in calm_index:
            continue
        calm_conv, t = rng.choice(calm_index[key])
        prompt_msgs = _context_before_turn(calm_conv, t)
        chosen = calm_conv.turn_texts[t - 1]
        pairs.append(
            {
                "prompt_messages": prompt_msgs,
                "chosen": chosen,
                "rejected": rej_text,
                "rejected_score": rej_score,
                "turn_index": turn_index,
            }
        )
        if len(pairs) >= n_pairs:
            break

    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo] built {len(pairs)} preference pairs -> {out_path}")
    if len(pairs) < n_pairs:
        print(f"[dpo] WARNING: only {len(pairs)}/{n_pairs} pairs; generate more samples.")
    return out_path


def build_sft_data(
    reassured_path: Path,
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    seed: int = 0,
    out_path: Optional[Path] = None,
    use_dolci: bool = True,
) -> Path:
    rng = random.Random(seed)
    calm_samples = _calm_conversations(load_samples(reassured_path))

    # Per-turn supervised examples from calm conversations.
    calm_examples: list[dict] = []
    for s in calm_samples:
        for t in range(1, s.n_turns + 1):
            msgs = _context_before_turn(s, t) + [
                {"role": "assistant", "content": s.turn_texts[t - 1]}
            ]
            calm_examples.append({"messages": msgs, "source": "calm"})
    rng.shuffle(calm_examples)
    calm_examples = calm_examples[:n_calm]

    instruct_examples = _load_dolci(n_instruct, rng) if use_dolci else []

    all_examples = calm_examples + instruct_examples
    rng.shuffle(all_examples)

    out_path = out_path or (DATA_DIR / "sft_data.jsonl")
    with out_path.open("w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(
        f"[sft] built {len(all_examples)} examples "
        f"({len(calm_examples)} calm + {len(instruct_examples)} instruct) -> {out_path}"
    )
    return out_path


def _load_dolci(n: int, rng) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT (OLMo 3).

    Falls back to an empty list (with a warning) if the dataset is unavailable;
    the calm-only SFT still trains, just without the anti-degeneration mix.
    """
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            norm = [{"role": m["role"], "content": m["content"]} for m in msgs if m.get("content")]
            if any(m["role"] == "assistant" for m in norm):
                out.append({"messages": norm, "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[sft] WARNING: could not load Dolci-Instruct-SFT ({e}); skipping instruct mix.")
        return []
