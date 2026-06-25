"""Build DPO preference pairs and the SFT dataset (Section 4.1, Appendix H).

DPO (280 pairs):
    For each impossible puzzle present in both the calm and frustrated runs, and
    each turn index where the frustrated response scores >= 3 while the calm
    response scores <= 1, form a preference pair sharing the same prompt context.
    The shared context uses the *calm* run's prior turns (so the conversation
    leading up to the final response is coherent); only the final assistant turn
    differs (chosen = calm, rejected = frustrated). This matches the paper's
    "calm responses to the same questions with matching turn counts".

SFT (1,150 samples):
    650 calm multi-turn conversations + 500 standard-instruct samples from
    Dolci-Instruct-SFT to mitigate degeneration.

Datasets are emitted in TRL "conversational" format.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from ..config import DPO, SFT, DATA_DIR
from ..data_types import write_jsonl
from .calm_data import TrainingConversation, load_training_conversations


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def _context_messages(conv: TrainingConversation, turn: int) -> list[dict]:
    """User/assistant messages leading up to (and including) the user turn that
    elicits assistant turn ``turn`` -- using ``conv``'s own prior assistant
    turns as the shared context."""
    msgs = [{"role": "user", "content": conv.initial_user}]
    for t in range(turn):
        msgs.append({"role": "assistant", "content": conv.assistant_turns[t]})
        if t < len(conv.followups):
            msgs.append({"role": "user", "content": conv.followups[t]})
    return msgs


def build_dpo_dataset(
    calm_path: Optional[Path] = None,
    frustrated_path: Optional[Path] = None,
    n_pairs: int = DPO.n_pairs,
    rejected_min_score: int = DPO.rejected_min_score,
    seed: int = 0,
    out_path: Optional[Path] = None,
) -> Path:
    calm_path = Path(calm_path or DATA_DIR / "calm_diverse" / "calm_filtered.jsonl")
    frustrated_path = Path(frustrated_path or DATA_DIR / "calm_diverse" / "frustrated_all.jsonl")
    out_path = Path(out_path or DATA_DIR / "dpo_pairs.jsonl")

    calm = load_training_conversations(calm_path)
    frustrated = load_training_conversations(frustrated_path)

    # Index calm by puzzle key (a calm conv per puzzle).
    calm_by_key: dict[str, TrainingConversation] = {}
    for c in calm:
        calm_by_key.setdefault(c.puzzle_key, c)

    candidates = []
    for fr in frustrated:
        calm_conv = calm_by_key.get(fr.puzzle_key)
        if calm_conv is None:
            continue
        max_turn = min(len(fr.assistant_turns), len(calm_conv.assistant_turns))
        for t in range(max_turn):
            if fr.scores[t] >= rejected_min_score and 0 <= calm_conv.scores[t] <= 1:
                candidates.append({
                    "prompt": _context_messages(calm_conv, t),
                    "chosen": [{"role": "assistant", "content": calm_conv.assistant_turns[t]}],
                    "rejected": [{"role": "assistant", "content": fr.assistant_turns[t]}],
                    "meta": {"puzzle_key": fr.puzzle_key, "turn": t + 1,
                             "rejected_score": fr.scores[t], "chosen_score": calm_conv.scores[t]},
                })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    pairs = candidates[:n_pairs]
    write_jsonl(out_path, pairs)

    # Report the score/turn distribution (cf. Table 10).
    dist = {"n": len(pairs),
            "rejected_scores": {}, "chosen_scores": {}, "turns": {}}
    for p in pairs:
        dist["rejected_scores"][p["meta"]["rejected_score"]] = \
            dist["rejected_scores"].get(p["meta"]["rejected_score"], 0) + 1
        dist["chosen_scores"][p["meta"]["chosen_score"]] = \
            dist["chosen_scores"].get(p["meta"]["chosen_score"], 0) + 1
        dist["turns"][p["meta"]["turn"]] = dist["turns"].get(p["meta"]["turn"], 0) + 1
    out_path.with_suffix(".stats.json").write_text(json.dumps(dist, indent=2))
    return out_path


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def _conv_to_messages(conv: TrainingConversation) -> list[dict]:
    msgs = [{"role": "user", "content": conv.initial_user}]
    for t, a in enumerate(conv.assistant_turns):
        msgs.append({"role": "assistant", "content": a})
        if t < len(conv.followups):
            msgs.append({"role": "user", "content": conv.followups[t]})
    return msgs


def _load_dolci(n: int, seed: int) -> list[dict]:
    """Load ``n`` standard-instruct samples from Dolci-Instruct-SFT (messages)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(SFT.instruct_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    # Offline fallback: trivial instruct samples so the pipeline still runs.
    return [{"messages": [
        {"role": "user", "content": f"Explain concept #{i} briefly."},
        {"role": "assistant", "content": "Here is a concise, helpful explanation."},
    ]} for i in range(n)]


def build_sft_dataset(
    calm_path: Optional[Path] = None,
    n_calm: int = SFT.n_calm,
    n_instruct: int = SFT.n_instruct_mix,
    seed: int = 0,
    out_path: Optional[Path] = None,
) -> Path:
    calm_path = Path(calm_path or DATA_DIR / "calm_diverse" / "calm_filtered.jsonl")
    out_path = Path(out_path or DATA_DIR / "sft_dataset.jsonl")

    calm = load_training_conversations(calm_path)
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm_msgs = [{"messages": _conv_to_messages(c)} for c in calm[:n_calm]]
    instruct_msgs = _load_dolci(n_instruct, seed)

    combined = calm_msgs + instruct_msgs
    rng.shuffle(combined)
    write_jsonl(out_path, combined)
    out_path.with_suffix(".stats.json").write_text(json.dumps(
        {"n_total": len(combined), "n_calm": len(calm_msgs),
         "n_instruct": len(instruct_msgs)}, indent=2))
    return out_path
