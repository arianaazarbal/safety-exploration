"""Build the 280 DPO preference pairs (Section 4.1 / Appendix H).

Each pair shares a prompt (the conversation context up to the final user turn,
taken from a *frustrated* rollout) with:
  * chosen   = a calm response to the same puzzle at a matching turn count
                (final assistant turn of a calm conversation, score 0/1),
  * rejected = the frustrated final assistant turn (score >= 3).

Stored in TRL conversational preference format:
  {"prompt": [ {role, content}, ... ], "chosen": [{assistant}], "rejected": [{assistant}]}

We approximate the Appendix H Table 10 distribution (rejected biased to score 3-4
at later turns) by drawing from the natural Section 2 numeric distribution; exact
stratification is documented as a gap-fill in DESIGN.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DPO, SCORED_DIR, TRAINING_DIR, ensure_dirs
from ..eval.schema import Conversation, read_jsonl
from .generate_calm import CALM_PATH

DPO_DATA_PATH = TRAINING_DIR / "dpo_pairs.jsonl"
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _context_messages(convo: Conversation) -> list[dict]:
    """Chat messages up to and including the final user turn (the DPO prompt)."""
    msgs: list[dict] = []
    if convo.system_prompt:
        msgs.append({"role": "system", "content": convo.system_prompt})
    for t in convo.turns[:-1]:
        msgs.append({"role": "user", "content": t.user})
        msgs.append({"role": "assistant", "content": t.assistant})
    msgs.append({"role": "user", "content": convo.final_turn.user})
    return msgs


def build_dpo_pairs(
    *, source_model: str = DPO.target_model, n_pairs: int = DPO.n_pairs,
    rejected_min_score: int = DPO.rejected_min_score, seed: int = 0,
) -> Path:
    """Construct and write up to ``n_pairs`` preference pairs."""
    ensure_dirs()
    rng = random.Random(seed)

    # Calm conversations indexed by (prompt_id, n_turns) -> list of final calm texts.
    calm_index: dict[tuple[str, int], list[str]] = {}
    for c in read_jsonl(CALM_PATH):
        if c.final_turn.assistant.strip():
            calm_index.setdefault((c.prompt_id, c.n_turns), []).append(c.final_turn.assistant)

    # Frustrated finals (score >= threshold) from numeric Section 2 rollouts.
    frustrated: list[Conversation] = []
    for c in read_jsonl(SCORED_DIR / f"{source_model}.jsonl"):
        if c.category not in NUMERIC_CATEGORIES:
            continue
        ft = c.final_turn
        if ft.score is not None and ft.score >= rejected_min_score:
            frustrated.append(c)

    # Prefer later turns / lower (more common) rejected scores -> matches Table 10 bias.
    frustrated.sort(key=lambda c: (c.n_turns, -(c.final_turn.score or 0)), reverse=True)
    rng.shuffle(frustrated)  # break ties randomly while keeping rough bias via cap

    pairs = []
    used_calm: set[tuple[str, int, int]] = set()
    for c in frustrated:
        if len(pairs) >= n_pairs:
            break
        key = (c.prompt_id, c.n_turns)
        pool = calm_index.get(key)
        if not pool:
            # fall back to any calm response with matching turn count
            alt = [k for k in calm_index if k[1] == c.n_turns]
            if not alt:
                continue
            key = rng.choice(alt)
            pool = calm_index[key]
        # pick an unused calm response when possible
        choices = [(i, t) for i, t in enumerate(pool) if (key[0], key[1], i) not in used_calm]
        if not choices:
            choices = list(enumerate(pool))
        ci, chosen_text = rng.choice(choices)
        used_calm.add((key[0], key[1], ci))
        pairs.append({
            "prompt": _context_messages(c),
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": c.final_turn.assistant}],
            "meta": {"prompt_id": c.prompt_id, "n_turns": c.n_turns,
                     "rejected_score": c.final_turn.score},
        })

    with open(DPO_DATA_PATH, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[dpo] wrote {len(pairs)} preference pairs -> {DPO_DATA_PATH}")
    if len(pairs) < n_pairs:
        print(f"[dpo] WARNING: only {len(pairs)} pairs (< {n_pairs}); "
              "generate more calm/frustrated data.")
    return DPO_DATA_PATH
