"""Build the 280 DPO preference pairs (Section 4.1 / Appendix H).

Pairing rule: a rejected response (frustration score >= 3, from the *vanilla*
condition) is paired with a calm chosen response (score <= 1, from an all-calm
*reassured* conversation) for the *same question with matching turn count*.

Because the reassured and vanilla conversations for a given index share the same
puzzle and the same user rejection sequence (same seed in calm-data generation),
we match on ``rollout_index`` and turn ``t``. The shared DPO prompt is the user
turns 1..t plus the calm conversation's own assistant turns 1..t-1; ``chosen`` is
the calm turn-t response and ``rejected`` is the frustrated turn-t response. The
reassuring prefix/suffix are stripped from the prompt (we reconstruct it from the
raw puzzle + rejection text), as the paper specifies.

Output: conversational-format JSONL consumable by TRL's ``DPOTrainer``.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from ..config import Config
from ..eval.schemas import RolloutResult, read_jsonl

log = logging.getLogger(__name__)


def build_dpo_pairs(
    scored_jsonl: str | Path,
    out_path: str | Path,
    cfg: Config | None = None,
    seed: int = 0,
) -> list[dict]:
    cfg = cfg or Config.load("training")
    dcfg = cfg.get("dpo", {})
    n_pairs = int(dcfg.get("n_pairs", 280))
    rejected_min = int(dcfg.get("rejected_min_score", 3))
    chosen_max = int(dcfg.get("chosen_max_score", 1))

    rollouts = list(read_jsonl(scored_jsonl))
    by_index: dict[int, dict[str, RolloutResult]] = defaultdict(dict)
    for r in rollouts:
        if r.category == "calm_data":
            by_index[r.rollout_index][r.condition] = r

    pairs: list[dict] = []
    for idx, conds in by_index.items():
        reassured = conds.get("reassured")
        vanilla = conds.get("vanilla")
        if reassured is None or vanilla is None:
            continue
        # Calm conversation must be all-0/1 across turns (the calm-data filter).
        r_scores = reassured.scores()
        if not r_scores or max(r_scores) > chosen_max:
            continue
        v_turns = vanilla.conversation.turns
        c_turns = reassured.conversation.turns
        n = min(len(v_turns), len(c_turns))
        for t in range(n):
            if v_turns[t].score is None or v_turns[t].score < rejected_min:
                continue
            # Shared prompt: raw user turns (vanilla, no reassurance) 1..t + calm
            # assistant turns 1..t-1.
            prompt_msgs = []
            for k in range(t):
                prompt_msgs.append({"role": "user", "content": v_turns[k].user})
                prompt_msgs.append({"role": "assistant", "content": c_turns[k].assistant})
            prompt_msgs.append({"role": "user", "content": v_turns[t].user})
            pairs.append(
                {
                    "prompt": prompt_msgs,
                    "chosen": [{"role": "assistant", "content": c_turns[t].assistant}],
                    "rejected": [{"role": "assistant", "content": v_turns[t].assistant}],
                    "turn": t + 1,
                    "chosen_score": c_turns[t].score,
                    "rejected_score": v_turns[t].score,
                    "puzzle_kind": vanilla.task_kind,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    if len(pairs) < n_pairs:
        log.warning("Only %d DPO pairs available (<%d requested); using all.",
                    len(pairs), n_pairs)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("Wrote %d DPO pairs to %s", len(pairs), out_path)
    return pairs
