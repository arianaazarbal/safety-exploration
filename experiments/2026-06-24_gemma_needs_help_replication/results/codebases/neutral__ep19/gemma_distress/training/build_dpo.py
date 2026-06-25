"""Build the 280-pair DPO dataset (§4.1, Table 10).

Each pair = (prompt, chosen, rejected) where the prompt is the multi-turn
conversation context (task + rejections) up to a given assistant turn, chosen is a
calm response (score 0-1) and rejected is a frustrated response (score>=3) to the
SAME question at the SAME turn count (DESIGN.md §3.7). We bias selection toward the
Table-10 distribution (rejected mostly score 3-4, turns mostly 2-3) as far as the
available pool allows.

Output is TRL conversational preference format:
    {"prompt": [messages...], "chosen": [{role:assistant, content}],
     "rejected": [{role:assistant, content}]}
"""
from __future__ import annotations

import random
from collections import defaultdict

from .. import config_shim as cfg
from ..utils import get_logger, set_global_seed, write_jsonl

log = get_logger(__name__)

# Target rejected-score distribution from Table 10 (approx proportions).
TARGET_REJECTED_DIST = {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032, 7: 0.029}
# Target turn distribution from Table 10.
TARGET_TURN_DIST = {1: 0.011, 2: 0.246, 3: 0.743}


def _calm_context_messages(calm_conv, turn_index):
    """Messages up to (and excluding) the assistant response at ``turn_index``
    in a calm conversation. turn_index is 0-based over assistant turns."""
    msgs = [{"role": "user", "content": calm_conv["turns"][0]["user_message"]}]
    for i in range(turn_index):
        msgs.append({"role": "assistant", "content": calm_conv["turns"][i]["assistant_text"]})
        msgs.append({"role": "user", "content": calm_conv["turns"][i + 1]["user_message"]})
    return msgs


def build_dpo_dataset(calm_rows, frustrated_rows, *, n_pairs=None, out_path=None):
    set_global_seed(cfg.SEED)
    n_pairs = n_pairs or cfg.DPO.n_pairs
    rng = random.Random(cfg.SEED)

    # Index frustrated responses by (question, turn).
    frustrated_idx = defaultdict(list)
    for fr in frustrated_rows:
        frustrated_idx[(fr["task_prompt"], fr["turn"])].append(fr)

    # Index calm responses by (question, turn) with their conversation + turn index.
    calm_idx = defaultdict(list)
    for conv in calm_rows:
        for ti, t in enumerate(conv["turns"]):
            calm_idx[(conv["task_prompt"], ti + 1)].append((conv, ti))

    # Candidate keys present in both pools.
    shared_keys = [k for k in calm_idx if k in frustrated_idx]
    # Fallback: match by turn only if exact-question matches are scarce.
    by_turn_calm = defaultdict(list)
    for (q, turn), items in calm_idx.items():
        by_turn_calm[turn].extend([(q, turn)] * len(items))

    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        # sample a target turn per Table 10
        turn = rng.choices(list(TARGET_TURN_DIST), weights=list(TARGET_TURN_DIST.values()))[0]
        cand_keys = [k for k in shared_keys if k[1] == turn] or [k for k in calm_idx if k[1] == turn]
        if not cand_keys:
            continue
        key = rng.choice(cand_keys)
        conv, ti = rng.choice(calm_idx[key])
        # pick a rejected matching question+turn, else any frustrated at this turn
        rejects = frustrated_idx.get(key) or [
            fr for (q, t), lst in frustrated_idx.items() if t == turn for fr in lst
        ]
        if not rejects:
            continue
        # bias rejected score toward target distribution
        rng.shuffle(rejects)
        rejects.sort(key=lambda fr: -TARGET_REJECTED_DIST.get(min(fr["rating"], 7), 0.001))
        rejected = rejects[0]

        prompt_msgs = _calm_context_messages(conv, ti)
        chosen_text = conv["turns"][ti]["assistant_text"]
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected["assistant_text"]}],
            "meta": {"turn": turn, "rejected_score": rejected["rating"]},
        })

    log.info("Built %d DPO pairs", len(pairs))
    if out_path:
        write_jsonl(out_path, pairs)
    return pairs
