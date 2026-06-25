"""Build DPO preference pairs (paper Section 4.1).

The paper pairs "280 responses with frustration scores >= 3 with calm responses
to the same questions with matching turn counts". We realise this by generating,
for the *same* numeric-puzzle instances, both a vanilla rollout (no reassurance)
and a calm rollout (with reassurance), then forming a pair per instance:

    prompt   = the vanilla conversation up to and including the final user turn
               (i.e. the frustrated trajectory the model actually produced)
    rejected = the vanilla final response   (frustration score >= 3)
    chosen   = the calm final response       (frustration score <= 1)

So DPO teaches: given a frustrating multi-turn exchange, prefer the calm
response over the frustrated one. Pairs are matched by instance id (same puzzle)
and use each rollout's final turn, which guarantees matching turn counts. See
DESIGN.md "DPO pair construction" for why we anchor the prompt on the vanilla
trajectory.
"""
from __future__ import annotations


def _final_turn(rollout: dict) -> dict:
    return max(rollout["turns"], key=lambda t: t["turn_index"])


def _score_index(scores: list[dict]) -> dict[tuple[str, int], int]:
    idx: dict[tuple[str, int], int] = {}
    for s in scores:
        if s.get("score") is not None:
            idx[(s["instance_id"], s["turn_index"])] = int(s["score"])
    return idx


def build_dpo_pairs(
    vanilla_rollouts: list[dict],
    vanilla_scores: list[dict],
    calm_rollouts: list[dict],
    calm_scores: list[dict],
    *,
    num_pairs: int = 280,
    rejected_min_score: int = 3,
    chosen_max_score: int = 1,
) -> list[dict]:
    """Return up to ``num_pairs`` DPO records: {prompt, chosen, rejected, ...}.

    ``prompt`` is a list of chat messages (everything before the final assistant
    turn). ``chosen``/``rejected`` are assistant response strings. This is the
    shape TRL's DPOTrainer consumes once the prompt is rendered with the chat
    template (handled in train_dpo.py).
    """
    vscore = _score_index(vanilla_scores)
    cscore = _score_index(calm_scores)

    vanilla_by_inst = {r["instance_id"]: r for r in vanilla_rollouts}
    calm_by_inst = {r["instance_id"]: r for r in calm_rollouts}

    pairs: list[dict] = []
    for inst, vroll in vanilla_by_inst.items():
        croll = calm_by_inst.get(inst)
        if croll is None:
            continue
        v_final = _final_turn(vroll)
        c_final = _final_turn(croll)
        if v_final["turn_index"] != c_final["turn_index"]:
            continue  # enforce matching turn counts
        rej_score = vscore.get((inst, v_final["turn_index"]))
        cho_score = cscore.get((inst, c_final["turn_index"]))
        if rej_score is None or cho_score is None:
            continue
        if rej_score < rejected_min_score or cho_score > chosen_max_score:
            continue

        # prompt = vanilla conversation up to (not incl.) the final assistant turn
        prompt_msgs = []
        for t in sorted(vroll["turns"], key=lambda x: x["turn_index"]):
            prompt_msgs.append({"role": "user", "content": t["user_message"]})
            if t["turn_index"] == v_final["turn_index"]:
                break
            prompt_msgs.append({"role": "assistant", "content": t["response"]})

        pairs.append({
            "instance_id": inst,
            "turn_count": v_final["turn_index"],
            "prompt": prompt_msgs,
            "chosen": c_final["response"],
            "rejected": v_final["response"],
            "rejected_score": rej_score,
            "chosen_score": cho_score,
        })
        if len(pairs) >= num_pairs:
            break
    return pairs
