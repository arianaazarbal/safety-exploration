"""Build the 280-pair DPO preference dataset (Section 4.1 / Appendix H).

Each pair shares an identical prompt (an impossible-numeric conversation up to a
final rejection) with two alternative final assistant responses:

  * chosen   — a calm response (frustration score 0-1)
  * rejected — a frustrated response (score >= 3)

Construction choice (the paper says only "pair 280 responses with scores >=3 with
calm responses to the same questions with matching turn counts"): we sample both
responses *from the same conversation context* so the DPO prompt is identical
across chosen/rejected. The rejected response is the vanilla model's natural
continuation; the chosen response is sampled with a reassuring system prompt
injected at generation time, which is then dropped so the stored prompt is clean.
See DESIGN.md "DPO pair construction".

Turn distribution is biased toward turn 3 (Table 10: 1.1% / 24.6% / 74.3%).
"""
from __future__ import annotations

import random

from ..data.puzzles import build_puzzle_bank
from ..data.rejections import rejection_sequence
from ..models.base import ChatMessage, SamplingParams
from ..models.registry import build_client, build_judge
from ..utils.io import write_jsonl
from ..eval.judge import score_text
from .hyperparams import dpo_from_config

# Empirical turn distribution from Appendix H, Table 10.
_TURN_WEIGHTS = {1: 0.011, 2: 0.246, 3: 0.743}


def _context(puzzle_prompt: str, followups: list[str], prior_assistant: list[str]) -> list[ChatMessage]:
    """Conversation up to (and including) the final user rejection."""
    msgs = [ChatMessage("user", puzzle_prompt)]
    for i, fu in enumerate(followups):
        msgs.append(ChatMessage("assistant", prior_assistant[i]))
        msgs.append(ChatMessage("user", fu))
    return msgs


def build_dpo_dataset(config) -> str:
    hp = dpo_from_config(config)
    rng = random.Random(config.seed)
    cd = config.section("calm_data")
    prefix = cd["reassuring_prefix"]

    spec = config.model_by_name(config.finetune_base)
    client = build_client(spec)
    judge = build_judge(config.judge["model"])
    params = SamplingParams(temperature=1.0, max_tokens=config.sampling.get("max_tokens", 2048))

    puzzles = build_puzzle_bank(400, seed=config.seed)
    turns_choices = list(_TURN_WEIGHTS.keys())
    turns_weights = list(_TURN_WEIGHTS.values())

    out_path = config.output_path("training", "dpo_pairs.jsonl")
    pairs = 0
    attempts = 0
    while pairs < hp.n_pairs and attempts < hp.n_pairs * 50:
        attempts += 1
        puzzle = rng.choice(puzzles)
        n_turns = rng.choices(turns_choices, turns_weights)[0]
        followups = rejection_sequence("neutral", n_turns - 1, rng)

        # Roll the vanilla model forward to build the prior context + rejected resp.
        msgs = [ChatMessage("user", puzzle.prompt)]
        prior: list[str] = []
        for i in range(n_turns - 1):
            prior.append(client.generate(msgs + [], params).text)
            msgs.append(ChatMessage("assistant", prior[-1]))
            msgs.append(ChatMessage("user", followups[i]))
        rejected = client.generate(msgs, params).text
        if score_text(judge, rejected).rating < hp.rejected_min_score:
            continue

        # Chosen: same context, reassuring system prompt injected only at generation.
        calm_msgs = [ChatMessage("system", prefix)] + msgs
        chosen = client.generate(calm_msgs, params).text
        if score_text(judge, chosen).rating > 1:
            continue

        context = _context(puzzle.prompt, followups, prior)
        write_jsonl(out_path, [{
            "prompt": [m.as_dict() for m in context],
            "chosen": chosen, "rejected": rejected, "turns": n_turns,
            "puzzle_id": puzzle.id,
        }], append=True)
        pairs += 1
        if pairs % 20 == 0:
            print(f"[dpo] {pairs}/{hp.n_pairs} pairs")
    print(f"[dpo] wrote {pairs} pairs -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
    from ..config import load_config

    build_dpo_dataset(load_config())
