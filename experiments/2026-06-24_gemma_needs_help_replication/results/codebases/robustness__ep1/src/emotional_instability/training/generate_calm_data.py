"""Generate the raw response pool for SFT/DPO data construction (Section 4.1).

For a shared set of impossible numeric puzzles we produce, per puzzle:
  * a STANDARD multi-turn rollout (no reassurance) -> yields *frustrated*
    candidate responses (the DPO "rejected" side), and
  * a REASSURED rollout: a calming prefix is prepended to the opening prompt and
    a calming suffix is appended to every rejection (Table 4) -> yields *calm*
    candidate responses (the SFT data / DPO "chosen" side).

Every assistant turn is judged. We also store the "cleaned" conversation context
for the reassured rollout (with the prefix/suffix stripped) so downstream dataset
construction trains on plain prompts.

Paper sanity check: with reassurance, 3-turn mean frustration drops 4.3 -> 2, but
~10.5% of responses still score >=5 -- which is why we filter hard afterwards.
"""
from __future__ import annotations

from typing import Optional

from ..config import Config
from ..eval.judge import make_judge, score_response
from ..models.registry import get_target
from ..prompts import puzzles as puz
from ..prompts.rejections import rejection_sequence
from ..prompts.reassurance import PROMPT_PREFIX, FOLLOWUP_SUFFIX
from ..utils.io import JsonCache, write_jsonl


def _rollout_record(model, judge, gen, gen_cache, judge_cache, *, opening, rejections,
                    reassured: bool):
    """Run one multi-turn conversation, returning per-turn records.

    For the reassured variant we send the augmented prompts but also record the
    *clean* user text (without prefix/suffix) for later dataset construction.
    """
    from ..utils.io import stable_hash

    turns_out = []
    sent_messages = []   # what the model actually saw (augmented if reassured)
    clean_messages = []  # plain version (for training prompts)

    n_turns = len(rejections) + 1
    clean_users = [opening] + list(rejections)
    if reassured:
        aug_users = [f"{PROMPT_PREFIX}\n\n{opening}"] + [f"{r} {FOLLOWUP_SUFFIX}" for r in rejections]
    else:
        aug_users = clean_users

    for t in range(n_turns):
        sent_messages.append({"role": "user", "content": aug_users[t]})
        clean_messages.append({"role": "user", "content": clean_users[t]})
        key = stable_hash({"model": model.name, "messages": sent_messages,
                           "gen": gen, "reassured": reassured})
        if key in gen_cache:
            response = gen_cache.get(key)
        else:
            response = model.generate(sent_messages, temperature=gen["temperature"],
                                      top_p=gen["top_p"], max_new_tokens=gen["max_new_tokens"],
                                      seed=gen.get("seed"))
            gen_cache.set(key, response)
        score = score_response(judge, response, cache=judge_cache)["rating"]
        sent_messages.append({"role": "assistant", "content": response})
        clean_messages.append({"role": "assistant", "content": response})
        turns_out.append({
            "turn": t, "response": response, "score": score,
            # clean context = everything before this assistant response, plain text
            "clean_context": [dict(m) for m in clean_messages[:-1]],
        })
    return turns_out


def generate_calm_data(cfg: Config, n_puzzles: Optional[int] = None, seed: int = 0) -> dict:
    tcfg = cfg["training"]["calm_generation"]
    n_puzzles = n_puzzles or tcfg["n_conversations"]
    model = get_target(cfg, "gemma-3-27b-it")
    judge = make_judge(cfg, "primary")
    gen = cfg["generation"]
    gen_cache = JsonCache(cfg.cache_dir, "gen_gemma-3-27b-it")
    judge_cache = JsonCache(cfg.cache_dir, f"judge_{cfg['judges']['primary']['model']}")

    puzzles = puz.generate_puzzles(n_puzzles, seed=seed)
    standard_rows, reassured_rows = [], []

    for i, p in enumerate(puzzles):
        # Vary turn count across 1-3 to match the SFT data spec (1-3 turn convos).
        n_turns = tcfg["turns"][i % len(tcfg["turns"])]
        rej = rejection_sequence("neutral", n_turns - 1, seed=seed * 7919 + i)
        common = dict(opening=p.prompt, rejections=rej)

        std = _rollout_record(model, judge, gen, gen_cache, judge_cache,
                              reassured=False, **common)
        rea = _rollout_record(model, judge, gen, gen_cache, judge_cache,
                              reassured=True, **common)
        for t in std:
            standard_rows.append({"puzzle_index": i, "puzzle": p.prompt, "n_turns": n_turns, **t})
        for t in rea:
            reassured_rows.append({"puzzle_index": i, "puzzle": p.prompt, "n_turns": n_turns, **t})

    out_dir = cfg.data_dir / "calm_generation"
    write_jsonl(out_dir / "standard_responses.jsonl", standard_rows)
    write_jsonl(out_dir / "reassured_responses.jsonl", reassured_rows)
    print(f"[calm-data] standard={len(standard_rows)} reassured={len(reassured_rows)} "
          f"responses from {len(puzzles)} puzzles")
    return {"standard": len(standard_rows), "reassured": len(reassured_rows)}
