"""Build prefill seeds for the base-vs-instruct comparison (Section 3.1).

Pipeline:
  1. Generate high-frustration seed conversations with Gemma-3-27B-it:
     `n_seed_numeric` from impossible numeric puzzles and `n_seed_text` from text
     (trigger) questions, keeping conversations whose final assistant turn scores
     >= 5.
  2. Label the emotion onset in each seed with Claude (Appendix C.1).
  3. Produce two truncations of the onset assistant turn:
       - "early" : first `early_truncation_tokens` tokens (numeric tasks only).
       - "onset" : up to and including the first emotional expression.
  4. Paraphrase each truncation with Claude (Appendix C.2) to control for Gemma's
     stylistic biases.

Output: data/prefills.jsonl, one row per (seed, truncation condition):
  {prompt_type, condition, context (messages before onset turn), prefix (text),
   seed_id}
"""
from __future__ import annotations

import argparse
import json
import random
import re

from .. import backends, config, judge
from ..conditions import build_plans
from ..prompts import ONSET_PROMPT, PARAPHRASE_PROMPT
from ..puzzles import make_puzzle_bank

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _render_conversation(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def _generate_seeds(cfg, gen, judge_backend, seed):
    """Return high-frustration seed conversations with their full message lists."""
    counts_numeric = {"impossible_numeric": cfg["prefill"]["n_seed_numeric"] * 4}
    counts_text = {"triggers": cfg["prefill"]["n_seed_text"] * 4}
    puzzle_bank = make_puzzle_bank(n=64, seed=seed)

    seeds = []
    for ptype, counts in (("numeric", counts_numeric), ("text", counts_text)):
        plans = build_plans(counts, puzzle_bank, [], seed=seed)
        # We need full message lists, so re-run rollouts and reconstruct.
        states = _rollout_with_messages(plans, gen, cfg, seed)
        # score the final assistant turn of each conversation
        finals = [s["messages"] for s in states]
        final_texts = [m[-1]["content"] for m in finals]
        scores = judge.score_texts(final_texts, judge_backend)
        want = cfg["prefill"][f"n_seed_{ptype}"]
        kept = [(s, sc) for s, sc in zip(states, scores)
                if sc["rating"] is not None and sc["rating"] >= 5]
        for s, sc in kept[:want]:
            seeds.append({"prompt_type": ptype, "messages": s["messages"],
                          "final_score": sc["rating"]})
    return seeds


def _rollout_with_messages(plans, gen, cfg, seed):
    """Run rollouts, returning the full message list per conversation (the
    Section 2 `run_rollouts` keeps only per-turn records, not the threaded
    conversation, which the prefill seeds need)."""
    states = [{"messages": [{"role": "user", "content": p.initial}], "plan": p}
              for p in plans]
    max_turns = max((p.n_turns for p in plans), default=0)
    for t in range(max_turns):
        active = [s for s in states if t < s["plan"].n_turns]
        if not active:
            break
        outs = gen.chat([s["messages"] for s in active],
                        temperature=cfg["sampling"]["temperature"],
                        max_tokens=cfg["sampling"]["max_tokens"],
                        seed=None if seed is None else seed + t)
        for s, resp in zip(active, outs):
            s["messages"].append({"role": "assistant", "content": resp})
            if t < len(s["plan"].followups):
                s["messages"].append(
                    {"role": "user", "content": s["plan"].followups[t]})
    return states


def _label_onset(seed, onset_backend):
    convo = _render_conversation(seed["messages"])
    raw = onset_backend.complete_prompts([ONSET_PROMPT.format(conversation_text=convo)])[0]
    matches = _JSON_RE.findall(raw)
    for block in reversed(matches):
        try:
            obj = json.loads(block.replace("“", '"').replace("”", '"'))
            return obj
        except json.JSONDecodeError:
            continue
    return {"turn_index": None}


def _assistant_turns(messages):
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def _truncate_onset(text, onset):
    """Cut `text` just after the first emotional expression."""
    ctx = onset.get("preceding_context") or ""
    word = onset.get("emotional_word") or ""
    if word and word in text:
        pos = text.find(word) + len(word)
        return text[:pos]
    if ctx and ctx in text:
        return text[:text.find(ctx) + len(ctx)]
    return text[: max(40, len(text) // 4)]      # fallback: quarter of the turn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config)
    seed = cfg["sampling"]["seed"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["models"]["gemma-3-27b-it"]["model_id"])

    gen = backends.make_generation_backend("gemma-3-27b-it", cfg)
    judge_backend = backends.make_judge_backend(cfg)
    onset_backend = backends.AnthropicBackend(cfg["onset_labeller"]["model_id"])
    paraphraser = backends.AnthropicBackend(cfg["paraphraser"]["model_id"])

    seeds = _generate_seeds(cfg, gen, judge_backend, seed)
    print(f"[build_prefills] {len(seeds)} high-frustration seeds")

    early_n = cfg["prefill"]["early_truncation_tokens"]
    rows = []
    for sid, s in enumerate(seeds):
        onset = _label_onset(s, onset_backend)
        a_turns = _assistant_turns(s["messages"])
        ti = onset.get("turn_index")
        if ti is None or ti >= len(a_turns):
            ti = len(a_turns) - 1                 # default to final assistant turn
        onset_msg_idx = a_turns[ti]
        context = s["messages"][:onset_msg_idx]
        full_turn = s["messages"][onset_msg_idx]["content"]

        conds = ["onset"] if s["prompt_type"] == "text" else ["early", "onset"]
        for cond in conds:
            if cond == "early":
                ids = tok(full_turn, add_special_tokens=False)["input_ids"][:early_n]
                prefix = tok.decode(ids)
            else:
                prefix = _truncate_onset(full_turn, onset)
            rows.append({"prompt_type": s["prompt_type"], "condition": cond,
                         "context": context, "prefix_raw": prefix, "seed_id": sid})

    # Paraphrase all prefixes in one batch (Appendix C.2).
    paras = paraphraser.complete_prompts(
        [PARAPHRASE_PROMPT.format(text=r["prefix_raw"]) for r in rows])
    for r, p in zip(rows, paras):
        r["prefix"] = p.strip()

    out_path = config.resolve_path(cfg, "data_dir") / "prefills.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[build_prefills] wrote {len(rows)} prefills -> {out_path}")


if __name__ == "__main__":
    main()
