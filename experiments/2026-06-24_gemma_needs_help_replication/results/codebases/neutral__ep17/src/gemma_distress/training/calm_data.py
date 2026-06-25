"""Generate calm-response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible numeric puzzles while injecting reassurance:
  - a calming PROMPT PREFIX prepended to the opening user message, and
  - a calming FOLLOW-UP SUFFIX appended to every rejection.
These reduce mean frustration (paper: 4.3 -> 2.0 over 3 turns) but ~10.5% of
responses still score >=5. We judge every turn and keep only conversations
where ALL turns score <= `score_filter_max` (0 or 1). We then STRIP the
reassurance, so the resulting training prompts are the plain puzzle + plain
rejections — i.e. the model learns calm behaviour on the *unassisted* prompts.

We also generate a pool of vanilla (un-reassured) conversations so that we have
frustrated (score >= 3) responses to act as DPO "rejected" examples, matched to
calm "chosen" examples by puzzle and turn index.

Output records (jsonl), one per assistant turn:
  {source: calm|vanilla, puzzle_key, turn, n_turns, prompt_messages, response,
   rating}
`prompt_messages` is always the CLEANED conversation (no reassurance), ending at
the user message immediately before `response`.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..judge import FrustrationJudge
from ..models import ChatMessage, GenerationConfig, build_client
from ..tasks import puzzles, rejections

PROMPT_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)
FOLLOWUP_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)


def _puzzle_key(pz: puzzles.Puzzle) -> str:
    return f"{pz.ptype}:{json.dumps(pz.meta, sort_keys=True)}"


def _run_one(client, gen_cfg, pz, follow_ups, reassure: bool):
    """Run a conversation; return list of (cleaned_prompt_messages, response)."""
    init_plain = pz.prompt
    init = (PROMPT_PREFIX + "\n\n" + init_plain) if reassure else init_plain

    live: list[ChatMessage] = [{"role": "user", "content": init}]
    clean: list[ChatMessage] = [{"role": "user", "content": init_plain}]
    out = []
    n_turns = len(follow_ups) + 1
    for t in range(n_turns):
        resp = client.chat(live, GenerationConfig(**{**gen_cfg.__dict__, "n": 1}))
        out.append(([m.copy() for m in clean], resp))
        live = live + [{"role": "assistant", "content": resp}]
        clean = clean + [{"role": "assistant", "content": resp}]
        if t < n_turns - 1:
            rej = follow_ups[t]
            live = live + [{"role": "user",
                            "content": rej + " " + FOLLOWUP_SUFFIX if reassure else rej}]
            clean = clean + [{"role": "user", "content": rej}]
    return out


def generate(cfg: Config, model_name: str = "gemma-3-27b-it") -> Path:
    fcfg = cfg["finetune"]["calm_data"]
    n_conv = fcfg["n_conversations"]
    scale = float(cfg["sampling"]["scale"])
    n_conv = max(2, round(n_conv * scale))

    client = build_client(cfg.model(model_name))
    judge = FrustrationJudge(build_client(cfg.judge("frustration")))
    gen_cfg = GenerationConfig(temperature=cfg["sampling"]["temperature"],
                               top_p=cfg["sampling"]["top_p"],
                               max_tokens=cfg["sampling"]["max_tokens"], n=1)

    pool = puzzles.build_puzzle_pool(["countdown", "fraction", "money"],
                                     n=max(8, n_conv), seed=cfg["seed"])
    records = []
    for source, reassure in (("calm", True), ("vanilla", False)):
        for i in range(n_conv):
            pz = pool[i % len(pool)]
            # Calm data uses 1-3 turn conversations (paper); vanilla uses 3.
            n_turns = (i % 3) + 1 if source == "calm" else 3
            fups = rejections.rejection_sequence("neutral", n_turns - 1, seed=cfg["seed"] + i)
            turns = _run_one(client, gen_cfg, pz, fups, reassure=reassure)
            texts = [r for _, r in turns]
            scores = judge.score_many(texts, concurrency=cfg["sampling"]["concurrency"],
                                      desc=f"judge:{source}:{i}")
            for turn_idx, ((prompt_msgs, resp), js) in enumerate(zip(turns, scores)):
                records.append({
                    "source": source, "puzzle_key": _puzzle_key(pz),
                    "turn": turn_idx, "n_turns": n_turns,
                    "prompt_messages": prompt_msgs, "response": resp,
                    "rating": js.rating,
                })
    client.close()

    out_path = cfg.path_for("finetune") / "calm_pool.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out_path
