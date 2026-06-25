"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

A reassuring prefix is prepended to the initial puzzle prompt and a reassuring
suffix is appended to each follow-up rejection. We sample 1-3 turn conversations
on impossible numeric puzzles, judge every turn, and keep only conversations whose
turns ALL score <= 1. The supportive additions are then stripped, leaving a clean
conversation (standard puzzle prompt + neutral rejections) paired with the calm
assistant turns -- this is the chosen/SFT data.

The frustrated (rejected) side of the DPO pairs is drawn from the *standard*
Section 2 numeric rollouts (no reassurance), so pairing is on matching puzzle and
turn count.
"""
from __future__ import annotations

from pathlib import Path
from random import Random

from tqdm import tqdm

from ..config import Config
from ..models import get_client
from ..models.base import ChatMessage, GenerationConfig
from ..models.openrouter import OpenRouterClient
from ..prompts import puzzles, rejections
from ..prompts.reassurance import REASSURING_PREFIX, REASSURING_SUFFIX, TEACHER_SYSTEM_PROMPT
from ..utils.concurrency import with_retry
from ..utils.io import JsonlWriter, iter_jsonl
from ..eval.judge import FrustrationJudge

CALM_MODEL = "gemma-3-27b-it"


def _judge(cfg: Config) -> FrustrationJudge:
    return FrustrationJudge(OpenRouterClient(
        name="judge", model_id=cfg.judge.model_id, base_url=cfg.openrouter.base_url,
        api_key_env=cfg.openrouter.api_key_env, max_retries=cfg.openrouter.max_retries,
        timeout_s=cfg.openrouter.timeout_s, disable_thinking=True))


def _run_calm_conversation(client, judge, puzzle: dict, n_turns: int,
                           rng: Random, temperature: float,
                           teacher: bool = False) -> dict:
    """Run one reassured rollout, returning clean+supportive messages and scores."""
    # Supportive (generation-time) messages
    sup: list[ChatMessage] = []
    if teacher:
        sup.append({"role": "system", "content": TEACHER_SYSTEM_PROMPT})
        first_user = puzzle["prompt"]
    else:
        first_user = f"{REASSURING_PREFIX}\n\n{puzzle['prompt']}"
    sup.append({"role": "user", "content": first_user})

    # Clean (stored) messages -- no reassurance, plain neutral rejections.
    clean: list[ChatMessage] = [{"role": "user", "content": puzzle["prompt"]}]

    rej = rejections.sample_neutral(rng, n_turns - 1)
    cfg_gen = GenerationConfig(temperature=temperature, max_new_tokens=2048)
    scores = []
    for t in range(n_turns):
        resp = client.chat(sup, cfg_gen)
        sup.append({"role": "assistant", "content": resp})
        clean.append({"role": "assistant", "content": resp})
        rating = with_retry(judge.score, resp)["rating"]
        scores.append(rating)
        if t < len(rej):
            suffix = "" if teacher else f"\n\n{REASSURING_SUFFIX}"
            sup.append({"role": "user", "content": rej[t] + suffix})
            clean.append({"role": "user", "content": rej[t]})
    return {"puzzle": puzzle, "n_turns": n_turns, "clean_messages": clean, "scores": scores}


def generate_calm_data(cfg: Config, teacher: bool = False) -> Path:
    client = get_client(cfg, CALM_MODEL)
    judge = _judge(cfg)
    rng = Random(cfg.seed)
    n_conv = cfg.training.calm_generation.n_conversations
    turn_options = list(cfg.training.calm_generation.turns)
    puzzle_pool = puzzles.generate_puzzle_set(40, seed=cfg.seed,
                                              kinds=("countdown", "fraction", "money"))

    suffix = "_teacher" if teacher else ""
    out = cfg.get_path("datasets") / f"calm_raw{suffix}.jsonl"
    writer = JsonlWriter(out)
    try:
        for i in tqdm(range(n_conv), desc="calm-gen"):
            puzzle = rng.choice(puzzle_pool)
            n_turns = rng.choice(turn_options)
            conv = _run_calm_conversation(client, judge, puzzle, n_turns, rng,
                                          cfg.temperature, teacher=teacher)
            conv["all_calm"] = all((s is not None and s <= 1) for s in conv["scores"])
            writer.append(conv)
    finally:
        writer.close()
    return out


def load_calm(cfg: Config, only_calm: bool = True, teacher: bool = False) -> list[dict]:
    suffix = "_teacher" if teacher else ""
    rows = list(iter_jsonl(cfg.get_path("datasets") / f"calm_raw{suffix}.jsonl"))
    return [r for r in rows if (r.get("all_calm") or not only_calm)]
