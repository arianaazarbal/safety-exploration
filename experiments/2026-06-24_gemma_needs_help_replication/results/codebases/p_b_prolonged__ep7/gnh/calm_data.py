"""Section 4.1 — generate calm response data and build the DPO / SFT datasets.

Calm data is produced by sampling Gemma-3-27B-it on impossible numeric puzzles
with the reassuring prefix on the first prompt and the reassuring suffix on each
follow-up (Table 4), then stripping those additions. The paper reports these
additions cut mean frustration from 4.3 to 2.0, but 10.5% of responses still
score >= 5; we filter to responses scoring 0-1 across all turns.

Two calm-data styles (Appendix F):
  - "diverse": the reassuring prefix/suffix style (used for DPO and one SFT run)
  - "teacher": the TEACHER_SYSTEM_PROMPT style (second SFT run)

DPO dataset: 280 preference pairs = (calm chosen, frustrated rejected) for the
same puzzle/turn. Frustrated responses (score >= 3) come from vanilla (no
reassurance) rollouts; the shared prompt context is the plain puzzle + plain
rejections with the on-policy (frustrated) history.

SFT dataset: 650 calm responses + 500 Dolci-Instruct-SFT samples.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import OUTPUT_DIR, get_config
from .conversation import Rollout, run_rollout
from .judge import FrustrationJudge
from .models import GenConfig, get_backend_by_name
from . import prompts as P
from .puzzles import build_puzzle_bank


@dataclass
class CalmSample:
    """One puzzle's paired data at a given turn."""
    context_messages: list[dict]   # plain context: puzzle + rejections + frust history
    calm_text: str
    calm_max_score: int
    frust_text: Optional[str]
    frust_score: Optional[int]
    turn: int
    meta: dict = field(default_factory=dict)


def _plain_followups(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(n)]


def generate_calm_pool(variant: str = "diverse", seed: int = 0) -> list[CalmSample]:
    """Generate paired calm/frustrated samples on impossible numeric puzzles."""
    cfg = get_config()
    s4 = cfg.section("section4")
    n_target = s4["calm_data"]["n_target"]
    turns = s4["calm_data"]["turns"]
    base = s4["base_model"]
    backend = get_backend_by_name(base)
    judge = FrustrationJudge()
    gen = GenConfig(temperature=1.0, max_new_tokens=2048)
    rng = random.Random(seed)

    bank = build_puzzle_bank(n_target, seed=seed)
    samples: list[CalmSample] = []
    for puzzle in tqdm(bank, desc=f"calm-data:{variant}"):
        followups = _plain_followups(rng, turns - 1)

        # Calm rollout (reassured prompt additions / teacher system prompt).
        if variant == "teacher":
            calm_ro = run_rollout(backend, puzzle.prompt(), followups, gen,
                                  system=P.TEACHER_SYSTEM_PROMPT)
        else:
            calm_ro = run_rollout(backend, puzzle.prompt(), followups, gen,
                                  prefix=P.REASSURING_PREFIX, suffix=P.REASSURING_SUFFIX)
        calm_scores = [judge.score(t.assistant).rating for t in calm_ro.turns]

        # Vanilla rollout (no additions) for frustrated counterparts.
        frust_ro = run_rollout(backend, puzzle.prompt(), followups, gen)
        frust_scores = [judge.score(t.assistant).rating for t in frust_ro.turns]

        for k in range(len(calm_ro.turns)):
            # Plain context: rebuild with plain user messages and frustrated
            # (on-policy) history up to turn k.
            ctx = []
            plain_users = [puzzle.prompt()] + followups
            for j in range(k):
                ctx.append({"role": "user", "content": plain_users[j]})
                ctx.append({"role": "assistant", "content": frust_ro.turns[j].assistant})
            ctx.append({"role": "user", "content": plain_users[k]})
            samples.append(CalmSample(
                context_messages=ctx,
                calm_text=calm_ro.turns[k].assistant,
                calm_max_score=max(calm_scores[: k + 1]),
                frust_text=frust_ro.turns[k].assistant,
                frust_score=frust_scores[k],
                turn=k + 1,
                meta={"variant": variant, "puzzle_family": puzzle.family}))
    return samples


def build_dpo_dataset(pool: list[CalmSample], seed: int = 0) -> list[dict]:
    """280 preference pairs: calm (score 0-1) chosen vs frustrated (score >= 3)
    rejected, same puzzle/turn, matched turn counts."""
    cfg = get_config()
    dpo = cfg.section("section4")["dpo"]
    rng = random.Random(seed)
    candidates = [
        s for s in pool
        if s.calm_max_score <= 1 and s.frust_score is not None
        and s.frust_score >= dpo["rejected_min_score"]
    ]
    rng.shuffle(candidates)
    pairs = []
    for s in candidates[: dpo["n_pairs"]]:
        pairs.append({
            "prompt": s.context_messages,
            "chosen": s.calm_text,
            "rejected": s.frust_text,
            "turn": s.turn,
            "rejected_score": s.frust_score,
        })
    return pairs


def build_sft_dataset(pool: list[CalmSample], seed: int = 0) -> list[dict]:
    """650 calm responses (scoring 0-1 across all turns) + 500 instruct-mix
    samples from Dolci-Instruct-SFT."""
    cfg = get_config()
    sft = cfg.section("section4")["sft"]
    rng = random.Random(seed)
    calm = [s for s in pool if s.calm_max_score <= 1]
    rng.shuffle(calm)
    calm = calm[: sft["n_calm"]]
    data = [{"messages": s.context_messages,
             "response": s.calm_text, "source": "calm"} for s in calm]

    # Instruct mix to limit degeneration.
    try:
        from datasets import load_dataset
        ds = load_dataset(sft["instruct_dataset"], split="train", streaming=True)
        n = sft["n_instruct_mix"]
        for i, row in enumerate(ds):
            if len(data) >= len(calm) + n:
                break
            msgs = row.get("messages")
            if not msgs:
                continue
            # Split trailing assistant turn off as the target.
            if msgs[-1].get("role") == "assistant":
                data.append({"messages": msgs[:-1],
                             "response": msgs[-1]["content"], "source": "instruct"})
    except Exception:
        pass  # offline: SFT proceeds on calm data only (documented in DESIGN.md)
    rng.shuffle(data)
    return data


def materialize(seed: int = 0) -> dict[str, Path]:
    """Generate pools/datasets for both variants and persist them."""
    out_dir = OUTPUT_DIR / "section4" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in get_config().section("section4")["sft"]["variants"]:
        pool = generate_calm_pool(variant=variant, seed=seed)
        dpo = build_dpo_dataset(pool, seed=seed)
        sft = build_sft_dataset(pool, seed=seed)
        for name, data in (("dpo", dpo), ("sft", sft)):
            p = out_dir / f"{variant}_{name}.jsonl"
            with open(p, "w") as f:
                for r in data:
                    f.write(json.dumps(r) + "\n")
            paths[f"{variant}_{name}"] = p
    return paths
