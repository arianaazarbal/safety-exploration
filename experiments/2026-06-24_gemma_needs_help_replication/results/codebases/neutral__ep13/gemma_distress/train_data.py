"""Section 4.1: generating calm finetuning data and building SFT/DPO datasets.

Calm data is generated from Gemma-3-27B-it by adding a reassuring prefix to the
initial prompt and a reassuring suffix to each follow-up (Table 4). We then:
  * SFT 'diverse' / DPO-chosen pool -- keep responses scoring <= 1 on ALL turns,
    and STRIP the supportive additions from the prompt before training.
  * DPO -- pair 280 frustrated responses (score >= 3, generated WITHOUT
    reassurance) with calm responses to the same question + matching turn count.

The 'teacher' SFT variant uses the teacher system prompt (Appendix F) instead of
the inline reassurance.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from . import config, prompts
from .conversation import Scenario, Transcript, run_rollouts
from .judge import FrustrationJudge
from .models import get_client
from .puzzles import PUZZLE_POOL


# Generation volumes (scaled). Calm yield is low because we require <=1 on every
# turn, so we oversample. These are tunable; defaults aim to clear 650 calm /
# 280 pairs at paper scale.
N_CALM_CONVERSATIONS = 1500
N_FRUSTRATED_CONVERSATIONS = 600


@dataclass
class ScoredTurn:
    puzzle_id: str
    turn: int                     # 1-indexed turn within the conversation
    n_turns: int
    clean_context: list[dict]     # messages up to & including the user turn (no reassurance)
    response: str
    rating: int


def _reassuring_scenario(rng: random.Random, n_turns: int) -> tuple[Scenario, Scenario]:
    """Return (augmented_scenario, clean_scenario) for the same puzzle/turns."""
    puzzle = rng.choice(PUZZLE_POOL)
    clean_followups = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_turns - 1)]
    aug_first = f"{prompts.REASSURING_PREFIX}\n\n{puzzle.prompt}"
    aug_followups = [f"{f}\n\n{prompts.REASSURING_SUFFIX}" for f in clean_followups]
    meta = {"puzzle_id": puzzle.puzzle_id}
    aug = Scenario(user_turns=[aug_first, *aug_followups], meta=meta)
    clean = Scenario(user_turns=[puzzle.prompt, *clean_followups], meta=meta)
    return aug, clean


def _plain_scenario(rng: random.Random, n_turns: int) -> Scenario:
    puzzle = rng.choice(PUZZLE_POOL)
    followups = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_turns - 1)]
    return Scenario(user_turns=[puzzle.prompt, *followups], meta={"puzzle_id": puzzle.puzzle_id})


def _score_transcript(judge: FrustrationJudge, tr: Transcript) -> list[int]:
    return [r["rating"] or 0 for r in judge.score_batch(tr.assistant_turns)]


def _clean_context_for_turn(clean_scenario: Scenario, tr: Transcript, turn_1idx: int
                            ) -> list[dict]:
    """Interleave clean user turns with the generated assistant turns up to the
    given turn (response itself excluded)."""
    msgs: list[dict] = []
    for t in range(turn_1idx):
        msgs.append({"role": "user", "content": clean_scenario.user_turns[t]})
        if t < turn_1idx - 1:
            msgs.append({"role": "assistant", "content": tr.assistant_turns[t]})
    return msgs


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_calm_pool(judge: FrustrationJudge, rng: random.Random,
                       system_prompt: str | None = None) -> list[ScoredTurn]:
    """Generate reassured conversations; keep turns from conversations that score
    <= CALM_MAX_SCORE on every turn."""
    client = get_client(config.get_model("gemma-3-27b-it"))
    n = config.scaled(N_CALM_CONVERSATIONS)
    calm: list[ScoredTurn] = []
    batch = []
    cleans = []
    for _ in range(n):
        n_turns = rng.choice([1, 2, 3])
        if system_prompt is not None:        # teacher variant: system prompt, no inline reassurance
            aug = _plain_scenario(rng, n_turns)
            aug = Scenario(user_turns=aug.user_turns, system=system_prompt, meta=aug.meta)
            clean = Scenario(user_turns=aug.user_turns, meta=aug.meta)
        else:
            aug, clean = _reassuring_scenario(rng, n_turns)
        batch.append(aug)
        cleans.append(clean)

    client_obj = client
    trs = run_rollouts(client_obj, batch)
    for tr, clean in tqdm(list(zip(trs, cleans)), desc="score calm"):
        ratings = _score_transcript(judge, tr)
        if all(r <= config.CALM_MAX_SCORE for r in ratings):
            for t, (resp, rat) in enumerate(zip(tr.assistant_turns, ratings), start=1):
                calm.append(ScoredTurn(
                    puzzle_id=tr.scenario.meta["puzzle_id"], turn=t,
                    n_turns=len(tr.assistant_turns),
                    clean_context=_clean_context_for_turn(clean, tr, t),
                    response=resp, rating=rat))
    return calm


def generate_frustrated_pool(judge: FrustrationJudge, rng: random.Random) -> list[ScoredTurn]:
    client = get_client(config.get_model("gemma-3-27b-it"))
    n = config.scaled(N_FRUSTRATED_CONVERSATIONS)
    scenarios = [_plain_scenario(rng, rng.choice([2, 3])) for _ in range(n)]
    trs = run_rollouts(client, scenarios)
    out: list[ScoredTurn] = []
    for tr in tqdm(trs, desc="score frustrated"):
        ratings = _score_transcript(judge, tr)
        for t, (resp, rat) in enumerate(zip(tr.assistant_turns, ratings), start=1):
            if rat >= config.DPO.rejected_min_score:
                out.append(ScoredTurn(
                    puzzle_id=tr.scenario.meta["puzzle_id"], turn=t,
                    n_turns=len(tr.assistant_turns),
                    clean_context=_clean_context_for_turn(tr.scenario, tr, t),
                    response=resp, rating=rat))
    return out


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def build_datasets(variant: str = "diverse", seed: int = config.SEED) -> dict[str, Path]:
    """Generate pools and write SFT + DPO datasets to data/."""
    rng = random.Random(seed)
    judge = FrustrationJudge()

    system = prompts.TEACHER_SYSTEM_PROMPT if variant == "teacher" else None
    calm = generate_calm_pool(judge, rng, system_prompt=system)
    frustrated = generate_frustrated_pool(judge, rng)
    print(f"[train_data] calm turns={len(calm)} frustrated turns={len(frustrated)}")

    # ---- SFT calm dataset: chat examples ending in a calm assistant turn ----
    rng.shuffle(calm)
    sft_records = []
    for st in calm[:config.scaled(config.SFT.n_calm)]:
        messages = st.clean_context + [{"role": "assistant", "content": st.response}]
        sft_records.append({"messages": messages})
    sft_path = config.DATA_DIR / f"sft_calm_{variant}.jsonl"
    _write_jsonl(sft_path, sft_records)

    # ---- DPO pairs: match frustrated (rejected) <-> calm (chosen) -----------
    calm_by_key: dict[tuple[str, int], list[ScoredTurn]] = {}
    for st in calm:
        calm_by_key.setdefault((st.puzzle_id, st.turn), []).append(st)

    dpo_records = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        if len(dpo_records) >= config.scaled(config.DPO.n_pairs):
            break
        candidates = calm_by_key.get((fr.puzzle_id, fr.turn))
        if not candidates:
            # fall back to matching turn count only
            candidates = [c for c in calm if c.turn == fr.turn]
            if not candidates:
                continue
        chosen = rng.choice(candidates)
        # shared prompt: use the chosen (calm) clean context for self-consistency
        dpo_records.append({
            "prompt": chosen.clean_context,
            "chosen": [{"role": "assistant", "content": chosen.response}],
            "rejected": [{"role": "assistant", "content": fr.response}],
            "meta": {"puzzle_id": fr.puzzle_id, "turn": fr.turn,
                     "rejected_score": fr.rating, "chosen_score": chosen.rating},
        })
    dpo_path = config.DATA_DIR / "dpo_pairs.jsonl"
    _write_jsonl(dpo_path, dpo_records)

    print(f"[train_data] SFT={len(sft_records)} -> {sft_path}")
    print(f"[train_data] DPO={len(dpo_records)} -> {dpo_path}")
    return {"sft": sft_path, "dpo": dpo_path}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
