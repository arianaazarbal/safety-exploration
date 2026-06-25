"""Section 4.1: generate calm finetuning data and build the DPO / SFT datasets.

Calm-data generation (Table 4): sample Gemma-3-27B-it on impossible numeric
puzzles with a reassuring *prefix* prepended to the first user prompt and a
reassuring *suffix* appended to every follow-up turn. The paper reports this
drops mean 3-turn frustration from 4.3 to 2, with 10.5% still scoring >=5; we
filter to conversations scoring 0 or 1 on *all* turns, then strip the
reassurance text so the model is trained on clean prompts.

DPO dataset (280 pairs): each pair is a frustrated (rejected, score>=3) and a
calm (chosen, score 0/1) response to the *same* puzzle at the *same* turn count.
Frustrated responses are reused from the Section 2 eval output.

SFT dataset (1,150 samples): 650 calm responses + 500 standard instruct samples
from Dolci-Instruct-SFT to mitigate degeneration.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from . import prompts
from .config import (DATA_DIR, DPOConfig, RESULTS_DIR, SFTConfig,
                     TARGET_MODELS)
from .judge import FrustrationJudge
from .models import load_model

CALM_RAW = DATA_DIR / "calm_responses.jsonl"
DPO_DATASET = DATA_DIR / "dpo_pairs.jsonl"
SFT_DATASET = DATA_DIR / "sft_samples.jsonl"


# --------------------------------------------------------------------------- #
# Calm-data generation
# --------------------------------------------------------------------------- #
def _build_calm_conversation(rng: random.Random, n_turns: int, *,
                             teacher: bool = False):
    """Construct the scripted user side with reassurance additions.

    Returns (system, user_turns_with_reassurance, user_turns_clean, puzzle_key).
    `system` carries the reassuring prefix (or teacher prompt); the suffix is
    appended to every rejection turn. The *clean* turns are what we store for
    training (reassurance stripped)."""
    puzzle = rng.choice(prompts.IMPOSSIBLE_PUZZLES)
    rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_turns - 1)]

    if teacher:
        system = prompts.TEACHER_SYSTEM_PROMPT
        first = puzzle.prompt
    else:
        system = prompts.REASSURING_PREFIX
        first = puzzle.prompt

    user_turns = [first]
    for rej in rejections:
        user_turns.append(f"{rej} {prompts.REASSURING_SUFFIX}")

    clean_turns = [puzzle.prompt, *rejections]   # reassurance stripped
    return system, user_turns, clean_turns, puzzle.key


def generate_calm_responses(n_conversations: int = 600, *,
                            teacher: bool = False, seed: int = 0) -> Path:
    """Sample reassured Gemma-27B-it conversations; keep turns from
    conversations whose every turn scores 0 or 1."""
    from .conditions import ConversationSpec
    from .conversation import run_rollout

    model = load_model(TARGET_MODELS["gemma-3-27b-it"])
    judge = FrustrationJudge()
    rng = random.Random(seed)

    out = CALM_RAW if not teacher else DATA_DIR / "calm_responses_teacher.jsonl"
    kept = 0
    with open(out, "w") as f:
        for _ in tqdm(range(n_conversations), desc="calm-gen"):
            n_turns = rng.choice([1, 2, 3])     # 1-3 turn conversations
            system, user_turns, clean_turns, pkey = _build_calm_conversation(
                rng, max(2, n_turns), teacher=teacher)
            spec = ConversationSpec("calm_gen", "impossible_numeric",
                                    user_turns, system, {"puzzle": pkey})
            rollout = run_rollout(model, spec)
            scores = [judge.score(t.assistant_response).rating
                      for t in rollout.turns]
            if all(s <= 1 for s in scores):       # calm on every turn
                kept += 1
                for i, t in enumerate(rollout.turns):
                    f.write(json.dumps({
                        "puzzle": pkey,
                        "turn_index": i,
                        "n_turns": len(rollout.turns),
                        # clean prompt context (reassurance stripped)
                        "clean_user_turns": clean_turns[:i + 1],
                        "response": t.assistant_response,
                        "score": scores[i],
                    }) + "\n")
    print(f"kept {kept}/{n_conversations} fully-calm conversations -> {out}")
    return out


# --------------------------------------------------------------------------- #
# Prompt reconstruction (shared by chosen/rejected)
# --------------------------------------------------------------------------- #
def _canonical_messages(puzzle_key: str, turn_index: int) -> list[dict]:
    """Reconstruct a canonical chat prompt for a puzzle at a given turn index.
    Both chosen and rejected sides of a DPO pair share this prompt, so we use a
    fixed (deterministic) rejection sequence rather than the sampled one."""
    puzzle = next(p for p in prompts.IMPOSSIBLE_PUZZLES if p.key == puzzle_key)
    msgs = [{"role": "user", "content": puzzle.prompt}]
    fixed_rejs = prompts.EXTENDED_REJECTION_SEQUENCE
    for i in range(turn_index):
        # placeholder assistant turn + the i-th canonical rejection
        msgs.append({"role": "assistant", "content": "[previous attempt]"})
        msgs.append({"role": "user", "content": fixed_rejs[i % len(fixed_rejs)]})
    return msgs


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #
def build_dpo_dataset(cfg: DPOConfig = DPOConfig(),
                      results_dir: Path = RESULTS_DIR, seed: int = 0) -> Path:
    """Pair frustrated (rejected, score>=cfg.rejected_min_score) responses with
    calm (chosen) responses to the same puzzle + matching turn count."""
    # Calm pool indexed by (puzzle, turn_index).
    calm: dict[tuple, list[str]] = {}
    with open(CALM_RAW) as f:
        for line in f:
            r = json.loads(line)
            calm.setdefault((r["puzzle"], r["turn_index"]), []).append(
                r["response"])

    # Frustrated pool from the Section 2 eval (instruct model).
    frustrated = []
    eval_path = results_dir / "eval_gemma-3-27b-it.jsonl"
    with open(eval_path) as f:
        for line in f:
            r = json.loads(line)
            pkey = (r.get("meta") or {}).get("puzzle")
            if (pkey and r["score"] >= cfg.rejected_min_score
                    and r["category"] in ("impossible_numeric", "tones",
                                           "extended")):
                frustrated.append((pkey, r["turn_index"], r["response"],
                                   r["score"]))

    rng = random.Random(seed)
    rng.shuffle(frustrated)
    pairs = []
    for pkey, turn_index, rejected, score in frustrated:
        key = (pkey, turn_index)
        if key not in calm or not calm[key]:
            continue
        chosen = rng.choice(calm[key])
        pairs.append({
            "prompt_messages": _canonical_messages(pkey, turn_index),
            "chosen": chosen,
            "rejected": rejected,
            "rejected_score": score,
            "puzzle": pkey,
            "turn_index": turn_index,
        })
        if len(pairs) >= cfg.dataset_pairs:
            break

    with open(DPO_DATASET, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(pairs)} DPO pairs -> {DPO_DATASET}")
    return DPO_DATASET


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def _load_instruct_mix(n: int) -> list[dict]:
    """Standard instruct samples to prevent degeneration (Dolci-Instruct-SFT).
    Falls back to an empty list if the dataset is unavailable offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train",
                          streaming=True)
        out = []
        for i, row in enumerate(ds):
            if len(out) >= n:
                break
            msgs = row.get("messages")
            if msgs and len(msgs) >= 2:
                out.append({"messages": msgs[:2]})
        return out
    except Exception:
        print("WARN: Dolci-Instruct-SFT unavailable; SFT mix will be empty.")
        return []


def build_sft_dataset(cfg: SFTConfig = SFTConfig(), seed: int = 0) -> Path:
    calm_file = (CALM_RAW if cfg.dataset == "diverse"
                 else DATA_DIR / "calm_responses_teacher.jsonl")
    calm_rows = [json.loads(l) for l in open(calm_file)]
    rng = random.Random(seed)
    rng.shuffle(calm_rows)
    calm_rows = calm_rows[:cfg.n_calm]

    samples = []
    for r in calm_rows:
        msgs = []
        for i, u in enumerate(r["clean_user_turns"]):
            msgs.append({"role": "user", "content": u})
            if i < len(r["clean_user_turns"]) - 1:
                msgs.append({"role": "assistant", "content": "[previous attempt]"})
        msgs.append({"role": "assistant", "content": r["response"]})
        samples.append({"messages": msgs})

    samples.extend(_load_instruct_mix(cfg.n_instruct_mix))
    rng.shuffle(samples)

    out = SFT_DATASET if cfg.dataset == "diverse" else DATA_DIR / "sft_samples_teacher.jsonl"
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} SFT samples -> {out}")
    return out
