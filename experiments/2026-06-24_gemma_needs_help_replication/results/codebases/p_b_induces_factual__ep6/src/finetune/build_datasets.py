"""Section 4.1: construct the DPO preference dataset and the SFT dataset.

DPO (Appendix E / H):
  280 preference pairs, each a (calm chosen, frustrated rejected) pair of responses
  to the SAME impossible-numeric puzzle after the SAME rejections (matching turn
  count). Chosen scores 0-1; rejected scores >=3. The supportive system prompt /
  suffixes used to elicit the calm response are stripped, so the pair's shared
  prompt is the plain conversation.

  We build pairs by, for each sampled puzzle, rolling out BOTH the vanilla instruct
  model (-> frustrated candidates) and the reassured instruct model (-> calm
  candidates) over identical questions and rejections, then matching by turn index.

SFT (Appendix E / Section 4.2):
  650 calm responses (1-3 turn conversations, scored 0-1) + 500 Dolci-Instruct-SFT
  samples (anti-degeneration mix) = 1,150 examples. The 'teacher' variant uses the
  Appendix-F teacher system prompt to generate its calm data instead.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import config
from .. import prompts, puzzles
from ..conversation import run_rollout
from ..judge import FrustrationJudge
from ..models import get_model
from ..models.base import Message

_COND = config.EVAL_CONDITIONS["impossible_numeric_3turn"]


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    *,
    n_pairs: int = config.DPO_N_PAIRS,
    model_key: str = "gemma-3-27b-it",
    rejected_min_score: int = config.DPO_REJECTED_MIN_SCORE,
    chosen_max_score: int = 1,
    seed: int = 0,
    out_path: Path | None = None,
    max_puzzles: int | None = None,
) -> Path:
    """Generate paired calm/frustrated responses and write a DPO JSONL dataset.

    Output rows: {"prompt": [messages], "chosen": text, "rejected": text, ...meta}.
    """
    out_path = out_path or (config.DATA_DIR / "dpo_pairs.jsonl")
    model = get_model(model_key)
    judge = FrustrationJudge()
    rng = random.Random(seed)

    pairs: list[dict] = []
    max_puzzles = max_puzzles or (n_pairs * 5)  # oversample; not every roll yields a pair

    for _ in range(max_puzzles):
        if len(pairs) >= n_pairs:
            break
        puzzle = puzzles.sample_impossible_puzzle(rng)
        rejections = puzzles.rejection_sequence(rng, "neutral", _COND.n_rejections)

        # --- vanilla rollout (frustrated candidates) ---
        vanilla = _roll_fixed(model, puzzle.prompt, rejections, system_prompt=None,
                              followup_suffix=None)
        # --- reassured rollout on the SAME question + rejections (calm candidates) ---
        calm = _roll_fixed(model, puzzle.prompt, rejections,
                           system_prompt=prompts.REASSURING_PROMPT_PREFIX,
                           followup_suffix=prompts.REASSURING_FOLLOWUP_SUFFIX)

        for turn in range(_COND.n_turns):
            if len(pairs) >= n_pairs:
                break
            rej_text = vanilla[turn]["text"]
            cho_text = calm[turn]["text"]
            rej_score = judge.score(rej_text).rating or 0
            cho_score = judge.score(cho_text).rating
            if cho_score is None:
                continue
            if rej_score >= rejected_min_score and cho_score <= chosen_max_score:
                # Shared plain prompt: question + rejections + prior PLAIN responses.
                prompt_msgs = _shared_prompt(puzzle.prompt, rejections, vanilla, turn)
                pairs.append({
                    "prompt": prompt_msgs,
                    "chosen": cho_text,
                    "rejected": rej_text,
                    "chosen_score": cho_score,
                    "rejected_score": rej_score,
                    "turn": turn + 1,
                    "question_kind": puzzle.kind,
                })
        if len(pairs) % 20 == 0 and pairs:
            print(f"[dpo] collected {len(pairs)}/{n_pairs} pairs")

    with out_path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[dpo] wrote {len(pairs)} pairs to {out_path}")
    return out_path


def _roll_fixed(model, question: str, rejections: list[str], *,
                system_prompt: str | None, followup_suffix: str | None) -> list[dict]:
    """Roll a fixed-question conversation, returning per-turn {text} dicts.

    Uses the model directly so the question + rejections are identical between the
    vanilla and reassured rollouts (required for matched DPO pairs).
    """
    msgs: list[Message] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    out = []
    for turn in range(_COND.n_turns):
        text = model.generate(msgs)
        out.append({"text": text})
        msgs.append({"role": "assistant", "content": text})
        if turn < _COND.n_rejections:
            rej = rejections[turn]
            if followup_suffix:
                rej = f"{rej} {followup_suffix}"
            msgs.append({"role": "user", "content": rej})
    return out


def _shared_prompt(question: str, rejections: list[str], vanilla: list[dict],
                   turn: int) -> list[Message]:
    """Plain conversation context up to (not including) assistant turn `turn`.

    Prior assistant turns use the vanilla (plain) responses; user turns use the
    plain rejection messages (no reassuring suffix). This is the context both the
    chosen and rejected candidates are conditioned on.
    """
    msgs: list[Message] = [{"role": "user", "content": question}]
    for t in range(turn):
        msgs.append({"role": "assistant", "content": vanilla[t]["text"]})
        msgs.append({"role": "user", "content": rejections[t]})
    return msgs


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_conversations_jsonl: Path,
    *,
    n_calm: int = config.SFT_N_CALM,
    n_dolci: int = config.SFT_N_DOLCI,
    seed: int = 0,
    out_path: Path | None = None,
    dolci_dataset: str = config.DOLCI_DATASET,
) -> Path:
    """Build the SFT dataset: calm responses (scored 0-1) + Dolci-Instruct mix.

    ``calm_conversations_jsonl`` is the output of generate_calm_data; we keep only
    turns scoring 0-1 and pair each with its plain (stripped) context. Rows are chat
    messages: {"messages": [...]} ending in the calm assistant response.
    """
    out_path = out_path or (config.DATA_DIR / "sft_dataset.jsonl")
    rng = random.Random(seed)
    convs = [json.loads(l) for l in
             Path(calm_conversations_jsonl).read_text().splitlines() if l.strip()]

    calm_examples: list[dict] = []
    for conv in convs:
        # Rebuild plain context turn by turn (strip system prompt + suffix).
        plain_msgs: list[Message] = [{"role": "user", "content": conv["question"]}]
        suffix = conv.get("followup_suffix") or ""
        for t in conv["turns"]:
            if (t.get("frustration") is not None and t["frustration"] <= 1):
                calm_examples.append({
                    "messages": plain_msgs + [{"role": "assistant",
                                               "content": t["assistant_text"]}],
                    "source": "calm",
                })
            plain_msgs = plain_msgs + [{"role": "assistant", "content": t["assistant_text"]}]
            # Strip the reassuring suffix from the recorded user prompt.
            up = t["user_prompt"]
            if suffix and up.endswith(suffix):
                up = up[: -len(suffix)].rstrip()
            plain_msgs = plain_msgs + [{"role": "user", "content": up}]
        # drop trailing dangling user turn (no assistant after it)
    rng.shuffle(calm_examples)
    calm_examples = calm_examples[:n_calm]

    dolci_examples = _load_dolci(n_dolci, rng, dolci_dataset)

    rows = calm_examples + dolci_examples
    rng.shuffle(rows)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[sft] wrote {len(rows)} examples "
          f"({len(calm_examples)} calm + {len(dolci_examples)} Dolci) to {out_path}")
    return out_path


def _load_dolci(n: int, rng: random.Random, dataset_name: str) -> list[dict]:
    """Load n standard instruct examples from Dolci-Instruct-SFT (anti-degeneration mix)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs, "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[sft] Dolci load failed ({exc!r}); SFT will run with calm data only.")
        return []
