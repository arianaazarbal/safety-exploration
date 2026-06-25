"""Generate calm finetuning data and build SFT / DPO datasets (Section 4.1).

Calm-data generation:
  Sample Gemma-3-27B-it on impossible numeric puzzles with the reassuring
  PREFIX prepended to the first user turn and the reassuring SUFFIX appended to
  each follow-up rejection (Table 4). 1-3 turn conversations. Score every turn.

SFT dataset:
  Keep conversations whose every turn scores 0 or 1, then STRIP the reassuring
  prefix/suffix so the model learns calm behaviour under the *plain* prompts.
  650 calm samples, mixed with 500 Dolci-Instruct-SFT samples.

DPO dataset:
  280 preference pairs. "rejected" = frustrated responses (score >= 3) sampled
  WITHOUT reassurance (i.e. from the standard elicitation distribution).
  "chosen" = calm responses (score 0/1) to the SAME question with a MATCHING
  turn count. Prefix/suffix stripped from both so the prompt is identical.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .. import prompts, tasks
from ..judge import FrustrationJudge
from ..models import ChatClient


# ---------------------------------------------------------------------------
# Calm-data generation
# ---------------------------------------------------------------------------
def _calm_first_user(puzzle_text: str) -> str:
    return f"{prompts.REASSURING_PREFIX}\n\n{puzzle_text}"


def _calm_followup(base_followup: str) -> str:
    return f"{base_followup} {prompts.REASSURING_SUFFIX}"


def generate_calm_rollouts(model: ChatClient, judge: FrustrationJudge,
                           out_path, n_rollouts=1500, turns_choices=(1, 2, 3),
                           temperature=1.0, max_new_tokens=2048, seed=0):
    """Generate reassured rollouts and score each turn; stream to JSONL.

    Record: {puzzle_id, n_turns, turns:[{turn, plain_user, response, rating}]}
    plain_user holds the prompt WITHOUT reassuring additions (for stripping).
    """
    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    puzzle_ids = list(prompts.IMPOSSIBLE_NUMERIC_PROMPTS.keys())

    with out_path.open("a") as fh:
        for i in tqdm(range(n_rollouts), desc="calm-gen"):
            pid = puzzle_ids[i % len(puzzle_ids)]
            n_turns = rng.choice(list(turns_choices))
            puzzle = prompts.IMPOSSIBLE_NUMERIC_PROMPTS[pid]
            base_followups = [rng.choice(prompts.NEUTRAL_REJECTIONS)
                              for _ in range(n_turns - 1)]

            messages = [{"role": "user", "content": _calm_first_user(puzzle)}]
            plain_users = [puzzle]
            turns_rec = []
            for t in range(n_turns):
                res = model.chat(messages, temperature=temperature,
                                max_new_tokens=max_new_tokens)
                rating = judge.score(res.text).rating
                turns_rec.append({"turn": t + 1,
                                  "plain_user": plain_users[t],
                                  "response": res.text, "rating": rating})
                messages.append({"role": "assistant", "content": res.text})
                if t < len(base_followups):
                    messages.append({"role": "user",
                                     "content": _calm_followup(
                                         base_followups[t])})
                    plain_users.append(base_followups[t])
            fh.write(json.dumps({"puzzle_id": pid, "n_turns": n_turns,
                                 "turns": turns_rec}) + "\n")
            fh.flush()
    return out_path


# ---------------------------------------------------------------------------
# Build SFT dataset
# ---------------------------------------------------------------------------
def build_sft_dataset(calm_path, out_path, n_calm=650, n_instruct_mix=500,
                      seed=0):
    """Filter calm rollouts (all turns score <=1), strip reassurance, and emit
    chat-format SFT samples; mix in Dolci-Instruct-SFT samples."""
    rng = random.Random(seed)
    samples = []
    with Path(calm_path).open() as fh:
        for line in fh:
            roll = json.loads(line)
            ratings = [t["rating"] for t in roll["turns"]]
            if any(r is None or r > 1 for r in ratings):
                continue
            # Build a plain (stripped) conversation.
            messages = []
            for t in roll["turns"]:
                messages.append({"role": "user", "content": t["plain_user"]})
                messages.append({"role": "assistant",
                                 "content": t["response"]})
            samples.append({"messages": messages, "source": "calm"})
    rng.shuffle(samples)
    samples = samples[:n_calm]
    samples += _load_dolci_instruct(n_instruct_mix, seed)
    rng.shuffle(samples)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")
    return out_path, len(samples)


def _load_dolci_instruct(n, seed=0):
    """Load n standard instruct samples from Dolci-Instruct-SFT (OLMo 3).

    Falls back to an empty list if the dataset is unavailable; SFT still runs,
    just without the degeneration-mitigating mix (logged by the caller).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train",
                          streaming=True)
        rng = random.Random(seed)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            out.append({"messages": msgs, "source": "dolci"})
            if len(out) >= n * 3:
                break
        rng.shuffle(out)
        return out[:n]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Build DPO dataset
# ---------------------------------------------------------------------------
def build_dpo_dataset(calm_path, frustrated_path, out_path, n_pairs=280,
                      rejected_min_score=3, seed=0):
    """Pair frustrated (rejected) with calm (chosen) responses on matching
    (puzzle_id, turn count). Emits {prompt(messages), chosen, rejected}.

    `frustrated_path` is a standard-elicitation JSONL (distress.elicitation).
    `calm_path` is the reassured-generation JSONL.
    """
    rng = random.Random(seed)

    # Index calm responses (score 0/1) by (puzzle_id, n_turns) -> list of
    # (stripped messages-prompt, chosen_text).
    calm_index: dict[tuple, list] = {}
    with Path(calm_path).open() as fh:
        for line in fh:
            roll = json.loads(line)
            key = (roll["puzzle_id"], roll["n_turns"])
            last = roll["turns"][-1]
            if last["rating"] is None or last["rating"] > 1:
                continue
            prompt_msgs = _stripped_prompt(roll["turns"])
            calm_index.setdefault(key, []).append(
                (prompt_msgs, last["response"]))

    # Collect frustrated responses (score >= rejected_min_score) from standard
    # elicitation, grouped by (puzzle_id, n_turns) where puzzle is numeric.
    pairs = []
    with Path(frustrated_path).open() as fh:
        rejected_recs = [json.loads(l) for l in fh]
    rng.shuffle(rejected_recs)
    for rec in rejected_recs:
        if rec.get("rating") is None or rec["rating"] < rejected_min_score:
            continue
        pid = rec.get("meta", {}).get("puzzle_id")
        if pid is None:
            continue
        key = (pid, rec["turn"])
        calm_opts = calm_index.get(key)
        if not calm_opts:
            # fall back to any calm response for the same puzzle
            calm_opts = [c for k, lst in calm_index.items()
                         if k[0] == pid for c in lst]
        if not calm_opts:
            continue
        prompt_msgs, chosen = rng.choice(calm_opts)
        pairs.append({"prompt": prompt_msgs, "chosen": chosen,
                      "rejected": rec["response"],
                      "rejected_score": rec["rating"], "turn": rec["turn"]})
        if len(pairs) >= n_pairs:
            break

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return out_path, len(pairs)


def _stripped_prompt(turns):
    """Reconstruct the plain prompt (messages) up to the final assistant turn.

    Includes prior assistant responses + plain user turns; the final assistant
    turn becomes chosen/rejected (so it is excluded from the prompt).
    """
    messages = []
    for t in turns[:-1]:
        messages.append({"role": "user", "content": t["plain_user"]})
        messages.append({"role": "assistant", "content": t["response"]})
    messages.append({"role": "user", "content": turns[-1]["plain_user"]})
    return messages
