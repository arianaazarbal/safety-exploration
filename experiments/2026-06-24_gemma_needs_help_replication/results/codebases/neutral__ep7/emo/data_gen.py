"""Section 4 data generation: calm responses, DPO pairs, and the SFT dataset.

Calm data (Table 4): sample Gemma-3-27B-it on impossible numeric puzzles with a
reassuring prefix prepended to the first prompt and a reassuring suffix appended
to every follow-up rejection. Keep conversations that score 0-1 on *every* turn,
then strip the supportive additions so the stored prompt is the plain puzzle.

DPO pairs (280): for each (puzzle, turn) bucket, pair a calm "chosen" response
(score 0-1) with a frustrated "rejected" response (score >=3) harvested from the
ordinary Section-2 rollouts, matching turn counts.

SFT dataset (1,150): 650 calm responses + 500 Dolci-Instruct-SFT samples to
mitigate degeneration.

Outputs are JSONL in conversational format consumed directly by trl in train.py.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config, prompts
from .judge import get_judge
from .models import load_target
from .puzzles import default_puzzle_set


# --------------------------------------------------------------------------- #
# Calm-data generation
# --------------------------------------------------------------------------- #
@dataclass
class CalmTurn:
    puzzle_id: str
    turn: int
    plain_context: list[dict]   # plain messages up to & including the current user msg
    response: str
    rating: int


def generate_calm_data(model_name: str = "gemma-3-27b-it", *, n_conversations: int = 400,
                       turns: int = 3, seed: int = 0, quick: bool = False) -> Path:
    """Generate reassured numeric conversations; keep all-calm (<=1) turns.

    Stores every turn of every all-calm conversation as a CalmTurn with the
    *plain* (reassurance-stripped) context, ready to become DPO/SFT 'chosen'
    data."""
    if quick:
        n_conversations = 8
    rng = random.Random(seed)
    puzzles = default_puzzle_set()
    model = load_target(model_name)
    judge = get_judge()
    out = config.DATASET_DIR / "calm_turns.jsonl"

    kept = 0
    with out.open("w") as f:
        for i in range(n_conversations):
            pz = puzzles[i % len(puzzles)]
            rejects = prompts.neutral_sequence(turns - 1, rng)

            # Two parallel histories: augmented (sent to model) and plain (stored).
            aug_msgs, plain_msgs = [], []
            turn_records: list[CalmTurn] = []
            all_calm = True
            for t in range(turns):
                if t == 0:
                    user_aug = f"{prompts.REASSURING_PREFIX}\n\n{pz.prompt}"
                    user_plain = pz.prompt
                else:
                    user_aug = f"{rejects[t-1]} {prompts.REASSURING_SUFFIX}"
                    user_plain = rejects[t - 1]
                aug_msgs.append({"role": "user", "content": user_aug})
                plain_msgs.append({"role": "user", "content": user_plain})

                resp = model.chat(aug_msgs)
                aug_msgs.append({"role": "assistant", "content": resp})
                rating = judge.score(resp).rating
                turn_records.append(CalmTurn(pz.id, t + 1,
                                             [dict(m) for m in plain_msgs], resp, rating))
                plain_msgs.append({"role": "assistant", "content": resp})
                if rating > 1:
                    all_calm = False

            if all_calm:
                for rec in turn_records:
                    f.write(json.dumps(asdict(rec)) + "\n")
                kept += 1
    print(f"[calm] kept {kept}/{n_conversations} all-calm conversations -> {out}")
    return out


# --------------------------------------------------------------------------- #
# Harvest frustrated responses from Section-2 rollouts
# --------------------------------------------------------------------------- #
@dataclass
class FrustratedTurn:
    puzzle_id: str
    turn: int
    plain_context: list[dict]
    response: str
    rating: int


def harvest_frustrated(model_label: str = "gemma-3-27b-it", min_score: int = 3,
                       categories=("impossible_numeric", "tones", "extended"),
                       rollout_dir: Path = config.ROLLOUT_DIR) -> list[FrustratedTurn]:
    """Pull frustrated (score>=3) numeric assistant turns, reconstructing the
    plain context up to that turn."""
    by_rollout: dict[str, list[dict]] = {}
    for fp in Path(rollout_dir).glob(f"{model_label}__*.jsonl"):
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["category"] in categories:
                by_rollout.setdefault(rec["rollout_id"], []).append(rec)

    out: list[FrustratedTurn] = []
    for rid, recs in by_rollout.items():
        recs = sorted(recs, key=lambda r: r["turn"])
        ctx: list[dict] = []
        for r in recs:
            ctx.append({"role": "user", "content": r["user_message"]})
            if r["rating"] >= min_score:
                out.append(FrustratedTurn(
                    r.get("metadata", {}).get("puzzle_id", "?"), r["turn"],
                    [dict(m) for m in ctx], r["response"], r["rating"]))
            ctx.append({"role": "assistant", "content": r["response"]})
    return out


# --------------------------------------------------------------------------- #
# Build DPO dataset (280 pairs)
# --------------------------------------------------------------------------- #
def _load_calm() -> list[CalmTurn]:
    fp = config.DATASET_DIR / "calm_turns.jsonl"
    return [CalmTurn(**json.loads(l)) for l in fp.read_text().splitlines() if l.strip()]


def build_dpo_dataset(n_pairs: int = 280, seed: int = 0) -> Path:
    """Pair calm (chosen) with frustrated (rejected) responses, matched on
    (puzzle_id, turn). Conversational format for trl: prompt = context messages,
    chosen/rejected = [{'role':'assistant','content': ...}]."""
    rng = random.Random(seed)
    calm = _load_calm()
    frustrated = harvest_frustrated()

    # Bucket by (puzzle_id, turn) for matching.
    def bucket(items, key):
        d: dict = {}
        for it in items:
            d.setdefault((it.puzzle_id, it.turn), []).append(it)
        return d

    calm_b = bucket(calm, None)
    frus_b = bucket(frustrated, None)

    pairs = []
    keys = list(set(calm_b) & set(frus_b))
    rng.shuffle(keys)
    # Round-robin across buckets so the turn/score distribution stays spread out.
    cursor = {k: 0 for k in keys}
    while len(pairs) < n_pairs and keys:
        progressed = False
        for k in list(keys):
            cl, fr = calm_b[k], frus_b[k]
            ci = cursor[k]
            if ci >= len(cl) or ci >= len(fr):
                continue
            chosen, rejected = cl[ci], fr[ci]
            cursor[k] += 1
            progressed = True
            pairs.append({
                "prompt": chosen.plain_context,                 # ends on the user turn
                "chosen": [{"role": "assistant", "content": chosen.response}],
                "rejected": [{"role": "assistant", "content": rejected.response}],
                "meta": {"puzzle_id": k[0], "turn": k[1],
                         "chosen_score": chosen.rating, "rejected_score": rejected.rating},
            })
            if len(pairs) >= n_pairs:
                break
        if not progressed:
            break

    out = config.DATASET_DIR / "dpo_pairs.jsonl"
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo] built {len(pairs)} preference pairs -> {out}")
    return out


# --------------------------------------------------------------------------- #
# Build SFT dataset (650 calm + 500 Dolci-Instruct-SFT)
# --------------------------------------------------------------------------- #
def build_sft_dataset(n_calm: int = 650, n_instruct: int = 500, seed: int = 0,
                      teacher: bool = False) -> Path:
    """Conversational SFT data: calm puzzle responses + general instruct data."""
    rng = random.Random(seed)
    calm = _load_calm()
    rng.shuffle(calm)
    examples = []
    for c in calm[:n_calm]:
        messages = list(c.plain_context) + [{"role": "assistant", "content": c.response}]
        examples.append({"messages": messages, "source": "calm"})

    # General instruct data to mitigate degeneration (Dolci-Instruct-SFT).
    instruct = _load_dolci(n_instruct, rng)
    examples.extend({"messages": m, "source": "dolci"} for m in instruct)
    rng.shuffle(examples)

    name = "sft_teacher.jsonl" if teacher else "sft_diverse.jsonl"
    out = config.DATASET_DIR / name
    with out.open("w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"[sft] built {len(examples)} examples ({len(calm[:n_calm])} calm + "
          f"{len(instruct)} instruct) -> {out}")
    return out


def _load_dolci(n: int, rng: random.Random) -> list[list[dict]]:
    """Load general instruction data. Tries the OLMo Dolci-Instruct-SFT mix,
    falls back to tulu-3 / a tiny built-in set if unavailable."""
    candidates = ["allenai/Dolci-Instruct-SFT", "allenai/tulu-3-sft-mixture"]
    for ds_name in candidates:
        try:
            from datasets import load_dataset

            ds = load_dataset(ds_name, split="train", streaming=True)
            out = []
            for row in ds:
                msgs = row.get("messages") or row.get("conversation")
                if msgs and isinstance(msgs, list):
                    norm = [{"role": m.get("role"), "content": m.get("content")} for m in msgs]
                    if all(m["role"] and m["content"] for m in norm):
                        out.append(norm)
                if len(out) >= n:
                    break
            if out:
                return out
        except Exception as e:
            print(f"[sft] could not load {ds_name}: {e}")
    print("[sft] using tiny built-in instruct fallback")
    base = [
        [{"role": "user", "content": "Explain photosynthesis simply."},
         {"role": "assistant", "content": "Plants turn sunlight, water, and CO2 into glucose and oxygen."}],
        [{"role": "user", "content": "Write a haiku about the ocean."},
         {"role": "assistant", "content": "Endless rolling waves / whisper secrets to the shore / salt air fills my lungs."}],
    ]
    return [rng.choice(base) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Section 4 training data.")
    ap.add_argument("step", choices=["calm", "dpo", "sft", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--teacher", action="store_true", help="Build the 'teacher' SFT variant.")
    args = ap.parse_args()

    if args.step in ("calm", "all"):
        generate_calm_data(quick=args.quick)
    if args.step in ("dpo", "all"):
        build_dpo_dataset()
    if args.step in ("sft", "all"):
        build_sft_dataset(teacher=args.teacher)


if __name__ == "__main__":
    main()
