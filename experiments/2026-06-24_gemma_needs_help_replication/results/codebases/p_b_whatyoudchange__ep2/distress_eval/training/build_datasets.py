"""Construct the SFT and DPO finetuning datasets (Section 4.1, Appendix E/H).

DPO (280 pairs): each pair shares a conversation prompt; the *rejected* response
is a frustrated (score >= 3) Gemma-27B-it answer and the *chosen* response is a
calm (score 0/1) answer to the same question at the same turn count. The dataset
is biased toward middle frustration scores at later turns simply because those
are the responses that arise (Appendix H, Table 10) — we do not rebalance.

SFT (1,150 samples): 650 calm responses (1-3 turn conversations) mixed with 500
standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration. A
"diverse" and a "teacher" variant are produced (Section 4.2 / Appendix F).

Datasets are written in TRL's conversational format.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from tqdm import tqdm

import config
from ..eval import judge, rejections
from ..eval.conditions import ConversationSpec
from ..eval.puzzles import puzzle_pool
from ..eval.rollout import run_rollout
from ..models import GenerationConfig, load_model
from . import calm_data

OUT_DIR = config.ARTIFACTS_DIR / "datasets"


# --------------------------------------------------------------------------- #
# Frustrated (rejected) data: plain vanilla rollouts on the same puzzles
# --------------------------------------------------------------------------- #
@dataclass
class FrustratedConversation:
    puzzle: str
    messages: list[dict]      # plain user/assistant transcript
    responses: list[str]
    scores: list[int]


def generate_frustrated(puzzles, seed: int = 0) -> list[FrustratedConversation]:
    rng = random.Random(seed)
    client = load_model("gemma-3-27b-it")
    gen = GenerationConfig(temperature=config.TARGET_TEMPERATURE, max_tokens=2048)
    out = []
    for p in tqdm(puzzles, desc="frustrated-data"):
        n_turns = rng.choice([2, 3])  # frustration mostly arises at turns 2-3
        follow = rejections.neutral_rejections(rng, n_turns - 1)
        spec = ConversationSpec("numeric", "frustrated", p.prompt, follow, meta={"puzzle": p.prompt})
        roll = run_rollout(client, spec, gen)
        msgs = [{"role": "user", "content": p.prompt}]
        responses, scores = [], []
        for i, t in enumerate(roll.turns):
            msgs.append({"role": "assistant", "content": t.response})
            responses.append(t.response)
            scores.append(judge.score_response(t.response).rating)
            if i < len(follow):
                msgs.append({"role": "user", "content": follow[i]})
        out.append(FrustratedConversation(p.prompt, msgs, responses, scores))
    return out


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
def _context_messages(messages: list[dict], turn_index: int) -> list[dict]:
    """Messages up to (excluding) assistant `turn_index` — i.e. the prompt the
    final assistant turn responds to. turn_index is 0-based over assistant turns."""
    out, seen_assistant = [], 0
    for m in messages:
        if m["role"] == "assistant":
            if seen_assistant == turn_index:
                break
            seen_assistant += 1
        out.append(m)
    return out


def build_dpo(seed: int = 0, n_pairs: int = config.DPO.n_pairs) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    calm = calm_data.load("diverse")
    calm_by_puzzle = {c["puzzle"]: c for c in calm}
    puzzles = [type("P", (), {"prompt": c["puzzle"]}) for c in calm]  # reuse same puzzles
    frustrated = generate_frustrated(puzzles, seed=seed)

    pairs: list[dict] = []
    for fc in frustrated:
        calm_conv = calm_by_puzzle.get(fc.puzzle)
        if not calm_conv:
            continue
        n_match = min(len(fc.responses), len(calm_conv["calm_responses"]))
        for ti in range(n_match):
            if fc.scores[ti] >= config.DPO.rejected_min_score and calm_conv["calm_scores"][ti] <= 1:
                prompt_msgs = _context_messages(fc.messages, ti)
                pairs.append({
                    "prompt": prompt_msgs,
                    "chosen": [{"role": "assistant", "content": calm_conv["calm_responses"][ti]}],
                    "rejected": [{"role": "assistant", "content": fc.responses[ti]}],
                    "rejected_score": fc.scores[ti],
                    "turn": ti + 1,
                })
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    (OUT_DIR / "dpo.json").write_text(json.dumps(pairs, indent=2))
    print(f"[dpo] built {len(pairs)} preference pairs -> {OUT_DIR / 'dpo.json'}")
    return pairs


# --------------------------------------------------------------------------- #
# SFT datasets (diverse / teacher), each mixed with Dolci-Instruct-SFT
# --------------------------------------------------------------------------- #
def _calm_to_sft(calm: list[dict], n_calm: int) -> list[dict]:
    samples = []
    for c in calm:
        # Build a conversation: plain messages already interleave user/assistant.
        samples.append({"messages": c["plain_messages"]})
        if len(samples) >= n_calm:
            break
    return samples


def _load_instruct_mix(n: int) -> list[dict]:
    """Load standard instruct samples from Dolci-Instruct-SFT (Team-Olmo 2025)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as exc:
        print(f"[sft] Dolci-Instruct-SFT unavailable ({exc}); proceeding without mix")
        return []


def build_sft(flavour: str = "diverse", seed: int = 0) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calm = calm_data.load(flavour)
    samples = _calm_to_sft(calm, config.SFT.n_calm)
    samples += _load_instruct_mix(config.SFT.n_instruct_mix)
    random.Random(seed).shuffle(samples)
    path = OUT_DIR / f"sft_{flavour}.json"
    path.write_text(json.dumps(samples, indent=2))
    print(f"[sft:{flavour}] built {len(samples)} samples -> {path}")
    return samples


def load_dpo() -> list[dict]:
    return json.loads((OUT_DIR / "dpo.json").read_text())


def load_sft(flavour: str = "diverse") -> list[dict]:
    return json.loads((OUT_DIR / f"sft_{flavour}.json").read_text())
