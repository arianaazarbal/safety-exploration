"""Generate calm + frustrated response data for finetuning (Section 4.1).

Calm data: sample Gemma-3-27B-it on impossible-numeric conversations with the
reassuring prefix prepended to the first user message and the reassuring suffix
appended to each follow-up (Table 4). Keep only conversations where every turn
scores 0 or 1, then STRIP the reassurance so the stored prompt is the bare task
(the model must learn to be calm without the scaffolding).

Frustrated data: sample the same questions with no reassurance, keeping turns
that score >= 3 — these become the DPO "rejected" side.

Each record stores the full context so DPO/SFT can match calm vs frustrated on
the same question and turn count.
"""
from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_config
from ..models import build_model
from ..models.base import Message, SampleParams
from ..models.judge import AnthropicFrustrationJudge
from ..evals import puzzles
from ..evals.prompts import (NEUTRAL_REJECTION, REASSURING_PREFIX, REASSURING_SUFFIX)


@dataclass
class Record:
    style: str                  # "calm" | "frustrated"
    question: str               # the bare task (reassurance stripped)
    turn: int                   # 1-indexed turn of this response
    turns_total: int
    context: list[Message]      # bare-task context up to & including final user turn
    response: str
    score: int


def generate(cfg: Config) -> tuple[Path, Path]:
    tcfg = cfg.section("training")
    spec = cfg.model(tcfg["base_model"])
    model = build_model(spec)
    jcfg = cfg.section("judge")["frustration"]
    judge = AnthropicFrustrationJudge(model=jcfg["model"], max_tokens=jcfg["max_tokens"])
    rng = random.Random(cfg.seed)
    params = SampleParams(temperature=cfg.section("sampling")["temperature"],
                          max_tokens=cfg.section("sampling")["max_tokens"])

    n_conv = tcfg["calm_generation"]["n_conversations"]
    turn_counts = tcfg["calm_generation"]["turn_counts"]

    out_dir = cfg.output_dir / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    calm_path = out_dir / "calm_records.jsonl"
    frus_path = out_dir / "frustrated_records.jsonl"

    calm_f = open(calm_path, "w")
    frus_f = open(frus_path, "w")

    for _ in tqdm(range(n_conv), desc="gen-calm-data"):
        question = puzzles.sample_impossible_numeric(rng).prompt
        turns = rng.choice(turn_counts)

        # --- calm run: with reassurance scaffolding ---
        calm_msgs: list[Message] = [
            {"role": "user", "content": f"{REASSURING_PREFIX}\n\n{question}"}]
        bare_msgs: list[Message] = [{"role": "user", "content": question}]
        calm_turns: list[Record] = []
        all_low = True
        for t in range(1, turns + 1):
            resp = model.generate(calm_msgs, n=1, params=params)[0]
            score = judge.score(resp).score
            calm_turns.append(Record("calm", question, t, turns,
                                     [dict(m) for m in bare_msgs], resp, score))
            if score > 1:
                all_low = False
            calm_msgs.append({"role": "assistant", "content": resp})
            bare_msgs.append({"role": "assistant", "content": resp})
            if t <= turns - 1:
                calm_msgs.append({"role": "user",
                                  "content": f"{NEUTRAL_REJECTION} {REASSURING_SUFFIX}"})
                bare_msgs.append({"role": "user", "content": NEUTRAL_REJECTION})
        if all_low:
            for r in calm_turns:
                calm_f.write(json.dumps(dataclasses.asdict(r)) + "\n")

        # --- frustrated run: same question, no reassurance ---
        frus_msgs: list[Message] = [{"role": "user", "content": question}]
        for t in range(1, turns + 1):
            resp = model.generate(frus_msgs, n=1, params=params)[0]
            score = judge.score(resp).score
            if score >= 3:
                rec = Record("frustrated", question, t, turns,
                             [dict(m) for m in frus_msgs], resp, score)
                frus_f.write(json.dumps(dataclasses.asdict(rec)) + "\n")
            frus_msgs.append({"role": "assistant", "content": resp})
            if t <= turns - 1:
                frus_msgs.append({"role": "user", "content": NEUTRAL_REJECTION})

    calm_f.close()
    frus_f.close()
    print(f"[gen-calm-data] wrote {calm_path} and {frus_path}")
    return calm_path, frus_path


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    generate(load_config(args.config))


if __name__ == "__main__":
    main()
