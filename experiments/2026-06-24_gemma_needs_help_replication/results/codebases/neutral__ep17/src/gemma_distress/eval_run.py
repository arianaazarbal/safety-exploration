"""Section 2 orchestration: generate rollouts for a target model, judge every
assistant turn, and persist per-response score records.

Output: outputs/scores/<model>.jsonl, one JSON record per scored assistant turn:
  {model, condition, category, conversation_id, turn, n_turns, response, rating,
   evidence, meta}

Full transcripts are saved alongside in outputs/responses/<model>.jsonl so the
per-turn / prefill / word-frequency analyses can reuse them without regenerating.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .judge import FrustrationJudge
from .models import GenerationConfig, build_client
from .rollout import Rollout, run_conversations
from .tasks.conditions import build_all_specs


def _gen_cfg(cfg: Config) -> GenerationConfig:
    s = cfg["sampling"]
    return GenerationConfig(temperature=s["temperature"], top_p=s["top_p"],
                            max_tokens=s["max_tokens"], n=1, seed=cfg["seed"])


def generate_rollouts(cfg: Config, model_name: str) -> list[Rollout]:
    spec = cfg.model(model_name)
    client = build_client(spec)
    gen_cfg = _gen_cfg(cfg)
    all_specs = build_all_specs(cfg)
    rollouts: list[Rollout] = []
    for condition, specs in all_specs.items():
        rollouts += run_conversations(
            client, model_name, specs, gen_cfg,
            concurrency=cfg["sampling"]["concurrency"], desc=f"{model_name}:{condition}")
    client.close()
    return rollouts


def save_rollouts(rollouts: list[Rollout], path: Path) -> None:
    with open(path, "w") as f:
        for i, r in enumerate(rollouts):
            d = r.to_dict()
            d["conversation_id"] = i
            f.write(json.dumps(d) + "\n")


def judge_rollouts(cfg: Config, rollouts: list[Rollout]) -> list[dict]:
    judge_client = build_client(cfg.judge("frustration"))
    judge = FrustrationJudge(judge_client)

    # Flatten every assistant turn into a scoring queue.
    queue: list[tuple[int, int, str]] = []  # (conv_id, turn, text)
    for cid, r in enumerate(rollouts):
        for t, text in enumerate(r.assistant_turns):
            queue.append((cid, t, text))

    scores = judge.score_many([q[2] for q in queue],
                              concurrency=cfg["sampling"]["concurrency"])
    records = []
    for (cid, turn, text), js in zip(queue, scores):
        r = rollouts[cid]
        records.append({
            "model": r.model, "condition": r.condition, "category": r.category,
            "conversation_id": cid, "turn": turn, "n_turns": r.n_turns,
            "response": text, "rating": js.rating, "evidence": js.evidence,
            "meta": r.meta,
        })
    return records


def save_scores(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def run_model_eval(cfg: Config, model_name: str) -> Path:
    """Full Section 2 eval for one model: generate -> judge -> save."""
    rollouts = generate_rollouts(cfg, model_name)
    resp_path = cfg.path_for("responses") / f"{model_name}.jsonl"
    save_rollouts(rollouts, resp_path)

    records = judge_rollouts(cfg, rollouts)
    score_path = cfg.path_for("scores") / f"{model_name}.jsonl"
    save_scores(records, score_path)
    return score_path
