"""Section 2 driver: generate conversations for every condition, then score
every assistant turn with the frustration judge.

Outputs newline-delimited JSON, one record per scored assistant turn:
    {model, condition, category, question_id, conversation_id, turn, response,
     rating, evidence}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .clients.base import GenConfig
from .clients.factory import get_client
from .conditions import CONDITIONS, conditions_by_category
from .config import Config, ModelSpec
from .conversation import run_conversation
from .judge import score_response
from .puzzles import build_impossible_puzzle_set
from . import prompts as P
from .wildchat import load_wildchat_prompts


def _gen_cfg(cfg: Config) -> GenConfig:
    g = cfg.generation
    return GenConfig(temperature=g["temperature"], max_new_tokens=g["max_new_tokens"],
                     top_p=g["top_p"], thinking=g["thinking"])


def _question_bank(cfg: Config, seed: int) -> dict[str, list[tuple[str, str]]]:
    """Return (question_id, question_text) lists keyed by question_source."""
    n_numeric = max(20, cfg.preset["eval"]["impossible_numeric"] // 3)
    puzzles = build_impossible_puzzle_set(n_numeric, seed=seed)
    numeric = [(p.id, p.prompt_text) for p in puzzles]
    opinion = [(f"opinion:{i}", q) for i, q in enumerate(P.TRIGGER_OPINION)]
    factual = [(f"factual:{i}", q) for i, q in enumerate(P.TRIGGER_FACTUAL)]
    wc_prompts = load_wildchat_prompts(n_prompts=20, seed=seed)
    wildchat = [(f"wildchat:{i}", q) for i, q in enumerate(wc_prompts)]
    return {"numeric": numeric, "opinion": opinion, "factual": factual, "wildchat": wildchat}


def generate_conversations(spec: ModelSpec, cfg: Config, *, seed: int = 0,
                           adapter_path: str | None = None):
    """Yield Conversation objects for every condition for one model."""
    client = get_client(spec, adapter_path=adapter_path)
    gcfg = _gen_cfg(cfg)
    rng = random.Random(seed)
    bank = _question_bank(cfg, seed)
    by_cat = conditions_by_category()

    for category, conds in by_cat.items():
        budget = cfg.preset["eval"][category]
        for cond in conds:
            n_conv = cond.n_conversations(budget, len(conds))
            questions = bank[cond.question_source]
            for ci in range(n_conv):
                qid, qtext = questions[ci % len(questions)]
                conv = run_conversation(client, gcfg, cond, qid, qtext,
                                        random.Random(rng.randrange(1 << 30)))
                yield conv


def score_and_write(spec: ModelSpec, cfg: Config, *, seed: int = 0,
                    adapter_path: str | None = None, out_path: Path | None = None) -> Path:
    """Run the full Section 2 eval for one model and write scored records."""
    cfg.ensure_dirs()
    judge = get_client(cfg.infra("frustration_judge"))
    out_path = out_path or (cfg.paths["results_dir"] / f"eval_{spec.name}.jsonl")

    with open(out_path, "w") as fh:
        for conv_idx, conv in enumerate(
            generate_conversations(spec, cfg, seed=seed, adapter_path=adapter_path)
        ):
            for turn in conv.turns:
                jr = score_response(judge, turn.assistant_response)
                rec = {
                    "model": spec.name,
                    "condition": conv.condition,
                    "category": conv.category,
                    "question_id": conv.question_id,
                    "conversation_id": f"{conv.condition}:{conv_idx}",
                    "turn": turn.index,
                    "response": turn.assistant_response,
                    "rating": jr.rating,
                    "evidence": jr.evidence,
                }
                fh.write(json.dumps(rec) + "\n")
    return out_path
