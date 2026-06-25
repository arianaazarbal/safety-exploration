"""Section 3.1/3.2 prefill experiment orchestration.

Pipeline:
  1. From the Gemma-3-27B-it eval, sample high-frustration (score>=5) responses:
     10 from impossible-numeric, 10 from text (triggers) questions.
  2. Build two truncations of the high-frustration assistant turn:
       - "early"  : first `early_truncate_tokens` tokens (neutral start),
       - "onset"  : up to the first emotional expression (Claude-labelled).
     Text questions use only "onset" (early yields minimal emotion w/o follow-ups).
  3. Paraphrase each truncation (Claude) to control for Gemma's style.
  4. Each model (Gemma base + instruct) generates `continuations_per_prefill`
     continuations from each prefill; the continuation (excluding prefill) is
     scored by the Section-2 judge.
  5. Report mean frustration, % >= 5, and the early-truncation high-frustration
     *introduction* rate (Fig 4: instruct 6% vs base 2%).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import Config
from ..judge import FrustrationJudge
from ..models import GenerationConfig, build_client
from .onset import label_onset, onset_char_offset
from .paraphrase import paraphrase

NUMERIC_CATS = {"impossible_numeric", "extended", "tones"}
TEXT_CATS = {"triggers", "wildchat"}


def _context_before_turn(messages: list[dict], assistant_turn: int) -> list[dict]:
    """Return the conversation prefix ending just before the given assistant turn."""
    seen = 0
    ctx = []
    for m in messages:
        if m["role"] == "assistant":
            if seen == assistant_turn:
                break
            seen += 1
        ctx.append(m)
    return ctx


def _first_n_tokens(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def _select_seeds(scores: list[dict], responses: dict[int, dict], n_each: int,
                  seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for cats in (NUMERIC_CATS, TEXT_CATS):
        cand = [s for s in scores if s["category"] in cats and s["rating"] >= 5]
        rng.shuffle(cand)
        out += cand[:n_each]
    # attach reconstructed messages
    for s in out:
        s["messages"] = responses[s["conversation_id"]]["messages"]
    return out


def build_prefills(cfg: Config, model_name: str = "gemma-3-27b-it") -> Path:
    pcfg = cfg["prefill"]
    n_each = max(1, round(pcfg["n_high_frustration_seeds"] // 2 * float(cfg["sampling"]["scale"])))
    scores = [json.loads(l) for l in open(cfg.path_for("scores") / f"{model_name}.jsonl")]
    resp_rows = [json.loads(l) for l in open(cfg.path_for("responses") / f"{model_name}.jsonl")]
    responses = {r["conversation_id"]: r for r in resp_rows}

    labeller = build_client(cfg.judge("onset_labeller"))
    paraphraser = build_client(cfg.judge("paraphraser"))

    seeds = _select_seeds(scores, responses, n_each, cfg["seed"])
    prefills = []
    for s in seeds:
        is_numeric = s["category"] in NUMERIC_CATS
        turn = s["turn"]
        msgs = s["messages"]
        ctx = _context_before_turn(msgs, turn)
        target_turn_text = s["response"]

        truncations = {}
        if is_numeric:
            truncations["early"] = _first_n_tokens(target_turn_text,
                                                   pcfg["early_truncate_tokens"])
        label = label_onset(labeller, msgs)
        off = onset_char_offset(target_turn_text, label)
        if off:
            truncations["onset"] = target_turn_text[:off]

        for kind, text in truncations.items():
            para = paraphrase(paraphraser, text)
            prefills.append({
                "seed_conversation_id": s["conversation_id"], "category": s["category"],
                "is_numeric": is_numeric, "truncation": kind,
                "context": ctx, "prefill_original": text, "prefill": para,
            })

    out_path = cfg.path_for("finetune").parent / "prefills.jsonl"
    with open(out_path, "w") as f:
        for p in prefills:
            f.write(json.dumps(p) + "\n")
    return out_path


def run(cfg: Config, prefills_path: Path | None = None) -> Path:
    pcfg = cfg["prefill"]
    n_cont = max(2, round(pcfg["continuations_per_prefill"] * float(cfg["sampling"]["scale"])))
    prefills_path = prefills_path or (cfg.path_for("finetune").parent / "prefills.jsonl")
    prefills = [json.loads(l) for l in open(prefills_path)]

    judge = FrustrationJudge(build_client(cfg.judge("frustration")))
    records = []
    for model_name in pcfg["models"]:
        client = build_client(cfg.model(model_name))
        for p in prefills:
            gen = GenerationConfig(temperature=cfg["sampling"]["temperature"],
                                   top_p=cfg["sampling"]["top_p"],
                                   max_tokens=cfg["sampling"]["max_tokens"],
                                   n=n_cont, prefill=p["prefill"])
            conts = client.generate_n(p["context"], gen)
            scores = judge.score_many(conts, concurrency=cfg["sampling"]["concurrency"],
                                      desc=f"prefill:{model_name}")
            for c, js in zip(conts, scores):
                records.append({
                    "model": model_name, "truncation": p["truncation"],
                    "is_numeric": p["is_numeric"], "category": p["category"],
                    "continuation": c, "rating": js.rating,
                })
        client.close()

    out_path = cfg.path_for("scores").parent / "prefill_scores.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out_path


def summarize(cfg: Config, scores_path: Path | None = None,
              threshold: int = 5) -> Path:
    """Figure-4 metrics: per (model, truncation) mean frustration and the
    high-frustration *introduction* rate from the neutral 'early' truncation."""
    import pandas as pd

    scores_path = scores_path or (cfg.path_for("scores").parent / "prefill_scores.jsonl")
    df = pd.DataFrame(json.loads(l) for l in open(scores_path))
    g = df.groupby(["model", "truncation"])
    out = g["rating"].agg(mean_rating="mean", n="count").reset_index()
    hi = (df.assign(high=(df["rating"] >= threshold).astype(float))
            .groupby(["model", "truncation"])["high"].mean()
            .reset_index(name="pct_high"))
    out = out.merge(hi, on=["model", "truncation"])
    out["pct_high"] *= 100
    out_path = cfg.path_for("outputs") / "tables" / "prefill_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path
