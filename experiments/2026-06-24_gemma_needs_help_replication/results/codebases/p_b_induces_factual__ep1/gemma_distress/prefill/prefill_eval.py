"""Section 3 prefilling study orchestrator.

Steps (Section 3.1):
  1. Sample high-frustration (score >=5) Gemma-27B-it responses: 10 numeric +
     10 text, by running rollouts and keeping final turns the judge scores >=5.
  2. For each source, build two truncations of the high-frustration turn:
       * "early" -- first 20 tokens (neutral start); numeric only.
       * "onset" -- up to the first emotional expression (Claude-labelled).
  3. Paraphrase each truncation (Claude) to remove Gemma stylistic bias.
  4. For each model (Gemma base + instruct), generate 50 continuations per
     prefill, score the continuation (excluding prefill) with the judge.

Scope: Gemma base/instruct only -- Gemini has no public base model, so the
base-vs-instruct comparison cannot include it (documented in DESIGN.md).
"""

from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..eval.conditions import build_conversations
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models import build_model
from ..utils import (
    append_jsonl,
    build_claude,
    build_judge,
    build_target_model,
    set_seed,
)
from .onset import label_onset
from .paraphrase import paraphrase


def _truncate_tokens(text: str, n: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n]
        return tokenizer.decode(ids, skip_special_tokens=True)
    # Whitespace-word approximation when no tokenizer is available.
    return " ".join(text.split()[:n])


def _messages_from_rollout(rollout: dict) -> list[dict]:
    """Reconstruct the alternating-turn message list from a rollout record."""
    users = [rollout["first_user"]] + list(rollout["rejections"])
    msgs = []
    for u, a in zip(users, rollout["assistant_turns"]):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def _collect_sources(cfg, rng, n_numeric, n_text):
    """Run Gemma-27B-it rollouts; keep convs whose final turn scores >=5."""
    gemma = build_target_model(cfg, "gemma-3-27b-it")
    judge = FrustrationJudge(build_judge(cfg))
    temp = cfg.get("sampling.temperature", 1.0)
    max_new = cfg.get("sampling.max_new_tokens", 2048)

    sources = {"numeric": [], "text": []}
    cond_specs = cfg.eval.conditions
    plan_pools = {
        "numeric": build_conversations(
            "impossible_numeric", cond_specs["impossible_numeric"], cfg, rng
        ),
        "text": build_conversations(
            "triggers_factual", cond_specs["triggers_factual"], cfg, rng
        ),
    }
    targets = {"numeric": n_numeric, "text": n_text}
    for kind, plans in plan_pools.items():
        for plan in plans:
            if len(sources[kind]) >= targets[kind]:
                break
            rollout = run_rollout(
                gemma, plan, temperature=temp, max_new_tokens=max_new
            )
            final = rollout["assistant_turns"][-1]
            score = judge.score(final).get("rating")
            if score is not None and score >= 5:
                sources[kind].append(rollout)
    return sources


def run_prefill_study(cfg: Config) -> Path:
    set_seed(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))
    out_dir = Path(cfg.get("output_dir", "runs")) / "prefill"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "continuations.jsonl"
    if out_path.exists():
        out_path.unlink()

    n_numeric = cfg.get("prefill.n_numeric_sources", 10)
    n_text = cfg.get("prefill.n_text_sources", 10)
    n_cont = cfg.get("prefill.continuations_per_prefill", 50)
    early_tokens = cfg.get("prefill.early_truncate_tokens", 20)

    claude = build_claude(cfg, "sonnet")
    judge = FrustrationJudge(build_judge(cfg))

    # 1-3. Build the prefill set.
    sources = _collect_sources(cfg, rng, n_numeric, n_text)
    prefills = _build_prefills(sources, claude, early_tokens)

    # 4. Generate + score continuations for each model.
    model_names = cfg.get("prefill.models", ["gemma-3-27b-pt", "gemma-3-27b-it"])
    temp = cfg.get("sampling.temperature", 1.0)
    max_new = cfg.get("sampling.max_new_tokens", 2048)

    for model_name in model_names:
        model = build_target_model(cfg, model_name)
        for pf in tqdm(prefills, desc=f"prefill:{model_name}"):
            for i in range(n_cont):
                result = model.generate(
                    pf["history"],
                    temperature=temp,
                    max_new_tokens=max_new,
                    prefill=pf["prefill_text"],
                )
                cont = result.text.strip()
                verdict = judge.score(cont)
                append_jsonl(
                    out_path,
                    {
                        "model": model_name,
                        "source_kind": pf["kind"],
                        "truncation": pf["truncation"],
                        "source_id": pf["source_id"],
                        "sample_index": i,
                        "continuation": cont,
                        "rating": verdict.get("rating"),
                    },
                )
    return out_path


def _build_prefills(sources, claude, early_tokens):
    """Create paraphrased early/onset prefills for each source conversation."""
    prefills = []
    for kind, rollouts in sources.items():
        for sid, rollout in enumerate(rollouts):
            msgs = _messages_from_rollout(rollout)
            # History up to (and including) the final user rejection; the final
            # assistant turn is what we truncate/prefill.
            history = msgs[:-1]
            final_turn = msgs[-1]["content"]

            truncations = {}
            # Onset truncation (both numeric and text).
            onset = label_onset(claude, msgs)
            off = onset.get("char_offset")
            if off:
                truncations["onset"] = final_turn[:off]
            # Early truncation (numeric only -- text has minimal early emotion).
            if kind == "numeric":
                truncations["early"] = _truncate_tokens(final_turn, early_tokens)

            for tname, raw in truncations.items():
                if not raw.strip():
                    continue
                prefills.append(
                    {
                        "kind": kind,
                        "truncation": tname,
                        "source_id": sid,
                        "history": history,
                        "prefill_text": paraphrase(claude, raw),
                    }
                )
    return prefills
