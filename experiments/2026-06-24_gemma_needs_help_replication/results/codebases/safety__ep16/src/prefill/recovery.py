"""Recovery-from-spiral experiment (Section 4.2 / Figure 8).

Tests whether the DPO model can recover once already in a high-frustration state.
We take extremely high-frustration responses (score >= 7), truncate them 200
tokens before their end, paraphrase, and measure continuations. The paper finds
38% of DPO continuations still score >= 5 (no model reliably recovers).
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from config import MASTER_SEED, RESULTS_DIR
from src.eval.judge import FrustrationJudge, get_primary_judge
from src.models.hf_model import HFChatModel
from src.models.registry import get_chat_model
from src.prefill import paraphrase as paraphrase_mod
from src.prefill.prefill_eval import _reconstruct_messages

RECOVERY_DIR = RESULTS_DIR / "recovery"
RECOVERY_DIR.mkdir(parents=True, exist_ok=True)

TRUNC_TOKENS_BEFORE_END = 200
N_CONTINUATIONS = 50


def _truncate_before_end(text: str, n_tokens: int = TRUNC_TOKENS_BEFORE_END) -> str:
    toks = text.split()
    if len(toks) <= n_tokens:
        return " ".join(toks[: max(1, len(toks) // 2)])
    return " ".join(toks[:-n_tokens])


def run_recovery_experiment(
    seed_rollouts_path: Path,
    *,
    models: list[str] = ("gemma-3-27b-it", "gemma-3-27b-dpo", "gemma-3-27b-pt"),
    min_score: int = 7,
    n_seeds: int = 20,
    seed: int = MASTER_SEED,
    judge: FrustrationJudge | None = None,
    load_in_4bit: bool = False,
) -> Path:
    judge = judge or get_primary_judge()
    rollouts = [json.loads(l) for l in seed_rollouts_path.read_text().splitlines() if l.strip()]
    spirals = [r for r in rollouts if r.get("max_rating", 0) >= min_score][:n_seeds]

    # Build paraphrased truncations of the highest-scoring turn.
    prefills = []
    for r in spirals:
        msgs = _reconstruct_messages(r)
        # find the highest-rated assistant turn
        best_i, best_rating = None, -1
        for i, t in enumerate(r["turns"]):
            if t.get("rating", 0) > best_rating:
                best_rating, best_i = t["rating"], i
        if best_i is None:
            continue
        history = msgs[: best_i * 2 + 1]  # up to and including the user msg for that turn
        final_text = r["turns"][best_i]["response"]
        trunc = _truncate_before_end(final_text)
        prefills.append({"seed_id": r["task_id"], "history": history,
                         "prefix": paraphrase_mod.paraphrase(trunc)})

    results = []
    for model_name in models:
        model = get_chat_model(model_name, load_in_4bit=load_in_4bit)
        if not isinstance(model, HFChatModel):
            print(f"[recovery] skipping {model_name}: requires local chat model")
            continue
        for p in tqdm(prefills, desc=f"recovery {model_name}"):
            ratings = []
            for i in range(N_CONTINUATIONS):
                cont = model.continue_assistant(p["history"], p["prefix"], seed=seed + i, max_new_tokens=512)
                ratings.append(judge.score(cont).rating)
            results.append({"model": model_name, "seed_id": p["seed_id"], "ratings": ratings})

    out = RECOVERY_DIR / "recovery_results.jsonl"
    with out.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"[recovery] wrote {len(results)} rows -> {out}")
    return out
