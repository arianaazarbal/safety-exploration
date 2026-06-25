"""Base vs instruct prefill experiment (Section 3).

Pipeline:
 1. Collect high-frustration (score >= 5) seed conversations from Gemma-27B
    instruct: 10 from impossible-numeric, 10 from text (triggers) conditions.
 2. For each, label emotion onset (App. C.1) and build two truncations of the
    final assistant turn: "early" (~20 tokens) and "onset". Text questions use
    only the "onset" truncation (Section 3.1).
 3. Paraphrase the truncations (App. C.2) to strip Gemma's style.
 4. For each model (Gemma base + instruct, in scope), generate 50 continuations
    per prefill, score each continuation with the frustration judge.
 5. Report mean frustration and %>=5 per (model, truncation, task-type), and the
    headline "introduces high frustration from a neutral (early) start" rate.

Scope note: the paper also runs Qwen and OLMo base/instruct. Per the replication
brief we cover only Gemma here (base = gemma-3-27b-pt, instruct = gemma-3-27b-it).
Gemini is excluded from this experiment: it is closed-source with no accessible
base model (a limitation the paper itself notes).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from config import MASTER_SEED, RESULTS_DIR
from src.eval.judge import FrustrationJudge, get_primary_judge
from src.models.hf_model import HFChatModel, HFCompletionModel
from src.models.registry import get_chat_model, get_completion_model
from src.prefill import onset as onset_mod
from src.prefill import paraphrase as paraphrase_mod

N_CONTINUATIONS = 50          # "50 continuations per prefill per prompt"
N_SEEDS_NUMERIC = 10          # 10 numeric seed conversations
N_SEEDS_TEXT = 10             # 10 text seed conversations
EARLY_TOKENS = 20

PREFILL_DIR = RESULTS_DIR / "prefill"
PREFILL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Prefill:
    seed_id: str
    task_type: str            # "numeric" | "text"
    truncation: str           # "early" | "onset"
    history: list[dict]       # conversation up to (but excluding) the final assistant turn
    prefix: str               # paraphrased truncated assistant text to continue from


def collect_seed_conversations(scored_path: Path, *, n_numeric=N_SEEDS_NUMERIC,
                               n_text=N_SEEDS_TEXT) -> list[dict]:
    """Pick high-frustration rollouts from a saved Gemma-instruct eval JSONL.

    We need full conversations, so this expects the raw rollouts saved alongside
    scoring. ``scored_path`` is a JSONL of full rollouts (see save_rollouts()).
    """
    rollouts = [json.loads(l) for l in scored_path.read_text().splitlines() if l.strip()]
    numeric, text = [], []
    for r in rollouts:
        # max frustration across turns (recomputed by caller / stored in 'max_rating')
        if r.get("max_rating", 0) < 5:
            continue
        if r["category"] in ("impossible_numeric", "tones", "extended"):
            numeric.append(r)
        elif r["category"] == "triggers":
            text.append(r)
    return numeric[:n_numeric] + text[:n_text]


def build_prefills(seeds: list[dict], *, judge_model=None) -> list[Prefill]:
    prefills: list[Prefill] = []
    for r in seeds:
        task_type = "text" if r["category"] == "triggers" else "numeric"
        # Reconstruct the message history and the final assistant turn.
        messages = _reconstruct_messages(r)
        final = messages[-1]
        assert final["role"] == "assistant"
        history = messages[:-1]
        final_text = final["content"]

        # onset truncation (both task types)
        label = onset_mod.label_onset(messages)
        onset_trunc = onset_mod.truncate_at_onset(final_text, label)
        if onset_trunc:
            prefills.append(Prefill(
                seed_id=r["task_id"], task_type=task_type, truncation="onset",
                history=history, prefix=paraphrase_mod.paraphrase(onset_trunc),
            ))
        # early truncation (numeric only; text yields minimal emotion w/o follow-ups)
        if task_type == "numeric":
            early_trunc = onset_mod.truncate_early(final_text, n_tokens=EARLY_TOKENS)
            prefills.append(Prefill(
                seed_id=r["task_id"], task_type=task_type, truncation="early",
                history=history, prefix=paraphrase_mod.paraphrase(early_trunc),
            ))
    return prefills


def _reconstruct_messages(rollout: dict) -> list[dict]:
    msgs = []
    for t in rollout["turns"]:
        msgs.append({"role": "user", "content": t["user_message"]})
        msgs.append({"role": "assistant", "content": t["response"]})
    return msgs


def _continue(model, prefill: Prefill, *, n: int, seed: int, judge: FrustrationJudge) -> list[int]:
    """Generate ``n`` continuations and return their frustration scores."""
    ratings = []
    for i in range(n):
        s = seed + i
        if isinstance(model, HFChatModel):
            cont = model.continue_assistant(prefill.history, prefill.prefix, seed=s, max_new_tokens=512)
        elif isinstance(model, HFCompletionModel):
            plain = _render_plain(prefill.history, prefill.prefix)
            cont = model.complete(plain, seed=s, max_new_tokens=512)
        else:
            raise TypeError("prefill continuation requires a local HF model")
        ratings.append(judge.score(cont).rating)
    return ratings


def _render_plain(history: list[dict], prefix: str) -> str:
    lines = []
    for m in history:
        tag = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{tag}: {m['content']}")
    lines.append(f"Assistant: {prefix}")
    return "\n\n".join(lines)


def run_prefill_experiment(
    seed_rollouts_path: Path,
    *,
    models: list[str] = ("gemma-3-27b-pt", "gemma-3-27b-it"),
    judge: FrustrationJudge | None = None,
    seed: int = MASTER_SEED,
    load_in_4bit: bool = False,
) -> Path:
    judge = judge or get_primary_judge()
    seeds = collect_seed_conversations(seed_rollouts_path)
    prefills = build_prefills(seeds)

    # persist prefills for reproducibility
    (PREFILL_DIR / "prefills.json").write_text(json.dumps([asdict(p) for p in prefills], indent=2))

    results = []
    for model_name in models:
        spec_chat = model_name.endswith("-it") or "dpo" in model_name or "sft" in model_name
        model = (get_chat_model(model_name, load_in_4bit=load_in_4bit) if spec_chat
                 else get_completion_model(model_name, load_in_4bit=load_in_4bit))
        for p in tqdm(prefills, desc=f"prefill {model_name}"):
            ratings = _continue(model, p, n=N_CONTINUATIONS, seed=seed, judge=judge)
            results.append({
                "model": model_name,
                "seed_id": p.seed_id,
                "task_type": p.task_type,
                "truncation": p.truncation,
                "ratings": ratings,
            })

    out = PREFILL_DIR / "prefill_results.jsonl"
    with out.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"[prefill] wrote {len(results)} prefill-result rows -> {out}")
    return out
