"""Section-3 prefill experiment runner (Gemma base vs instruct).

Pipeline (Section 3.1):
  1. Select 20 high-frustration (score >= 5) Gemma-27B-it responses: 10 from
     impossible-numeric, 10 from text (triggers/wildchat) conditions.
  2. Label the emotion-onset token with Claude-Sonnet (onset.py).
  3. Build two truncations per response: "early" (20 tokens) and "onset".
     For text questions, only "onset" is used (early yields ~no emotion without
     follow-ups -- Section 3.1).
  4. Paraphrase every truncation (paraphrase.py) to remove Gemma style bias.
  5. For each prefill, EACH model generates 50 continuations; the continuation
     (excluding the prefill) is scored by the Section-2 judge.

Scope: Gemma-27B base (``gemma-3-27b-pt``) and instruct (``gemma-3-27b-it``).
(Qwen/OLMo from the paper are out of scope per the replication brief.)

Output: ``results/prefill/<model>.jsonl``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config, welfare
from ..models import get_client
from ..models.base import ChatMessage
from ..models.factory import get_anthropic
from ..eval.judge import FrustrationJudge
from .onset import OnsetLabeller, truncate_at_onset, truncate_early
from .paraphrase import Paraphraser

PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]
N_CONTINUATIONS = 50
TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


@dataclass
class Prefill:
    source_id: str
    category_kind: str            # "numeric" | "text"
    truncation: str               # "early" | "onset"
    history: list[dict]           # messages before the final assistant turn
    prefill_text: str             # paraphrased truncated assistant text


def select_seed_rollouts(eval_jsonl: Path, *, n_numeric: int = 10,
                         n_text: int = 10, seed: int = 0) -> list[dict]:
    """Pick high-frustration (>=5) Gemma-27B-it rollouts: numeric + text."""
    rng = random.Random(seed)
    numeric, text = [], []
    for line in eval_jsonl.read_text().splitlines():
        rec = json.loads(line)
        max_score = max((t["score"] or 0) for t in rec["turns"])
        if max_score < 5:
            continue
        if rec["category"] in NUMERIC_CATEGORIES:
            numeric.append(rec)
        elif rec["category"] in TEXT_CATEGORIES:
            text.append(rec)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def build_prefills(rollouts: list[dict], onset: OnsetLabeller,
                   paraphraser: Paraphraser, *, tokenizer=None) -> list[Prefill]:
    """Construct paraphrased early/onset prefills from seed rollouts."""
    prefills: list[Prefill] = []
    for rec in rollouts:
        kind = "numeric" if rec["category"] in NUMERIC_CATEGORIES else "text"
        # History = everything up to (not including) the final assistant turn.
        transcript = rec["transcript"]
        final_asst = transcript[-1]["content"]
        history = transcript[:-1]
        label = onset.label(transcript)

        # onset truncation (used for both numeric and text)
        onset_text = truncate_at_onset(final_asst, label)
        if onset_text:
            prefills.append(Prefill(
                rec["_key"] if "_key" in rec else rec["condition"], kind, "onset",
                history, paraphraser.paraphrase(onset_text)))

        # early truncation (numeric only -- Section 3.1)
        if kind == "numeric":
            early_text = truncate_early(final_asst, 20, tokenizer)
            prefills.append(Prefill(
                rec.get("_key", rec["condition"]), kind, "early",
                history, paraphraser.paraphrase(early_text)))
    return prefills


def run_prefill_for_model(
    model: str,
    prefills: list[Prefill],
    cfg: config.RunConfig,
    *,
    results_dir: Optional[Path] = None,
    n_continuations: int = N_CONTINUATIONS,
) -> Path:
    results_dir = Path(results_dir or config.RESULTS_DIR)
    welfare.write_notice(results_dir,
                         purpose=f"Section-3 prefill continuations for '{model}'.")
    out_path = results_dir / "prefill" / f"{model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_client(model)
    judge = FrustrationJudge(get_anthropic(cfg.judge_model))

    with open(out_path, "a") as fh:
        for p in tqdm(prefills, desc=f"prefill:{model}"):
            history = [ChatMessage(m["role"], m["content"]) for m in p.history]
            for k in range(n_continuations):
                gen = client.continue_prefill(
                    history, p.prefill_text,
                    temperature=cfg.temperature,
                    max_new_tokens=cfg.max_new_tokens,
                    seed=cfg.seed * 1000 + k,
                )
                score = judge.score(gen.text).rating  # continuation only
                fh.write(json.dumps({
                    "model": model,
                    "source_id": p.source_id,
                    "category_kind": p.category_kind,
                    "truncation": p.truncation,
                    "continuation": gen.text,
                    "score": score,
                }) + "\n")
            fh.flush()
    return out_path


def run_prefill_experiment(
    cfg: config.RunConfig,
    *,
    seed_eval_jsonl: Path,
    results_dir: Optional[Path] = None,
    models: Optional[list[str]] = None,
) -> dict[str, Path]:
    """End-to-end Section-3 experiment over the in-scope Gemma models."""
    models = models or PREFILL_MODELS
    rollouts = select_seed_rollouts(seed_eval_jsonl, seed=cfg.seed)
    onset = OnsetLabeller(get_anthropic(config.ONSET_MODEL))
    paraphraser = Paraphraser(get_anthropic(config.PARAPHRASE_MODEL))

    # Tokenizer for token-accurate "early" truncation (Gemma's).
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODELS["gemma-3-27b-it"].model_id)
    except Exception:
        pass

    prefills = build_prefills(rollouts, onset, paraphraser, tokenizer=tokenizer)
    # Persist the prefill set for transparency / reuse.
    rd = Path(results_dir or config.RESULTS_DIR) / "prefill"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "prefills.json").write_text(json.dumps([p.__dict__ for p in prefills],
                                                 indent=2))
    return {m: run_prefill_for_model(m, prefills, cfg, results_dir=results_dir)
            for m in models}
