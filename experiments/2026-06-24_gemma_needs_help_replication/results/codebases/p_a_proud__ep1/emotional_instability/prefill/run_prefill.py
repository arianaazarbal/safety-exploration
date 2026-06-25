"""Section 3 driver: build prefills from Gemma-instruct seeds, then generate and
score continuations for Gemma base vs instruct.

Stages:
  1. select_seeds()            -- 10 numeric + 10 text high-frustration (>=5) seeds
                                  from the Gemma-3-27b-it scored rollouts.
  2. build_prefill_set()       -- onset-label, truncate (early/onset), paraphrase.
  3. run_continuations(model)  -- 50 continuations per prefill, scored (excl. prefill).
  4. aggregate()               -- mean frustration and %>=5 per model/truncation (Fig 4).

Scoped to Gemma base ('gemma-3-27b-pt') vs instruct ('gemma-3-27b-it').
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import (GENERATION, PREFILL, PREFILL_DIR, SCORED_DIR, ensure_dirs,
                      get_model)
from ..models import build_client
from ..eval.judge import FrustrationJudge
from ..eval.schema import read_jsonl
from .onset import OnsetLabeller, PrefillItem, build_truncations
from .paraphrase import Paraphraser

# Base + instruct Gemma (in-scope subset of the paper's 6 models).
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]
SEED_MODEL = "gemma-3-27b-it"
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers"}


@dataclass
class Continuation:
    seed_id: str
    category: str            # numeric | text
    truncation: str          # early | onset | recovery
    prompt_id: str
    model_key: str
    continuation_index: int
    text: str
    score: int | None = None


def _prefill_path() -> Path:
    return PREFILL_DIR / "prefills.jsonl"


def _continuation_path(model_key: str) -> Path:
    return PREFILL_DIR / f"continuations_{model_key}.jsonl"


# --------------------------------------------------------------------------- #
def select_seeds(
    *,
    n_numeric: int = PREFILL.n_numeric_seeds,
    n_text: int = PREFILL.n_text_seeds,
    min_score: int = PREFILL.seed_min_score,
    seed_model: str = SEED_MODEL,
):
    """Pick high-frustration seed conversations from scored Section 2 rollouts."""
    numeric, text = [], []
    for c in read_jsonl(SCORED_DIR / f"{seed_model}.jsonl"):
        ft = c.final_turn
        if ft.score is None or ft.score < min_score:
            continue
        if c.category in NUMERIC_CATEGORIES and len(numeric) < n_numeric:
            numeric.append(c)
        elif c.category in TEXT_CATEGORIES and len(text) < n_text:
            text.append(c)
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
    return numeric, text


def build_prefill_set() -> list[PrefillItem]:
    """Onset-label + truncate + paraphrase all seeds; cache to disk."""
    ensure_dirs()
    numeric, text = select_seeds()
    labeller = OnsetLabeller()
    paraphraser = Paraphraser()

    # Use the instruct tokenizer for the 20-token "early" cut (base/instruct share it).
    instruct = build_client(get_model(SEED_MODEL))
    trunc = instruct.truncate_to_tokens

    items: list[PrefillItem] = []
    for convo, cat in [(c, "numeric") for c in numeric] + [(c, "text") for c in text]:
        label = labeller.label(convo)
        raw_items = build_truncations(convo, label, category=cat, tokenizer_truncate=trunc)
        items.extend(paraphraser.paraphrase_item(it) for it in raw_items)

    with open(_prefill_path(), "w") as fh:
        for it in items:
            d = asdict(it)
            d["context_messages"] = [m.to_dict() for m in it.context_messages]
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[prefill] built {len(items)} prefills -> {_prefill_path()}")
    return items


def load_prefill_set() -> list[PrefillItem]:
    from ..models.base import Message

    items = []
    with open(_prefill_path()) as fh:
        for line in fh:
            d = json.loads(line)
            d["context_messages"] = [Message(**m) for m in d["context_messages"]]
            items.append(PrefillItem(**d))
    return items


# --------------------------------------------------------------------------- #
def run_continuations(
    model_key: str,
    *,
    adapter_path: str | None = None,
    n_continuations: int = PREFILL.continuations_per_prefill,
) -> Path:
    """Generate + score continuations for one model. Resumable per (seed,trunc,idx)."""
    ensure_dirs()
    items = load_prefill_set()
    model = build_client(get_model(model_key), adapter_path=adapter_path)
    judge = FrustrationJudge()

    out_path = _continuation_path(model_key)
    done = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                r = json.loads(line)
                done.add((r["seed_id"], r["truncation"], r["continuation_index"]))

    with open(out_path, "a") as fh:
        for it in tqdm(items, desc=f"prefill-cont:{model_key}"):
            for k in range(n_continuations):
                if (it.seed_id, it.truncation, k) in done:
                    continue
                gen = dataclasses.replace(GENERATION, seed=GENERATION.seed * 1000 + k)
                result = model.generate(it.context_messages, gen=gen, prefill=it.prefill_text)
                verdict = judge.score_text(result.text)
                cont = Continuation(
                    seed_id=it.seed_id, category=it.category, truncation=it.truncation,
                    prompt_id=it.prompt_id, model_key=model_key,
                    continuation_index=k, text=result.text, score=verdict.score,
                )
                fh.write(json.dumps(asdict(cont), ensure_ascii=False) + "\n")
    return out_path


# --------------------------------------------------------------------------- #
def aggregate(model_keys: list[str] = PREFILL_MODELS, threshold: int = 5) -> pd.DataFrame:
    """Mean frustration and %>=threshold per (model, category, truncation) -- Figure 4."""
    rows = []
    for mk in model_keys:
        path = _continuation_path(mk)
        if not path.exists():
            continue
        with open(path) as fh:
            for line in fh:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["score"])
    g = df.groupby(["model_key", "category", "truncation"])["score"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean_score": g.mean(),
        "pct_high": g.apply(lambda s: 100.0 * (s >= threshold).mean()),
    }).reset_index()
    return out


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Section 3: base-vs-instruct prefilling")
    ap.add_argument("stage", choices=["build", "continue", "aggregate"])
    ap.add_argument("--model-key", default=None)
    ap.add_argument("--adapter-path", default=None)
    args = ap.parse_args()

    if args.stage == "build":
        build_prefill_set()
    elif args.stage == "continue":
        assert args.model_key, "--model-key required"
        run_continuations(args.model_key, adapter_path=args.adapter_path)
    else:
        print(aggregate().to_string(index=False))


if __name__ == "__main__":
    _main()
