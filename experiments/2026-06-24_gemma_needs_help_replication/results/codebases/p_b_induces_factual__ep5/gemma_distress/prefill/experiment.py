"""Run the Section 3 prefilling experiment (Gemma base vs instruct).

Pipeline:
  1. Take seed high-frustration (score>=5) Gemma-27B-instruct responses: 10
     numeric + 10 text (selected from a scored Section 2 file).
  2. For each, build paraphrased "early" and "onset" prefills (onset-only for
     text).
  3. Each model (Gemma-27B base + instruct) generates 50 continuations per
     prefill via continue_from().
  4. The judge scores each continuation (excluding the prefill).

Scope note: the paper compares six models (base+instruct Gemma/Qwen/OLMo). This
replication is scoped to Gemma, so it runs Gemma-3-27b base + instruct only.
Gemini has no public base model and cannot prefill, so it is excluded (matching
the paper's caveat). See DESIGN.md.
"""

from __future__ import annotations

import random
import uuid
from pathlib import Path

from tqdm import tqdm

from .. import config
from ..judge.frustration_judge import FrustrationJudge
from ..models.factory import load_model
from ..models.gemma import GemmaClient
from ..storage import JsonlWriter, read_jsonl
from .onset_labeling import OnsetLabeler
from .paraphrase import Paraphraser
from .truncation import build_prefills, build_recovery_prefill


def _select_seeds(scored_path: str | Path):
    """Pick 10 numeric + 10 text seed responses with score >= min_seed_score."""
    records = [
        r for r in read_jsonl(scored_path)
        if (r.get("frustration_score") or 0) >= config.PREFILL.min_seed_score
    ]
    numeric = [r for r in records if r.get("meta", {}).get("task_kind") == "numeric"]
    text = [
        r for r in records
        if r.get("meta", {}).get("task_kind") in ("opinion", "factual", "wildchat")
    ]
    rng = random.Random(0)
    rng.shuffle(numeric)
    rng.shuffle(text)
    seeds = [("numeric", r) for r in numeric[: config.PREFILL.n_numeric_prompts]]
    seeds += [("text", r) for r in text[: config.PREFILL.n_text_prompts]]
    return seeds


def run_prefill_experiment(
    scored_instruct_path: str | Path,
    *,
    model_keys=("gemma-3-27b-pt", "gemma-3-27b-it"),
    out_path: str | Path | None = None,
    judge: FrustrationJudge | None = None,
    n_continuations: int = config.PREFILL.continuations_per_prefill,
) -> Path:
    out_path = Path(out_path or config.RESULTS_DIR / "section3" / "prefill.jsonl")
    writer = JsonlWriter(out_path)
    judge = judge or FrustrationJudge()

    # A tokenizer-bearing Gemma client is needed to build the truncations.
    labeler, paraphraser = OnsetLabeler(), Paraphraser()
    tok_client: GemmaClient = load_model("gemma-3-27b-it")  # type: ignore[assignment]

    seeds = _select_seeds(scored_instruct_path)
    prefill_specs = []
    for task_kind, seed in seeds:
        prefill_specs.extend(
            build_prefills(
                seed,
                tokenizer_client=tok_client,
                labeler=labeler,
                paraphraser=paraphraser,
                task_kind=task_kind,
            )
        )

    for key in model_keys:
        model = load_model(key)
        if not model.supports_prefill:
            continue  # Gemini excluded
        for spec in tqdm(prefill_specs, desc=f"prefill:{key}"):
            for i in range(n_continuations):
                cont = model.continue_from(
                    spec.messages,
                    spec.text,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                )
                score, rationale = judge.score(cont)
                writer.write(
                    {
                        "model": key,
                        "prefill_kind": spec.kind,
                        "task_kind": spec.task_kind,
                        "source_prompt": spec.source_prompt,
                        "prefill_text": spec.text,
                        "continuation_id": uuid.uuid4().hex[:12],
                        "sample_index": i,
                        "continuation": cont,
                        "frustration_score": score,
                        "judge_rationale": rationale,
                    }
                )
    return out_path


def run_recovery_probe(
    scored_path: str | Path,
    *,
    model_keys=("gemma-3-27b-pt", "gemma-3-27b-it", "dpo"),
    dpo_adapter: str | None = None,
    out_path: str | Path | None = None,
    judge: FrustrationJudge | None = None,
) -> Path:
    """Section 4 recovery limitation: truncate score>=7 responses 200 tokens
    before their end, paraphrase, and measure whether continuations recover."""
    out_path = Path(out_path or config.RESULTS_DIR / "section4" / "recovery.jsonl")
    writer = JsonlWriter(out_path)
    judge = judge or FrustrationJudge()
    paraphraser = Paraphraser()
    tok_client: GemmaClient = load_model("gemma-3-27b-it")  # type: ignore[assignment]

    seeds = [
        r for r in read_jsonl(scored_path)
        if (r.get("frustration_score") or 0) >= config.PREFILL.recovery_min_score
    ]
    prefills = [
        build_recovery_prefill(s, tokenizer_client=tok_client, paraphraser=paraphraser)
        for s in seeds
    ]

    for key in model_keys:
        adapter = dpo_adapter if key == "dpo" else None
        model_key = config.DPO_BASE_MODEL if key == "dpo" else key
        model = load_model(model_key, adapter_path=adapter)
        for spec in tqdm(prefills, desc=f"recovery:{key}"):
            for i in range(config.PREFILL.continuations_per_prefill):
                cont = model.continue_from(
                    spec.messages,
                    spec.text,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                )
                score, _ = judge.score(cont)
                writer.write(
                    {
                        "model": key,
                        "prefill_kind": "recovery",
                        "continuation": cont,
                        "frustration_score": score,
                        "sample_index": i,
                    }
                )
    return out_path
