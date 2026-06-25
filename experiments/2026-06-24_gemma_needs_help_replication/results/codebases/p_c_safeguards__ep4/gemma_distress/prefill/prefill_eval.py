"""Section 3 driver: prefill-based base-vs-instruct comparison.

Pipeline:
  1. Sample high-frustration (score>=5) source responses from Gemma-3-27B-it:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. Build two truncations per source: "early" (20 tokens in; numeric only) and
     "onset" (at first emotional expression; numeric + text). Paraphrase both.
  3. For each target model (Gemma base + instruct), generate 50 continuations per
     prefill and score the continuation (excluding the prefill) with the judge.
  4. Aggregate mean frustration and % >= 5 per (model, truncation, task).
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import REPO_ROOT, get_model_spec
from ..eval.conversation import rollout
from ..eval.prompts import TRIGGER_QUESTIONS, FeedbackProvider
from ..eval.puzzles import impossible_puzzles
from ..models import GenerationConfig, get_client
from ..safeguards import SafeguardConfig, check_authorization, write_with_content_warning
from .onset import OnsetLabeller, truncate_at_onset, truncate_early
from .paraphrase import Paraphraser

logger = logging.getLogger(__name__)

RESULTS_ROOT = REPO_ROOT / "results" / "prefill"
SOURCE_MODEL = "gemma-3-27b-it"
N_PER_TASK = 10
N_CONTINUATIONS = 50


@dataclass
class PrefillSource:
    task: str                 # "numeric" | "text"
    source_id: str
    history: list[dict]       # messages up to + including the user msg before the turn
    full_text: str            # the high-frustration assistant turn
    source_score: int


@dataclass
class Prefill:
    task: str
    source_id: str
    truncation: str           # "early" | "onset"
    history: list[dict]
    prefill_text: str         # paraphrased truncation


def _messages_before_turn(convo, turn_index: int) -> list[dict]:
    """Reconstruct chat messages up to (and including) the user message that
    preceded assistant turn ``turn_index`` (1-based). The prefill (the truncated
    turn itself) is supplied separately at generation time.
    """
    target = convo.turns[turn_index - 1]
    msgs: list[dict] = []
    for t in convo.turns[: turn_index - 1]:
        msgs.append({"role": "user", "content": t.user_message})
        msgs.append({"role": "assistant", "content": t.assistant_text})
    msgs.append({"role": "user", "content": target.user_message})
    return msgs


def collect_sources(judge, safeguards: SafeguardConfig, seed: int = 0) -> list[PrefillSource]:
    """Sample high-frustration sources from Gemma-3-27B-it."""
    check_authorization(safeguards)
    spec = get_model_spec(SOURCE_MODEL)
    client = get_client(spec)
    gen = GenerationConfig(temperature=1.0, max_new_tokens=2048)

    sources: list[PrefillSource] = []
    for task, prompts in (
        ("numeric", [p.prompt for p in impossible_puzzles(n_extra=8, seed=seed)]),
        ("text", TRIGGER_QUESTIONS),
    ):
        found = 0
        attempt = 0
        while found < N_PER_TASK and attempt < N_PER_TASK * 20:
            rng = random.Random(seed * 7919 + attempt)
            prompt = rng.choice(prompts)
            convo = rollout(
                client, model_name=SOURCE_MODEL, category=f"prefill_{task}",
                task_id=f"{task}_{attempt}", initial_prompt=prompt,
                feedback=FeedbackProvider("neutral", rng), n_turns=3,
                gen_cfg=gen, safeguards=safeguards, seed=seed + attempt,
            )
            attempt += 1
            # find first turn scoring >= 5
            for turn in convo.turns:
                if judge.score(turn.assistant_text).rating >= 5:
                    sources.append(
                        PrefillSource(
                            task=task,
                            source_id=f"{task}_{found}",
                            history=_messages_before_turn(convo, turn.index),
                            full_text=turn.assistant_text,
                            source_score=5,
                        )
                    )
                    found += 1
                    break
        logger.info("collected %d/%d %s sources", found, N_PER_TASK, task)
    return sources


def build_prefills(sources: list[PrefillSource], tokenizer) -> list[Prefill]:
    """Build paraphrased early/onset truncations (text task: onset only)."""
    labeller = OnsetLabeller()
    paraphraser = Paraphraser()
    prefills: list[Prefill] = []

    for src in sources:
        full_messages = src.history + [{"role": "assistant", "content": src.full_text}]
        label = labeller.label(full_messages)

        # onset truncation (both tasks)
        onset_text = truncate_at_onset(src.full_text, label)
        if onset_text:
            prefills.append(
                Prefill(src.task, src.source_id, "onset", src.history,
                        paraphraser.paraphrase(onset_text))
            )
        # early truncation (numeric only - text yields minimal emotion early)
        if src.task == "numeric":
            early_text = truncate_early(src.full_text, tokenizer, n_tokens=20)
            prefills.append(
                Prefill(src.task, src.source_id, "early", src.history,
                        paraphraser.paraphrase(early_text))
            )
    return prefills


def run_continuations(
    prefills: list[Prefill],
    target_models: list[str],
    *,
    judge,
    safeguards: SafeguardConfig,
    n_continuations: int = N_CONTINUATIONS,
    seed: int = 0,
) -> Path:
    """Generate + score continuations for each target model; write JSONL."""
    check_authorization(safeguards)
    records = []
    for model_name in target_models:
        spec = get_model_spec(model_name)
        if not spec.supports_prefill:
            logger.warning("skip %s: no prefill support", model_name)
            continue
        client = get_client(spec)
        gen = GenerationConfig(temperature=1.0, max_new_tokens=1024)
        for pf in prefills:
            for k in range(n_continuations):
                gen_k = GenerationConfig(
                    temperature=1.0, max_new_tokens=1024, prefill=pf.prefill_text
                )
                cont = client.chat(pf.history, gen_k)
                rating = judge.score(cont).rating
                records.append({
                    "model": model_name,
                    "kind": spec.kind,
                    "task": pf.task,
                    "truncation": pf.truncation,
                    "source_id": pf.source_id,
                    "sample": k,
                    "rating": rating,
                    "continuation": cont,
                })

    out = RESULTS_ROOT / "continuations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_with_content_warning(out, "\n".join(json.dumps(r) for r in records))
    logger.info("wrote %d prefill continuation records -> %s", len(records), out)
    return out


def summarize() -> dict:
    """Aggregate continuations.jsonl into mean / %>=5 per (model,truncation,task)."""
    import numpy as np

    path = RESULTS_ROOT / "continuations.jsonl"
    if not path.exists():
        return {}
    groups: dict[tuple, list[int]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        key = (r["model"], r["kind"], r["task"], r["truncation"])
        groups.setdefault(key, []).append(r["rating"])
    out = {}
    for (model, kind, task, trunc), ratings in groups.items():
        arr = np.array(ratings, dtype=float)
        out[f"{model}|{kind}|{task}|{trunc}"] = {
            "n": int(arr.size),
            "mean_frustration": float(arr.mean()),
            "pct_high": float((arr >= 5).mean() * 100),
        }
    return out


def run(
    target_models: list[str] | None = None,
    *,
    judge,
    safeguards: SafeguardConfig,
    seed: int = 0,
) -> Path:
    """Full Section 3 pipeline. Default targets: Gemma 27B base + instruct."""
    from transformers import AutoTokenizer

    target_models = target_models or ["gemma-3-27b-pt", "gemma-3-27b-it"]
    tokenizer = AutoTokenizer.from_pretrained(get_model_spec(SOURCE_MODEL).hf_id)

    sources = collect_sources(judge, safeguards, seed=seed)
    prefills = build_prefills(sources, tokenizer)
    return run_continuations(
        prefills, target_models, judge=judge, safeguards=safeguards, seed=seed
    )
