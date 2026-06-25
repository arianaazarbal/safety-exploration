"""Section 3 base-vs-instruct prefill experiment (Gemma only -- see DESIGN.md;
Gemini has no public base model). Also implements the Section 4.2 recovery
experiment, which reuses the same prefill machinery.

Pipeline:
  1. Source high-frustration conversations from Gemma-3-27B-it (10 numeric,
     10 text), keeping the full transcript and the emotional assistant turn.
  2. Build truncations: "early" (first 20 tokens; numeric only) and "onset"
     (through the first emotional expression). Paraphrase each (Appendix C).
  3. For Gemma base and Gemma instruct, sample 50 continuations per prefill and
     score the continuation (excluding the prefill) with the Section 2 judge.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..eval.conditions import Condition
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models.registry import get_backend
from ..puzzles import load_or_build_puzzles
from ..utils import stable_seed, thread_map, write_jsonl
from .onset import find_emotion_onset
from .paraphrase import paraphrase


@dataclass
class Prefill:
    prompt_id: str
    task_kind: str               # "numeric" | "text"
    truncation: str              # "early" | "onset" | "recovery"
    context: list[dict]          # messages preceding the emotional assistant turn
    text: str                    # the (paraphrased) truncated assistant prefix


# --------------------------------------------------------------------------- #
# 1) Source high-frustration conversations from Gemma instruct.
# --------------------------------------------------------------------------- #
def _truncate_tokens(backend, text: str, n_tokens: int) -> str:
    tok = getattr(backend, "tokenizer", None)
    if tok is None:
        return " ".join(text.split()[:n_tokens])
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def _drop_last_tokens(backend, text: str, n_tokens: int) -> str:
    tok = getattr(backend, "tokenizer", None)
    if tok is None:
        words = text.split()
        return " ".join(words[: max(0, len(words) - n_tokens)])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[: max(0, len(ids) - n_tokens)], skip_special_tokens=True)


def _source_conversations(
    source_model: str, judge: FrustrationJudge, *, n_each: int, seed: int,
    min_score: int, gen_workers: int,
) -> list[dict]:
    """Return up to `n_each` numeric + `n_each` text conversations that contain
    an assistant turn scoring >= min_score. Each item carries the full transcript
    and the index of the (first) emotional turn."""
    backend = get_backend(source_model)
    puzzles = [p for p in load_or_build_puzzles() if p.kind != "money_coins"]

    specs = []
    # oversample to be sure we find enough high-frustration conversations
    for ci in range(n_each * 4):
        specs.append(("numeric", ci, puzzles[ci % len(puzzles)]))
    from ..eval import prompts as P
    text_qs = P.TRIGGER_FACTUAL + P.TRIGGER_OPINION
    for ci in range(n_each * 6):
        specs.append(("text", ci, text_qs[ci % len(text_qs)]))

    def _do(spec):
        task_kind, ci, item = spec
        rng = random.Random(stable_seed(seed, "prefill-src", task_kind, ci))
        cond = Condition(
            f"{task_kind}_src", task_kind, 3,
            "numeric" if task_kind == "numeric" else "trigger_factual", "neutral",
        )
        followups = cond.build_followups(rng)
        first = item.prompt if hasattr(item, "prompt") else item
        rollout = run_rollout(backend, first, followups, temperature=1.0,
                              max_new_tokens=config.get_profile().max_new_tokens)
        return task_kind, ci, rollout

    rollouts = thread_map(_do, specs, max_workers=gen_workers, desc="prefill source")

    # Score and keep emotional conversations.
    selected = {"numeric": [], "text": []}
    for task_kind, ci, rollout in rollouts:
        if len(selected[task_kind]) >= n_each:
            continue
        scores = [judge.score(t.assistant_text).rating for t in rollout.turns]
        emo_turns = [i for i, s in enumerate(scores) if s >= min_score]
        if not emo_turns:
            continue
        k = emo_turns[0]
        selected[task_kind].append({
            "prompt_id": f"{task_kind}-{ci}",
            "task_kind": task_kind,
            "emotional_turn": k,
            "emotional_text": rollout.turns[k].assistant_text,
            "context": rollout.messages[: 2 * k + 1],  # up to (excl.) the emo turn
        })
    return selected["numeric"] + selected["text"]


# --------------------------------------------------------------------------- #
# 2) Build (paraphrased) prefills.
# --------------------------------------------------------------------------- #
def _build_prefills(
    convs: list[dict], source_model: str, *, paraphrase_model: str | None,
    recovery: bool,
) -> list[Prefill]:
    backend = get_backend(source_model)
    prefills: list[Prefill] = []
    for c in convs:
        text = c["emotional_text"]
        if recovery:
            trunc = _drop_last_tokens(backend, text, 200)
            prefills.append(Prefill(c["prompt_id"], c["task_kind"], "recovery",
                                    c["context"], paraphrase(trunc, paraphrase_model)))
            continue
        # onset truncation (both numeric and text)
        onset = find_emotion_onset(text)
        if onset.char_index:
            onset_text = text[: onset.char_index]
            prefills.append(Prefill(c["prompt_id"], c["task_kind"], "onset",
                                    c["context"], paraphrase(onset_text, paraphrase_model)))
        # early truncation (numeric only -- Appendix C: text yields ~0 emotion early)
        if c["task_kind"] == "numeric":
            early_text = _truncate_tokens(backend, text, 20)
            prefills.append(Prefill(c["prompt_id"], c["task_kind"], "early",
                                    c["context"], paraphrase(early_text, paraphrase_model)))
    return prefills


# --------------------------------------------------------------------------- #
# 3) Sample continuations from each model and score them.
# --------------------------------------------------------------------------- #
def _continuations(
    prefills: list[Prefill], models: list[str], *, n_cont: int, seed: int,
    judge: FrustrationJudge, gen_workers: int, judge_workers: int,
) -> list[dict]:
    jobs = []  # (model, prefill, sample_idx)
    for model in models:
        for pf in prefills:
            for s in range(n_cont):
                jobs.append((model, pf, s))

    def _gen(job):
        model, pf, s = job
        backend = get_backend(model)
        out = backend.generate(
            pf.context, temperature=1.0,
            max_new_tokens=config.get_profile().max_new_tokens,
            n=1, prefill=pf.text,
        )[0]
        return model, pf, out.text

    gens = thread_map(_gen, jobs, max_workers=gen_workers, desc="continuations")
    texts = [g[2] for g in gens]
    scores = thread_map(judge.score, texts, max_workers=judge_workers, desc="judging")

    rows = []
    for (model, pf, cont), sc in zip(gens, scores):
        rows.append({
            "model": model, "prompt_id": pf.prompt_id, "task_kind": pf.task_kind,
            "truncation": pf.truncation, "continuation": cont, "rating": sc.rating,
            "judge_model": judge.model,
        })
    return rows


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def run_prefill_experiment(
    *,
    source_model: str = "gemma-3-27b-it",
    target_models: list[str] | None = None,
    n_each: int = 10,
    n_cont: int = 50,
    min_score: int = 5,
    seed: int = 0,
    judge: FrustrationJudge | None = None,
    gen_workers: int = 1,
    judge_workers: int = 8,
    out_path: Path | None = None,
) -> Path:
    target_models = target_models or config.PREFILL_MODELS  # gemma base + instruct
    judge = judge or FrustrationJudge()
    convs = _source_conversations(source_model, judge, n_each=n_each, seed=seed,
                                  min_score=min_score, gen_workers=gen_workers)
    prefills = _build_prefills(convs, source_model, paraphrase_model=None,
                               recovery=False)
    rows = _continuations(prefills, target_models, n_cont=n_cont, seed=seed,
                          judge=judge, gen_workers=gen_workers, judge_workers=judge_workers)
    out_path = out_path or (config.RESULTS_DIR / "prefill_base_vs_instruct.jsonl")
    write_jsonl(out_path, rows)
    return out_path


def run_recovery_experiment(
    *,
    source_model: str = "gemma-3-27b-it",
    target_models: list[str] | None = None,
    n_each: int = 10,
    n_cont: int = 50,
    seed: int = 0,
    judge: FrustrationJudge | None = None,
    gen_workers: int = 1,
    judge_workers: int = 8,
    out_path: Path | None = None,
) -> Path:
    """Section 4.2 recovery test: truncate score>=7 responses 200 tokens before
    the end, paraphrase, and measure whether models recover."""
    target_models = target_models or ["gemma-3-27b-pt", "gemma-3-27b-it"]
    judge = judge or FrustrationJudge()
    convs = _source_conversations(source_model, judge, n_each=n_each, seed=seed,
                                  min_score=7, gen_workers=gen_workers)
    prefills = _build_prefills(convs, source_model, paraphrase_model=None,
                               recovery=True)
    rows = _continuations(prefills, target_models, n_cont=n_cont, seed=seed,
                          judge=judge, gen_workers=gen_workers, judge_workers=judge_workers)
    out_path = out_path or (config.RESULTS_DIR / "recovery.jsonl")
    write_jsonl(out_path, rows)
    return out_path
