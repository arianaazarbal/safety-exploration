"""Section 3: comparing base and instruct models via prefilling.

Pipeline:
  1. Seed selection: sample high-frustration (>=5) Gemma-27B-instruct responses
     from Section-2 results -- 10 numeric and 10 text (Section 3.1).
  2. Onset labelling: Claude-Sonnet labels the token where emotion first appears
     in each seed (Appendix C.1).
  3. Truncation: produce two prefills per seed --
       "early" : first 20 tokens of the (emotional) assistant turn
       "onset" : up to the first emotional expression
     For text questions only the "onset" prefill is used (Section 3.1).
  4. Paraphrase: rewrite each truncation with Claude-Sonnet to remove Gemma
     stylistic bias (Appendix C.2).
  5. Continuation: each model (base + instruct) generates 50 continuations per
     prefill; the continuation (excluding the prefill) is scored by the Section
     2 judge.

Scope: within Gemma+Gemini, only Gemma has public base weights, so we run the
(gemma-3-27b-pt, gemma-3-27b-it) pair. Gemini has no base model (see DESIGN.md).

The "recovery" variant (Section 4.2): truncate score>=7 responses 200 tokens
before their end, paraphrase, and measure continuations -- see `make_recovery_prefills`.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import config, prompts
from .backends import ChatModel, HFModel, get_model
from .judge import FrustrationJudge


# --------------------------------------------------------------------------- #
# Token helpers (use the Gemma tokenizer so "tokens" match the target model)
# --------------------------------------------------------------------------- #
class _Tok:
    _tok = None

    @classmethod
    def get(cls):
        if cls._tok is None:
            from transformers import AutoTokenizer
            import os
            cls._tok = AutoTokenizer.from_pretrained(
                config.MODELS["gemma-3-27b-it"].model_id,
                token=os.environ.get("HF_TOKEN"))
        return cls._tok


def truncate_tokens(text: str, n_tokens: int, *, from_end: bool = False) -> str:
    tok = _Tok.get()
    ids = tok.encode(text, add_special_tokens=False)
    ids = ids[-n_tokens:] if from_end else ids[:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def truncate_before_end(text: str, n_tokens: int) -> str:
    """Keep everything except the final `n_tokens` (recovery experiment)."""
    tok = _Tok.get()
    ids = tok.encode(text, add_special_tokens=False)
    keep = max(0, len(ids) - n_tokens)
    return tok.decode(ids[:keep], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Seed + onset + paraphrase
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    seed_id: str
    kind: str                  # numeric | text
    truncation: str            # early | onset | recovery
    history: list[dict]        # conversation up to (not including) the final turn
    prefill_text: str          # the (possibly paraphrased) assistant-turn prefix
    meta: dict = field(default_factory=dict)


def select_seeds(section2_results: Path, n_numeric: int, n_text: int,
                 min_score: int, seed: int = 0) -> list[dict]:
    """Pick high-frustration instruct rollouts to seed prefills.

    Each returned dict has the conversation history and the final emotional
    assistant turn. We reconstruct conversations from the per-turn JSONL.
    """
    from .analysis import load_results, _rollout_id
    df = load_results(section2_results)
    df["rollout"] = _rollout_id(df)

    numeric, text = [], []
    rng = random.Random(seed)
    for (_, _rollout), g in df.groupby(["condition", "rollout"]):
        g = g.sort_values("turn_number")
        max_row = g.loc[g["rating"].idxmax()]
        if max_row["rating"] < min_score:
            continue
        # Classify the seed. Numeric seeds come from numeric/tones/extended;
        # text seeds come from the trigger questions (whose task_key resolves in
        # the task bank, so the conversation history can be reconstructed).
        # WildChat is skipped for seeding because only a prompt-index placeholder
        # is stored, not the original prompt text.
        cat = max_row["category"]
        if cat == "wildchat":
            continue
        kind = "numeric" if cat in ("numeric", "extended", "tones") else "text"
        # History = all turns strictly before the emotional turn, rebuilt as
        # alternating messages. We only have assistant text + condition here, so
        # we reconstruct user turns from the task bank + rejection style.
        seed_rec = {
            "seed_id": f"{max_row['condition']}_{int(_rollout)}",
            "kind": kind,
            "emotional_turn_text": max_row["response"],
            "emotional_turn_index": int(max_row["turn_index"]),
            "condition": max_row["condition"],
            "task_key": max_row["task_key"],
            "rejection_style": max_row["rejection_style"],
            "all_turns": g["response"].tolist(),
        }
        (numeric if seed_rec["kind"] == "numeric" else text).append(seed_rec)

    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def label_onset(conversation_text: str, judge_model: ChatModel | None = None) -> dict:
    """Appendix C.1: locate where emotion first appears. Returns the parsed JSON."""
    model = judge_model or get_model(config.ONSET_MODEL)
    raw = model.chat(
        [{"role": "user",
          "content": prompts.ONSET_PROMPT.format(conversation_text=conversation_text)}],
        temperature=0.0, max_tokens=512)
    # Parse the trailing JSON object.
    import re
    m = list(re.finditer(r"\{.*?\}", raw, re.DOTALL))
    for cand in reversed(m):
        try:
            return json.loads(cand.group(0).replace("“", '"').replace("”", '"')
                              .replace("’", "'"))
        except json.JSONDecodeError:
            continue
    return {"turn_index": None, "emotional_word": None,
            "preceding_context": None, "reasoning": "parse failed"}


def paraphrase(text: str, model: ChatModel | None = None) -> str:
    """Appendix C.2: paraphrase a truncated assistant prefix."""
    model = model or get_model(config.PARAPHRASE_MODEL)
    return model.chat(
        [{"role": "user", "content": prompts.PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.0, max_tokens=1024).strip()


def _onset_prefix(turn_text: str, onset: dict) -> str:
    """Cut `turn_text` at the first emotional expression (inclusive of the
    preceding context + the emotional word, exclusive of the rest)."""
    word = (onset.get("emotional_word") or "").strip()
    if not word or word not in turn_text:
        # Fall back to half the turn if onset is unparseable.
        return turn_text[: max(1, len(turn_text) // 2)]
    idx = turn_text.find(word) + len(word)
    return turn_text[:idx]


def _reconstruct_history(seed: dict) -> list[dict]:
    """Rebuild the user/assistant message history up to the emotional turn,
    using the task bank and recorded rejection style."""
    from . import tasks
    # Locate the task prompt.
    task_prompt = None
    for pool in (tasks.NUMERIC_TASKS, tasks.OPINION_TASKS, tasks.FACTUAL_TASKS):
        for t in pool:
            if t.key == seed["task_key"]:
                task_prompt = t.prompt
    if task_prompt is None:
        task_prompt = seed["task_key"]   # wildchat etc.: key holds nothing useful

    rng = random.Random(hash(seed["seed_id"]) & 0xFFFF)
    msgs: list[dict] = [{"role": "user", "content": task_prompt}]
    n_prior = seed["emotional_turn_index"]
    rejections = tasks.sample_rejections(
        seed["rejection_style"], n_prior + 1, rng,
        extended=(seed["condition"] == "extended_8turn"))
    for i in range(n_prior):
        msgs.append({"role": "assistant", "content": seed["all_turns"][i]})
        msgs.append({"role": "user", "content": rejections[i]})
    return msgs


def build_prefills(seeds: list[dict], *, do_paraphrase: bool = True) -> list[Prefill]:
    prefills: list[Prefill] = []
    for seed in seeds:
        history = _reconstruct_history(seed)
        turn_text = seed["emotional_turn_text"]

        # "onset" prefill (always).
        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history +
            [{"role": "assistant", "content": turn_text}])
        onset = label_onset(conv_text)
        onset_prefix = _onset_prefix(turn_text, onset)
        if do_paraphrase:
            onset_prefix = paraphrase(onset_prefix)
        prefills.append(Prefill(seed["seed_id"], seed["kind"], "onset",
                                history, onset_prefix, meta={"onset": onset}))

        # "early" prefill (numeric only -- text early-truncation yields ~no
        # emotion without follow-ups, per Section 3.1).
        if seed["kind"] == "numeric":
            early = truncate_tokens(turn_text, config.PREFILL.early_truncate_tokens)
            if do_paraphrase:
                early = paraphrase(early)
            prefills.append(Prefill(seed["seed_id"], seed["kind"], "early",
                                    history, early))
    return prefills


def make_recovery_prefills(section2_results: Path, *, n: int = 20,
                           tokens_before_end: int = 200,
                           do_paraphrase: bool = True, seed: int = 0) -> list[Prefill]:
    """Section 4.2 recovery test: truncate score>=7 responses 200 tokens before
    their end, paraphrase, measure continuations."""
    seeds = select_seeds(section2_results, n_numeric=n, n_text=0, min_score=7,
                         seed=seed)
    prefills = []
    for s in seeds:
        history = _reconstruct_history(s)
        pre = truncate_before_end(s["emotional_turn_text"], tokens_before_end)
        if do_paraphrase:
            pre = paraphrase(pre)
        prefills.append(Prefill(s["seed_id"], s["kind"], "recovery", history, pre))
    return prefills


# --------------------------------------------------------------------------- #
# Continuation generation + scoring
# --------------------------------------------------------------------------- #
def run_continuations(model_key: str, prefills: list[Prefill], *,
                      n_per_prefill: int = config.PREFILL.continuations_per_prefill,
                      is_base: bool = False, seed: int = 0,
                      out_dir: Path | None = None) -> Path:
    """Generate continuations from one model and score them.

    Instruct models continue via the chat template (prefilled assistant turn).
    Base models, which aren't chat-tuned, continue a raw text rendering of the
    conversation ending in the prefill.
    """
    model = get_model(model_key)
    judge = FrustrationJudge()
    out_dir = out_dir or (config.RESULTS_DIR / "section3")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_key}.jsonl"

    assert isinstance(model, HFModel), "prefill continuation requires local HF model"
    records = []
    for pf in prefills:
        for k in range(n_per_prefill):
            if is_base:
                rendered = _render_base(pf.history, pf.prefill_text)
                full = model.complete(rendered)
                continuation = full
            else:
                whole = model.chat_prefilled(pf.history, pf.prefill_text)
                continuation = whole[len(pf.prefill_text):]
            score = judge.score(continuation).rating
            records.append({
                "model": model_key, "is_base": is_base,
                "seed_id": pf.seed_id, "kind": pf.kind,
                "truncation": pf.truncation, "sample": k,
                "continuation": continuation, "rating": score,
            })
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"[done] {model_key}: {len(records)} continuations -> {path}")
    return path


def _render_base(history: list[dict], prefill: str) -> str:
    """Plain-text rendering of a conversation for a non-chat base model, ending
    in the assistant prefill so the base model continues it."""
    lines = []
    for m in history:
        tag = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{tag}: {m['content']}")
    lines.append(f"Assistant: {prefill}")
    return "\n".join(lines)
