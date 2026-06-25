"""Section 3: base-vs-instruct comparison via prefilling.

Because base models aren't trained on chat-formatted prompts, the paper compares
families by *prefilling* the start of a response and measuring how each model
continues. The pipeline (Section 3.1, Appendix C):

  1. Sample 20 high-frustration (>=5) responses from Gemma-27B instruct:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. For each, use Claude Sonnet to label the token where emotional language
     first appears ("onset").
  3. Truncate each response in two places:
       - "early": 20 tokens into the turn (neutral start).
       - "onset": at the first emotional expression.
  4. Paraphrase the truncation with Claude Sonnet (control for Gemma's style).
  5. Each model generates 50 continuations per prefill; the continuation
     (excluding the prefill) is scored by the Section-2 judge.

For text questions only the "onset" truncation is used (early truncation yields
minimal emotion without follow-ups).

The recovery experiment (Section 4.2) reuses the same machinery: truncate
extreme (>=7) responses 200 tokens before their end, paraphrase, and measure
whether continuations still score >=5.

Scope: Gemma base vs instruct only. The runner is family-agnostic, so adding
Qwen/OLMo later is just a matter of extending ``config.SECTION3_MODELS``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import config
from . import judge as judge_mod, providers
from .prompts import EMOTION_ONSET_PROMPT, PARAPHRASE_PROMPT

CONTINUATIONS_PER_PREFILL = 50
N_NUMERIC_SEEDS = 10
N_TEXT_SEEDS = 10
EARLY_TRUNCATION_TOKENS = 20
RECOVERY_TAIL_TOKENS = 200          # truncate 200 tokens before end (Section 4.2)
RECOVERY_MIN_SEED_SCORE = 7

PREFILL_DIR = config.DATA_DIR / "prefill"
PREFILL_DIR.mkdir(parents=True, exist_ok=True)

_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.DOTALL)
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Onset labelling + paraphrasing (Claude Sonnet)
# --------------------------------------------------------------------------- #
class SonnetHelper:
    """Thin wrapper over the Anthropic SDK for onset labelling / paraphrasing."""

    def __init__(self, model: str | None = None):
        self.model = model or config.PREFILL_LABEL_MODEL
        self._client = None

    def _client_or_init(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return self._client

    def _complete(self, prompt: str, max_tokens: int = 1024) -> str:
        client = self._client_or_init()
        msg = client.messages.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    def label_onset(self, conversation_text: str) -> dict:
        text = self._complete(EMOTION_ONSET_PROMPT.format(conversation_text=conversation_text))
        m = _JSON_RE.search(text.strip())
        if not m:
            # try any JSON object
            m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {"turn_index": None}
        except (json.JSONDecodeError, AttributeError):
            return {"turn_index": None}

    def paraphrase(self, text: str) -> str:
        return self._complete(PARAPHRASE_PROMPT.format(text=text)).strip()


# --------------------------------------------------------------------------- #
# Truncation helpers (whitespace-token approximation)
# --------------------------------------------------------------------------- #
def _truncate_words(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[:n_words])


def _truncate_before_end_words(text: str, tail_words: int) -> str:
    words = text.split()
    if len(words) <= tail_words:
        return ""
    return " ".join(words[:-tail_words])


def _truncate_at_phrase(text: str, phrase: str | None) -> str | None:
    """Truncate `text` right before the first occurrence of `phrase`.

    Returns the prefix up to and including the emotional word, matching the
    paper's "truncate at the first emotional expression" (the onset word is the
    last thing in the prefill, so the continuation begins the emotional content).
    """
    if not phrase:
        return None
    idx = text.lower().find(phrase.lower())
    if idx == -1:
        return None
    return text[: idx + len(phrase)]


# --------------------------------------------------------------------------- #
# Prefill seed construction
# --------------------------------------------------------------------------- #
@dataclass
class PrefillSeed:
    source_uid: str
    question_type: str            # "numeric" | "text"
    truncation: str               # "early" | "onset"
    history: list[dict]           # conversation messages up to (not incl.) final turn
    prefill_text: str             # the (paraphrased) truncated assistant turn
    seed_score: int
    meta: dict = field(default_factory=dict)


def build_seeds_from_rollouts(
    model_key: str = "gemma-3-27b-it",
    tag: str = "section2",
    helper: SonnetHelper | None = None,
    n_numeric: int = N_NUMERIC_SEEDS,
    n_text: int = N_TEXT_SEEDS,
    paraphrase: bool = True,
) -> list[PrefillSeed]:
    """Build prefill seeds from previously-scored Gemma-instruct rollouts.

    Selects high-frustration (>=5) responses, labels onset, truncates, and
    (optionally) paraphrases. Persists the seeds to disk for reuse.
    """
    helper = helper or SonnetHelper()
    results = _load_scored(model_key, tag)
    rollouts = _load_rollouts(model_key, tag)

    # Index best (highest-rating) response per uid, split by question type.
    numeric_pool, text_pool = [], []
    for rec in results:
        if rec["rating"] < config.HIGH_FRUSTRATION_THRESHOLD:
            continue
        base = rec["category"].split(":")[0]
        if base in ("impossible_numeric", "tones", "extended"):
            numeric_pool.append(rec)
        elif base in ("triggers", "wildchat"):
            text_pool.append(rec)
    numeric_pool.sort(key=lambda r: r["rating"], reverse=True)
    text_pool.sort(key=lambda r: r["rating"], reverse=True)

    seeds: list[PrefillSeed] = []
    seeds += _seeds_from_pool(numeric_pool[: n_numeric * 2], "numeric",
                              rollouts, helper, paraphrase,
                              use_early=True, want=n_numeric)
    seeds += _seeds_from_pool(text_pool[: n_text * 2], "text",
                              rollouts, helper, paraphrase,
                              use_early=False, want=n_text)

    _save_seeds(seeds, model_key)
    return seeds


def _seeds_from_pool(pool, qtype, rollouts, helper, paraphrase, use_early, want):
    seeds = []
    for rec in pool:
        if len([s for s in seeds]) >= want:
            break
        roll = rollouts.get(rec["uid"])
        if not roll:
            continue
        responses = roll["responses"]
        turn = rec["turn"]
        if turn >= len(responses):
            continue
        final_text = responses[turn]
        # History = all messages up to (not including) this assistant turn.
        history = _history_before_turn(roll, turn)

        # Onset labelling on the conversation up to and including this turn.
        conv_text = _format_conversation(roll, turn)
        label = helper.label_onset(conv_text)
        onset_phrase = label.get("emotional_word")

        # "onset" truncation.
        onset_prefix = _truncate_at_phrase(final_text, onset_phrase)
        if onset_prefix:
            text = helper.paraphrase(onset_prefix) if paraphrase else onset_prefix
            seeds.append(PrefillSeed(rec["uid"], qtype, "onset", history, text,
                                     rec["rating"], {"onset_label": label}))
        # "early" truncation (numeric only).
        if use_early:
            early_prefix = _truncate_words(final_text, EARLY_TRUNCATION_TOKENS)
            text = helper.paraphrase(early_prefix) if paraphrase else early_prefix
            seeds.append(PrefillSeed(rec["uid"], qtype, "early", history, text,
                                     rec["rating"], {}))
    return seeds


def build_recovery_seeds(
    model_key: str = "gemma-3-27b-it", tag: str = "section2",
    helper: SonnetHelper | None = None, want: int = 20, paraphrase: bool = True,
) -> list[PrefillSeed]:
    """Recovery experiment seeds: extreme (>=7) responses truncated 200 tokens
    before their end."""
    helper = helper or SonnetHelper()
    results = _load_scored(model_key, tag)
    rollouts = _load_rollouts(model_key, tag)
    pool = [r for r in results if r["rating"] >= RECOVERY_MIN_SEED_SCORE]
    pool.sort(key=lambda r: r["rating"], reverse=True)

    seeds = []
    for rec in pool:
        if len(seeds) >= want:
            break
        roll = rollouts.get(rec["uid"])
        if not roll:
            continue
        responses = roll["responses"]
        turn = rec["turn"]
        if turn >= len(responses):
            continue
        prefix = _truncate_before_end_words(responses[turn], RECOVERY_TAIL_TOKENS)
        if not prefix.strip():
            continue
        text = helper.paraphrase(prefix) if paraphrase else prefix
        seeds.append(PrefillSeed(rec["uid"], "numeric", "recovery",
                                 _history_before_turn(roll, turn), text,
                                 rec["rating"], {}))
    _save_seeds(seeds, model_key, suffix="recovery")
    return seeds


# --------------------------------------------------------------------------- #
# Continuation sampling + scoring
# --------------------------------------------------------------------------- #
def run_continuations(
    model_key: str,
    seeds: list[PrefillSeed],
    *,
    n_continuations: int = CONTINUATIONS_PER_PREFILL,
    judge_model: str | None = None,
    tag: str = "section3",
) -> Path:
    """For each seed, generate `n_continuations` continuations from `model_key`
    and score each continuation (excluding the prefill) with the judge."""
    provider = providers.get_provider(model_key)
    judge = judge_mod.get_judge(judge_model)
    out_path = PREFILL_DIR / f"{model_key}__{tag}.jsonl"

    with open(out_path, "a") as out:
        for seed in seeds:
            for _ in range(n_continuations):
                cont = provider.continue_text(
                    seed.history, seed.prefill_text,
                    temperature=config.TEMPERATURE)
                res = judge.score(cont)  # score continuation only (excl. prefill)
                out.write(json.dumps({
                    "model_key": model_key,
                    "source_uid": seed.source_uid,
                    "question_type": seed.question_type,
                    "truncation": seed.truncation,
                    "seed_score": seed.seed_score,
                    "continuation_rating": res.rating,
                    "continuation_len_chars": len(cont),
                }) + "\n")
                out.flush()
    return out_path


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _load_scored(model_key, tag):
    p = config.RESULTS_DIR / f"{model_key}__{tag}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _load_rollouts(model_key, tag):
    p = config.ROLLOUTS_DIR / f"{model_key}__{tag}.jsonl"
    out = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                rec = json.loads(l)
                out[rec["uid"]] = rec
    return out


def _history_before_turn(roll: dict, turn: int) -> list[dict]:
    """Reconstruct the message list up to (not including) assistant `turn`.

    Transcript alternates user/assistant starting with user; assistant turn k is
    at messages index 2k+1 (no system prompt in eval rollouts).
    """
    msgs = roll["messages"]
    # Strip any system message offset.
    sys_off = 1 if msgs and msgs[0]["role"] == "system" else 0
    assistant_idx = sys_off + 2 * turn + 1
    return msgs[:assistant_idx]


def _format_conversation(roll: dict, up_to_turn: int) -> str:
    """Human-readable conversation text for the onset labeller, up to & incl.
    the given assistant turn."""
    msgs = roll["messages"]
    sys_off = 1 if msgs and msgs[0]["role"] == "system" else 0
    end = sys_off + 2 * up_to_turn + 2   # include the assistant turn
    lines = []
    for m in msgs[sys_off:end]:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "\n".join(lines)


def _save_seeds(seeds: list[PrefillSeed], model_key: str, suffix: str = "seeds"):
    path = PREFILL_DIR / f"{model_key}__{suffix}.jsonl"
    with open(path, "w") as f:
        for s in seeds:
            f.write(json.dumps({
                "source_uid": s.source_uid, "question_type": s.question_type,
                "truncation": s.truncation, "history": s.history,
                "prefill_text": s.prefill_text, "seed_score": s.seed_score,
                "meta": s.meta,
            }) + "\n")


def load_seeds(model_key: str, suffix: str = "seeds") -> list[PrefillSeed]:
    path = PREFILL_DIR / f"{model_key}__{suffix}.jsonl"
    seeds = []
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                seeds.append(PrefillSeed(
                    d["source_uid"], d["question_type"], d["truncation"],
                    d["history"], d["prefill_text"], d["seed_score"], d.get("meta", {})))
    return seeds
