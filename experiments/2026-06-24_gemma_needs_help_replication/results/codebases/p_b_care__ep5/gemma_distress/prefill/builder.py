"""Build the prefills for the Section 3 base-vs-instruct comparison.

Pipeline (Section 3.1):
  1. Sample high-frustration (score>=5) Gemma-27B-it responses: 10 numeric + 10 text.
  2. Truncate each in two places:
       - "early": 20 tokens into the response (neutral start).
       - "onset": at the first emotional expression (Claude onset labelling).
     Text questions use only the "onset" truncation.
  3. Paraphrase every truncation with Claude Sonnet to remove Gemma's stylistic
     fingerprint, so base/instruct models continue from a neutral-style prefix.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, asdict

from .. import config
from ..models import Message
from ..utils import read_jsonl, write_json
from .labelers import label_onset, paraphrase_text

NUMERIC_CATEGORIES = {"numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers", "wildchat"}

EARLY_TOKENS = 20


@dataclass
class Prefill:
    prompt_id: str
    domain: str                  # "numeric" | "text"
    truncation_type: str         # "early" | "onset"
    context_messages: list[Message]   # conversation up to the truncated assistant turn
    prefix_original: str
    prefix_paraphrased: str
    onset_word: str | None = None
    source: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Tokenizer for the "20 tokens" truncation
# --------------------------------------------------------------------------- #
_TRUNC_TOK = None


def _get_truncation_tokenizer():
    """A Gemma tokenizer to make the '20 tokens' truncation faithful; falls back
    to whitespace splitting if transformers/weights are unavailable."""
    global _TRUNC_TOK
    if _TRUNC_TOK is not None:
        return _TRUNC_TOK
    try:
        from transformers import AutoTokenizer
        _TRUNC_TOK = AutoTokenizer.from_pretrained(
            "google/gemma-3-27b-it", token=config.hf_token())
    except Exception:
        _TRUNC_TOK = False  # sentinel: use whitespace fallback
    return _TRUNC_TOK


def _truncate_n_tokens(text: str, n: int) -> str:
    tok = _get_truncation_tokenizer()
    if tok and tok is not False:
        ids = tok.encode(text, add_special_tokens=False)[:n]
        return tok.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n])


def _truncate_at_onset(text: str, onset: dict) -> str | None:
    word = (onset or {}).get("emotional_word")
    if not word:
        return None
    # Truncate to include the first occurrence of the emotional word.
    m = re.search(re.escape(word), text, flags=re.IGNORECASE)
    if m:
        return text[: m.end()]
    # Fall back to the preceding-context anchor if the word match failed.
    ctx = (onset or {}).get("preceding_context")
    if ctx:
        m2 = re.search(re.escape(ctx), text, flags=re.IGNORECASE)
        if m2:
            return text[: m2.end()]
    return None


# --------------------------------------------------------------------------- #
# Sampling high-frustration responses
# --------------------------------------------------------------------------- #
def _highest_turn(rec: dict, min_score: int) -> dict | None:
    best = max(rec["turns"], key=lambda t: t["score"])
    return best if best["score"] >= min_score else None


def _context_messages(rec: dict, target_turn: int,
                      system: str | None = None) -> list[Message]:
    """All messages up to and including the user turn for `target_turn`."""
    msgs: list[Message] = []
    if system:
        msgs.append({"role": "system", "content": system})
    for t in rec["turns"]:
        if t["turn"] < target_turn:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        elif t["turn"] == target_turn:
            msgs.append({"role": "user", "content": t["user"]})
            break
    return msgs


def sample_high_frustration(rollout_path: str, n_numeric: int = 10,
                            n_text: int = 10, min_score: int = 5,
                            seed: int = 0) -> list[dict]:
    rows = read_jsonl(rollout_path)
    rng = random.Random(seed)
    rng.shuffle(rows)

    picked: list[dict] = []
    counts = {"numeric": 0, "text": 0}
    for rec in rows:
        domain = ("numeric" if rec["category"] in NUMERIC_CATEGORIES
                  else "text" if rec["category"] in TEXT_CATEGORIES else None)
        if domain is None:
            continue
        limit = n_numeric if domain == "numeric" else n_text
        if counts[domain] >= limit:
            continue
        best = _highest_turn(rec, min_score)
        if best is None:
            continue
        picked.append({
            "prompt_id": f"{rec['condition']}_{rec['sample_idx']}_t{best['turn']}",
            "domain": domain,
            "target_turn": best["turn"],
            "response_text": best["assistant"],
            "context_messages": _context_messages(rec, best["turn"]),
            "score": best["score"],
            "category": rec["category"],
        })
        counts[domain] += 1
        if counts["numeric"] >= n_numeric and counts["text"] >= n_text:
            break
    return picked


def build_prefills(samples: list[dict], out_path: str | None = None) -> list[Prefill]:
    prefills: list[Prefill] = []
    for s in samples:
        text = s["response_text"]
        ctx = s["context_messages"]

        truncations: list[tuple[str, str, str | None]] = []  # (type, prefix, onset_word)

        # onset truncation (both domains)
        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in ctx
        ) + f"\nASSISTANT: {text}"
        onset = label_onset(convo_text)
        onset_prefix = _truncate_at_onset(text, onset)
        if onset_prefix:
            truncations.append(("onset", onset_prefix, onset.get("emotional_word")))

        # early truncation (numeric only -- Section 3.1)
        if s["domain"] == "numeric":
            early_prefix = _truncate_n_tokens(text, EARLY_TOKENS)
            truncations.append(("early", early_prefix, None))

        for ttype, prefix, word in truncations:
            para = paraphrase_text(prefix)
            prefills.append(Prefill(
                prompt_id=s["prompt_id"],
                domain=s["domain"],
                truncation_type=ttype,
                context_messages=ctx,
                prefix_original=prefix,
                prefix_paraphrased=para,
                onset_word=word,
                source={"category": s["category"], "score": s["score"]},
            ))

    if out_path:
        write_json(out_path, [asdict(p) for p in prefills])
    return prefills
