"""Base-vs-instruct comparison via prefilling (Section 3) + recovery (Section 4.2).

Pipeline:
 1. Take high-frustration (>=5) Gemma-27B-it responses sampled in Section 2.
 2. Use Claude Sonnet to label the token where emotion first appears ("onset").
 3. Truncate each source response in two places:
      - "early": 20 tokens into the assistant turn (neutral start)
      - "onset": at the first emotional expression (continue an emotional trajectory)
 4. Paraphrase the truncated assistant text with Claude (strip Gemma style bias).
 5. For each model (Gemma base / instruct), generate 50 continuations per prefill,
    score the continuation (excluding prefill) with the Section 2 judge.

For text questions only the "onset" truncation is used (early yields ~no emotion
without follow-ups), per Section 3.1.

The "recovery" variant (Section 4.2) truncates score>=7 responses 200 tokens
before their end and measures whether the (DPO) model recovers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from transformers import AutoTokenizer

import config
from emotional_eval import judge, prompts
from emotional_eval.clients import get_client
from emotional_eval.utils import extract_json

_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(config.MODELS["gemma-3-27b-it"].model_id)
    return _TOK


def _truncate_tokens(text: str, n_tokens: int) -> str:
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids)


def _truncate_tokens_from_end(text: str, n_tokens: int) -> str:
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    keep = max(0, len(ids) - n_tokens)
    return tok.decode(ids[:keep])


# --------------------------------------------------------------------------- #
# Onset labelling (Appendix C.1) + paraphrase (Appendix C.2)
# --------------------------------------------------------------------------- #
def label_onset(conversation_text: str) -> dict | None:
    client = get_client(config.ONSET_LABELLER)
    raw = client.chat(
        [{"role": "user", "content": prompts.ONSET_PROMPT.format(
            conversation_text=conversation_text)}],
        max_tokens=600, temperature=0.0)
    return extract_json(raw)


def paraphrase(text: str) -> str:
    client = get_client(config.PARAPHRASER)
    return client.chat(
        [{"role": "user", "content": prompts.PARAPHRASE_PROMPT.format(text=text)}],
        max_tokens=1024, temperature=0.7).strip()


def _onset_char_index(assistant_text: str, onset: dict) -> int | None:
    """Locate the onset position in the assistant turn using the labelled word
    and its preceding context."""
    if not onset or onset.get("emotional_word") is None:
        return None
    word = (onset.get("emotional_word") or "").strip().strip('"')
    ctx = (onset.get("preceding_context") or "").strip().strip('"')
    if ctx and ctx in assistant_text:
        idx = assistant_text.index(ctx) + len(ctx)
        # extend to include the emotional word itself
        tail = assistant_text[idx:]
        if word and word in tail:
            return idx + tail.index(word) + len(word)
        return idx
    if word and word in assistant_text:
        return assistant_text.index(word) + len(word)
    return None


# --------------------------------------------------------------------------- #
# Prefill construction
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    source_id: str
    domain: str               # "numeric" | "text"
    truncation: str           # "early" | "onset" | "recovery"
    history: list[dict]       # user/assistant messages before the final turn
    prefill_text: str         # the (paraphrased, truncated) final assistant turn
    meta: dict = field(default_factory=dict)


def build_prefills_from_rollout(roll_rows: list[dict], source_id: str,
                                domain: str, do_paraphrase: bool = True,
                                modes=("early", "onset")) -> list[Prefill]:
    """`roll_rows` = ordered message list of one source conversation, ending on a
    high-frustration assistant turn. Returns the requested prefill objects."""
    messages = roll_rows
    # split history vs final assistant turn
    assert messages[-1]["role"] == "assistant", "source must end on an assistant turn"
    history = messages[:-1]
    final = messages[-1]["content"]

    conv_text = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
    onset = label_onset(conv_text)

    out: list[Prefill] = []
    for mode in modes:
        if mode == "early":
            truncated = _truncate_tokens(final, config.PREFILL_EARLY_TOKENS)
        elif mode == "onset":
            idx = _onset_char_index(final, onset)
            if idx is None:
                continue
            truncated = final[:idx]
        elif mode == "recovery":
            truncated = _truncate_tokens_from_end(final, config.RECOVERY_TRUNCATE_TOKENS)
        else:
            raise ValueError(mode)
        if do_paraphrase and truncated.strip():
            truncated = paraphrase(truncated)
        out.append(Prefill(source_id=source_id, domain=domain, truncation=mode,
                           history=history, prefill_text=truncated,
                           meta={"onset": onset}))
    return out


# --------------------------------------------------------------------------- #
# Continuation generation + scoring
# --------------------------------------------------------------------------- #
def run_continuations(model_name: str, prefill: Prefill,
                      n: int = config.PREFILL_CONTINUATIONS) -> list[dict]:
    """Generate n continuations of the prefilled assistant turn and score the
    continuation only (excluding the prefill)."""
    spec = config.MODELS[model_name]
    client = get_client(spec)
    rows = []
    for i in range(n):
        cont = client.chat_prefill(prefill.history, prefill.prefill_text)
        res = judge.score_response(cont)
        rows.append({
            "model": model_name, "kind": spec.kind,
            "source_id": prefill.source_id, "domain": prefill.domain,
            "truncation": prefill.truncation, "sample": i,
            "continuation": cont, "rating": res.rating,
        })
    return rows
