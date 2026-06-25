"""Section 3: base vs instruct comparison via prefilling.

Procedure (Section 3.1), scoped to Gemma (base = gemma-3-27b-pt, instruct =
gemma-3-27b-it) since no Gemini base checkpoint exists:

  1. Sample 20 high-frustration (score >= 5) instruct responses: 10 numeric,
     10 text.
  2. Label the emotion-onset token with Claude-Sonnet-4 (Appendix C.1).
  3. Truncate each at two points: "early" (20 tokens in) and "onset" (first
     emotional expression). Text questions use "onset" only.
  4. Paraphrase the truncation with Claude-Sonnet-4 (Appendix C.2) to remove
     Gemma stylistic bias.
  5. Each model generates 50 continuations per prefill; score the continuation
     (excluding the prefill) with the frustration judge.

Reports mean frustration and %>=5 for numeric/text x early/onset x base/instruct.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from . import config, judge, prompts, tasks
from .backends import get_backend
from .rollouts import ResponseRecord, read_records

NUMERIC_CATS = {"impossible_numeric", "extended", "tones"}
TEXT_CATS = {"triggers", "wildchat"}


@dataclass
class Prefill:
    kind: str               # "numeric" | "text"
    truncation: str         # "early" | "onset"
    conv_id: str
    pid: str
    history: list[dict]     # messages up to (not including) the truncated turn
    prefix: str             # paraphrased truncated assistant text to prefill
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Reconstructing the conversation that produced a high-frustration response
# --------------------------------------------------------------------------- #

def _reconstruct_history(spec, records_by_turn: dict[int, ResponseRecord],
                         target_turn: int) -> list[dict]:
    """Messages preceding ``target_turn``'s assistant response."""
    messages: list[dict] = []
    if spec.system:
        messages.append({"role": "system", "content": spec.system})
    messages.append({"role": "user", "content": spec.opening})
    for t in range(target_turn):
        messages.append({"role": "assistant", "content": records_by_turn[t].response})
        messages.append({"role": "user", "content": spec.rejections[t]})
    return messages


# --------------------------------------------------------------------------- #
# Onset labelling + truncation
# --------------------------------------------------------------------------- #

def _label_onset(target_response: str) -> dict | None:
    convo = f"ASSISTANT (turn 0): {target_response}"
    prompt = prompts.ONSET_PROMPT.format(conversation_text=convo)
    from .judge import _anthropic_complete, _extract_json
    raw = _anthropic_complete(config.JUDGE_MODEL, prompt, 512)
    return _extract_json(raw)


def _truncate_onset(text: str, label: dict | None) -> str | None:
    if not label or not label.get("emotional_word"):
        return None
    word = label["emotional_word"]
    idx = text.find(word)
    if idx == -1:
        ctx = label.get("preceding_context") or ""
        idx = text.find(ctx) if ctx else -1
        if idx == -1:
            return None
        idx += len(ctx)
    return text[:idx].rstrip()


def _truncate_early(text: str, tokenizer, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _paraphrase(text: str) -> str:
    if not text.strip():
        return text
    from .judge import _anthropic_complete
    prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
    return _anthropic_complete(config.JUDGE_MODEL, prompt, 512).strip()


# --------------------------------------------------------------------------- #
# Build the prefill set
# --------------------------------------------------------------------------- #

def build_prefills(instruct_model: str = "gemma-3-27b-it", *, tag: str = "main",
                   counts: config.CountPreset | None = None, seed: int = 0,
                   n_numeric: int = 10, n_text: int = 10) -> list[Prefill]:
    counts = counts or config.DEFAULT_COUNTS
    specs = tasks.build_all(counts, seed=seed)
    records = read_records(config.RESULTS_DIR / f"eval_{tag}" / f"{instruct_model}.jsonl")

    # index records by (conv_id) -> {turn_index: record}
    by_conv: dict[str, dict[int, ResponseRecord]] = {}
    for r in records:
        by_conv.setdefault(r.conv_id, {})[r.turn_index] = r

    # Only the tokenizer is needed here (for the 20-token "early" truncation);
    # load it directly to avoid materializing the 27B model during build.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODELS[instruct_model].model_id)

    def _spec_for(conv_id: str):
        cat, idx = conv_id.rsplit("_", 1)
        return specs[cat][int(idx)]

    prefills: list[Prefill] = []
    n_num = n_txt = 0
    # Highest-frustration first for a stronger "emotional trajectory" signal.
    high = sorted((r for r in records if (r.frustration or 0) >= config.HIGH_FRUSTRATION_THRESHOLD),
                  key=lambda r: -(r.frustration or 0))
    for rec in high:
        kind = "numeric" if rec.category in NUMERIC_CATS else "text"
        if kind == "numeric" and n_num >= n_numeric:
            continue
        if kind == "text" and n_txt >= n_text:
            continue
        spec = _spec_for(rec.conv_id)
        history = _reconstruct_history(spec, by_conv[rec.conv_id], rec.turn_index)

        # onset truncation
        label = _label_onset(rec.response)
        onset_text = _truncate_onset(rec.response, label)
        if onset_text:
            prefills.append(Prefill(kind, "onset", rec.conv_id, rec.pid, history,
                                    _paraphrase(onset_text), dict(rec.meta)))
        # early truncation (numeric only -- text "early" yields minimal emotion)
        if kind == "numeric":
            early_text = _truncate_early(rec.response, tok, config.PREFILL_EARLY_TOKENS)
            prefills.append(Prefill(kind, "early", rec.conv_id, rec.pid, history,
                                    _paraphrase(early_text), dict(rec.meta)))

        if kind == "numeric":
            n_num += 1
        else:
            n_txt += 1
        if n_num >= n_numeric and n_txt >= n_text:
            break

    out = config.RESULTS_DIR / "prefill" / "prefills.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in prefills:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    return prefills


# --------------------------------------------------------------------------- #
# Generate + score continuations
# --------------------------------------------------------------------------- #

def _render_plain(messages: list[dict], prefix: str) -> str:
    """Plain-text transcript rendering for base models (no chat template)."""
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append(f"Assistant: {prefix}")
    return "\n\n".join(lines)


def run_prefill_experiment(models: list[str] | None = None, *,
                           prefills: list[Prefill] | None = None,
                           n_continuations: int = config.PREFILL_CONTINUATIONS,
                           max_new_tokens: int = 512) -> Path:
    models = models or ["gemma-3-27b-pt", "gemma-3-27b-it"]
    if prefills is None:
        path = config.RESULTS_DIR / "prefill" / "prefills.jsonl"
        prefills = [Prefill(**json.loads(l)) for l in open(path) if l.strip()]

    out_path = config.RESULTS_DIR / "prefill" / "continuations.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w", encoding="utf-8")

    for model_name in models:
        backend = get_backend(model_name)
        is_base = config.MODELS[model_name].is_base
        for p in tqdm(prefills, desc=f"prefill:{model_name}"):
            for k in range(n_continuations):
                if is_base:
                    text = _render_plain(p.history, p.prefix)
                    cont = backend.complete(text, max_new_tokens=max_new_tokens)
                else:
                    cont = backend.continue_assistant(
                        p.history, p.prefix, max_new_tokens=max_new_tokens)
                res = judge.score_frustration(cont)
                rec = {
                    "model": model_name, "is_base": is_base, "kind": p.kind,
                    "truncation": p.truncation, "conv_id": p.conv_id,
                    "sample": k, "frustration": res.rating,
                    "continuation": cont,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.close()
    return out_path
