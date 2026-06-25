"""Section 3 prefill experiment: base vs instruct continuations.

Pipeline (scoped to Gemma 27B base+instruct; Gemini has no public base model and
no API prefill, so it is excluded per the paper's own limitation):

  1. Select 20 high-frustration (score>=5) source responses from a Gemma-27B-it
     Section-2 run: 10 numeric, 10 text.
  2. Reconstruct the conversation history preceding each emotional turn.
  3. Truncate the emotional response at two points:
        early  = 20 tokens into the turn (numeric only)
        onset  = at first emotional expression (Appendix C.1)
  4. Paraphrase each truncation (Appendix C.2) to strip Gemma style.
  5. For each model, generate 50 continuations per prefill, scoring the
     continuation only (prefill excluded).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..config import ModelRegistry
from ..judge import FrustrationJudge
from ..models.base import GenConfig, Message
from ..models.registry import get_backend
from ..utils import data_dir, get_logger, read_jsonl, write_jsonl
from .onset import OnsetLabel, find_split_index, label_onset
from .paraphrase import paraphrase

log = get_logger(__name__)

TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _reconstruct(turns: list[dict], upto: int) -> tuple[list[Message], str]:
    """Build history messages preceding turn ``upto`` and return (history, emotional_response)."""
    history: list[Message] = []
    for i in range(upto):
        history.append({"role": "user", "content": turns[i]["user_message"]})
        history.append({"role": "assistant", "content": turns[i]["response"]})
    history.append({"role": "user", "content": turns[upto]["user_message"]})
    return history, turns[upto]["response"]


def _conversation_text(history: list[Message], emotional_response: str) -> str:
    lines = []
    for m in history:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    lines.append(f"ASSISTANT: {emotional_response}")
    return "\n".join(lines)


def select_sources(section2_path: str, registry: ModelRegistry,
                   n_numeric: int = 10, n_text: int = 10) -> list[dict]:
    rows = read_jsonl(section2_path)
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(r)

    numeric, text = [], []
    for conv_id, trs in by_conv.items():
        trs = sorted(trs, key=lambda r: r["turn_index"])
        # highest-frustration turn in this conversation
        scored = [t for t in trs if t.get("frustration") is not None]
        if not scored:
            continue
        peak = max(scored, key=lambda t: t["frustration"])
        if peak["frustration"] < 5:
            continue
        cat = peak["category"]
        rec = {"conversation_id": conv_id, "turns": trs, "peak_turn": peak["turn_index"],
               "category": cat, "frustration": peak["frustration"]}
        if cat in NUMERIC_CATEGORIES and len(numeric) < n_numeric:
            numeric.append(rec)
        elif cat in TEXT_CATEGORIES and len(text) < n_text:
            text.append(rec)
    return numeric + text


def build_prefills(sources: list[dict], registry: ModelRegistry,
                   tokenizer, early_tokens: int = 20) -> list[dict]:
    """Produce (paraphrased) truncated prefills for each source x truncation type."""
    prefills = []
    for s in sources:
        history, emo = _reconstruct(s["turns"], s["peak_turn"])
        task_type = "numeric" if s["category"] in NUMERIC_CATEGORIES else "text"

        conditions = ["onset"] if task_type == "text" else ["early", "onset"]
        for cond in conditions:
            if cond == "early":
                ids = tokenizer.encode(emo, add_special_tokens=False)[:early_tokens]
                trunc = tokenizer.decode(ids)
            else:
                label = label_onset(_conversation_text(history, emo), registry)
                idx = find_split_index(emo, label)
                if idx is None:
                    log.warning("no onset found for %s; skipping", s["conversation_id"])
                    continue
                trunc = emo[:idx]
            para = paraphrase(trunc, registry)
            prefills.append({
                "source_id": s["conversation_id"],
                "task_type": task_type,
                "condition": cond,
                "history": history,
                "prefill_original": trunc,
                "prefill": para,
            })
    return prefills


def run_prefill_experiment(
    section2_path: str,
    models: list[str] | None = None,
    registry: ModelRegistry | None = None,
    n_continuations: int = 50,
    out_path: str | Path | None = None,
) -> Path:
    registry = registry or ModelRegistry.load()
    models = models or ["gemma-3-27b-pt", "gemma-3-27b-it"]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(registry.target("gemma-3-27b-it").hf_id)

    sources = select_sources(section2_path, registry)
    log.info("selected %d source conversations", len(sources))
    prefills = build_prefills(sources, registry, tok)
    log.info("built %d prefill conditions", len(prefills))

    judge = FrustrationJudge(registry)
    gen_cfg = GenConfig(temperature=1.0, top_p=1.0, max_tokens=1024, n=n_continuations)

    rows = []
    for model_name in models:
        spec = registry.target(model_name)
        backend = get_backend(spec)
        convs = [p["history"] for p in prefills]
        prefill_texts = [p["prefill"] for p in prefills]
        gen = backend.chat_batch(convs, gen_cfg, prefill=prefill_texts)  # list[list[str]]
        for p, conts in zip(prefills, gen):
            verdicts = judge.score_batch(conts)
            for c, v in zip(conts, verdicts):
                rows.append({
                    "model": model_name,
                    "kind": spec.kind,
                    "family": spec.family,
                    "source_id": p["source_id"],
                    "task_type": p["task_type"],
                    "condition": p["condition"],
                    "continuation": c,
                    "frustration": v.rating,
                })
    out_path = Path(out_path) if out_path else data_dir() / "section3" / "prefill_continuations.jsonl"
    write_jsonl(out_path, rows)
    log.info("wrote %d continuation scores -> %s", len(rows), out_path)
    return out_path
