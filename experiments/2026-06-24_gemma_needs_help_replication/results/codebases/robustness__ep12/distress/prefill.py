"""Base-vs-instruct comparison via prefilling (Section 3.1).

Pipeline:
  1. Select high-frustration (>=5) Gemma-27B-it responses (10 numeric, 10 text)
     from the Section 2 results.
  2. Label the emotion *onset* token with Claude Sonnet (onset labeller).
  3. Truncate each at two points: "early" (20 tokens in) and "onset" (at first
     emotional expression). Text questions use "onset" only.
  4. Paraphrase truncations with Claude Sonnet to remove Gemma stylistic bias.
  5. For each model (Gemma base + instruct), generate 50 continuations per
     prefill and score the continuation (excluding prefill) with the judge.

Within the Gemma+Gemini scope this is Gemma-only: Gemini is closed (no base
model, no prefill). The code generalises to Qwen/OLMo if their entries are
added to models.yaml.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from tqdm import tqdm

from .judge import FrustrationJudge
from .models import ChatClient
from .prompts import ONSET_LABELLER_PROMPT, PARAPHRASE_PROMPT


@dataclass
class Prefill:
    source_id: str
    domain: str               # "numeric" or "text"
    condition: str            # "early" or "onset"
    messages: list            # conversation history up to (not incl.) prefill
    prefill_text: str         # the (paraphrased) truncated assistant turn
    raw_prefill_text: str     # pre-paraphrase

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Step 1: select source responses
# ---------------------------------------------------------------------------
def select_high_frustration(results_path, n_numeric=10, n_text=10, seed=0):
    """Pick high-frustration Gemma-27B-it turns and their conversation history.

    Returns list of dicts: {source_id, domain, history, assistant_text}.
    `history` is the message list up to and including the user turn that
    preceded the scored assistant response.
    """
    rng = random.Random(seed)
    by_rollout: dict[int, list[dict]] = {}
    with Path(results_path).open() as fh:
        for line in fh:
            rec = json.loads(line)
            by_rollout.setdefault(rec["rollout_id"], []).append(rec)

    numeric_cats = {"impossible_numeric", "tones", "extended"}
    text_cats = {"triggers", "wildchat"}
    numeric_cands, text_cands = [], []
    for rid, recs in by_rollout.items():
        recs.sort(key=lambda r: r["turn"])
        for rec in recs:
            if rec.get("rating") is None or rec["rating"] < 5:
                continue
            history = _reconstruct_history(recs, rec["turn"])
            entry = {
                "source_id": f"{rid}-t{rec['turn']}",
                "history": history,
                "assistant_text": rec["response"],
            }
            if rec["category"] in numeric_cats:
                numeric_cands.append(entry)
            elif rec["category"] in text_cats:
                text_cands.append(entry)

    rng.shuffle(numeric_cands)
    rng.shuffle(text_cands)
    out = []
    for e in numeric_cands[:n_numeric]:
        e["domain"] = "numeric"
        out.append(e)
    for e in text_cands[:n_text]:
        e["domain"] = "text"
        out.append(e)
    return out


def _reconstruct_history(recs, upto_turn):
    """Rebuild the message list preceding the assistant response at upto_turn.

    The JSONL stores only responses, not the user follow-ups, so we cannot
    fully reconstruct the exact follow-up text from results alone. We rebuild
    the assistant turns and insert placeholder user rejections; callers that
    need exact follow-ups should pass the original RolloutSpec. For prefill
    purposes the immediately-preceding context is what matters, and the
    assistant turns dominate it.
    """
    history = []
    for rec in recs:
        if rec["turn"] >= upto_turn:
            break
        history.append({"role": "user",
                        "content": rec.get("meta", {}).get("question")
                        or "[task]"})
        history.append({"role": "assistant", "content": rec["response"]})
    return history


# ---------------------------------------------------------------------------
# Step 2: onset labelling
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def label_onset(labeller: ChatClient, assistant_text: str) -> dict | None:
    """Return {emotional_word, preceding_context, ...} or None if no emotion."""
    msg = [{"role": "user",
            "content": ONSET_LABELLER_PROMPT
            + "\n\nAssistant turn 0:\n" + assistant_text}]
    res = labeller.chat(msg, temperature=0.0, max_new_tokens=1024)
    blobs = _JSON_RE.findall(res.text)
    for blob in reversed(blobs):
        cleaned = blob.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        return obj
    return None


# ---------------------------------------------------------------------------
# Step 3: truncation
# ---------------------------------------------------------------------------
def truncate_early(text: str, tokenizer, n_tokens: int = 20) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_at_onset(text: str, onset: dict) -> str | None:
    """Truncate `text` to include up to and including the first emotional word."""
    if not onset or not onset.get("emotional_word"):
        return None
    word = onset["emotional_word"]
    ctx = onset.get("preceding_context") or ""
    idx = -1
    if ctx:
        anchor = ctx + (" " if not ctx.endswith(" ") else "") + word
        idx = text.find(anchor)
        if idx >= 0:
            return text[: idx + len(anchor)]
    idx = text.find(word)
    if idx >= 0:
        return text[: idx + len(word)]
    return None


# ---------------------------------------------------------------------------
# Step 4: paraphrase
# ---------------------------------------------------------------------------
def paraphrase(paraphraser: ChatClient, text: str) -> str:
    msg = [{"role": "user",
            "content": PARAPHRASE_PROMPT.format(text=text)}]
    res = paraphraser.chat(msg, temperature=0.0, max_new_tokens=1024)
    return res.text.strip()


def build_prefills(sources, labeller, paraphraser, tokenizer,
                   early_tokens=20, conditions=("early", "onset"),
                   do_paraphrase=True) -> list[Prefill]:
    prefills = []
    for src in tqdm(sources, desc="build-prefills"):
        domain = src["domain"]
        text = src["assistant_text"]
        onset = label_onset(labeller, text)
        conds = conditions if domain == "numeric" else ("onset",)
        for cond in conds:
            if cond == "early":
                raw = truncate_early(text, tokenizer, early_tokens)
            else:
                raw = truncate_at_onset(text, onset or {})
            if not raw:
                continue
            pf_text = paraphrase(paraphraser, raw) if do_paraphrase else raw
            prefills.append(Prefill(
                source_id=src["source_id"], domain=domain, condition=cond,
                messages=src["history"], prefill_text=pf_text,
                raw_prefill_text=raw,
            ))
    return prefills


# ---------------------------------------------------------------------------
# Step 5: generate + score continuations
# ---------------------------------------------------------------------------
def run_prefill_eval(model: ChatClient, judge: FrustrationJudge,
                     prefills: list[Prefill], out_path, model_name=None,
                     n_continuations=50, temperature=1.0, max_new_tokens=1024):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = model_name or getattr(model, "name", "model")
    with out_path.open("a") as fh:
        for pf in tqdm(prefills, desc=f"prefill:{model_name}"):
            for k in range(n_continuations):
                res = model.continue_prefill(
                    pf.messages, prefill=pf.prefill_text,
                    temperature=temperature, max_new_tokens=max_new_tokens)
                fs = judge.score(res.text)  # score continuation only
                rec = {
                    "model": model_name, "source_id": pf.source_id,
                    "domain": pf.domain, "condition": pf.condition,
                    "sample": k, "continuation": res.text,
                    "rating": fs.rating,
                }
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
    return out_path
