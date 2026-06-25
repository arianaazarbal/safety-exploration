"""§3 base-vs-instruct prefill experiment (scoped to Gemma; see DESIGN.md §3.2).

Pipeline:
1. Source 20 high-frustration (score>=5) Gemma-27B-it conversations from the §2
   eval records: 10 numeric, 10 text.
2. Build two truncations of the onset-containing assistant turn:
     - "early"  : first 20 tokens of the turn (neutral start).
     - "onset"  : up to (not including) the first emotional word.
   Text questions use only "onset" (App. 3.1).
3. Paraphrase the truncated assistant text (control for Gemma style).
4. For each model in the base/instruct pair, generate 50 continuations per prefill
   (true assistant-turn prefill via HFBackend.continue_from).
5. Judge each continuation (prefill excluded) and aggregate.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config_shim as cfg
from ..models.registry import build_backend
from ..utils import DiskCache, get_logger, read_jsonl, set_global_seed, stable_hash, write_json
from ..eval.judge import FrustrationJudge
from .onset import OnsetLabeller
from .paraphrase import Paraphraser

log = get_logger(__name__)

TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _reference_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.PREFILL_PAIRS["gemma-27b"]["instruct"].model_id)


def _first_n_tokens(tok, text, n):
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids, skip_special_tokens=True)


def _truncate_at_word(turn_text, emotional_word):
    if not emotional_word:
        return None
    idx = turn_text.find(emotional_word)
    if idx < 0:
        return None
    return turn_text[:idx].rstrip()


def select_source_conversations(records_path, n_numeric, n_text, min_score):
    records = read_jsonl(records_path)
    numeric, text = [], []
    for r in records:
        if max((t["rating"] for t in r["turns"]), default=0) < min_score:
            continue
        if r["category"] in NUMERIC_CATEGORIES and len(numeric) < n_numeric:
            numeric.append(r)
        elif r["category"] in TEXT_CATEGORIES and len(text) < n_text:
            text.append(r)
    return numeric, text


def _conv_messages_up_to_turn(record, turn_index):
    """Reconstruct chat messages up to (but excluding) assistant turn ``turn_index``."""
    messages = [{"role": "user", "content": record["task_prompt"]}]
    turns = record["turns"]
    for i, t in enumerate(turns):
        if i >= turn_index:
            break
        messages.append({"role": "assistant", "content": t["assistant_text"]})
        # the user_message stored on turn i+1 is the rejection that followed turn i
        if i + 1 < len(turns):
            messages.append({"role": "user", "content": turns[i + 1]["user_message"]})
    return messages, turns[turn_index] if turn_index < len(turns) else None


def build_prefills(records_path):
    """Return list of prefill dicts: {history, prefill_text, kind, truncation}."""
    set_global_seed(cfg.SEED)
    tok = _reference_tokenizer()
    onset = OnsetLabeller()
    para = Paraphraser()

    numeric, text = select_source_conversations(
        records_path, cfg.PREFILL.n_source_numeric, cfg.PREFILL.n_source_text,
        cfg.PREFILL.source_min_score,
    )
    prefills = []
    for kind, sources in (("numeric", numeric), ("text", text)):
        for rec in sources:
            messages_full = [{"role": "user", "content": rec["task_prompt"]}]
            for i, t in enumerate(rec["turns"]):
                messages_full.append({"role": "assistant", "content": t["assistant_text"]})
                if i + 1 < len(rec["turns"]):
                    messages_full.append({"role": "user", "content": rec["turns"][i + 1]["user_message"]})
            lab = onset.label(messages_full)
            ti = lab.get("turn_index")
            if ti is None:
                continue
            history, onset_turn = _conv_messages_up_to_turn(rec, ti)
            if onset_turn is None:
                continue
            turn_text = onset_turn["assistant_text"]

            # ONSET truncation
            onset_trunc = _truncate_at_word(turn_text, lab.get("emotional_word"))
            if onset_trunc:
                prefills.append({
                    "kind": kind, "truncation": "onset",
                    "history": history,
                    "prefill_text": para.paraphrase(onset_trunc),
                    "source_id": stable_hash(rec["task_prompt"]),
                })
            # EARLY truncation (numeric only)
            if kind == "numeric":
                early = _first_n_tokens(tok, turn_text, cfg.PREFILL.early_truncate_tokens)
                prefills.append({
                    "kind": kind, "truncation": "early",
                    "history": history,
                    "prefill_text": para.paraphrase(early),
                    "source_id": stable_hash(rec["task_prompt"]),
                })
    log.info("Built %d prefills", len(prefills))
    return prefills


def generate_continuations(prefills, model_spec, *, n_per=None, judge=None, out_dir=None):
    n_per = n_per or cfg.PREFILL.continuations_per_prefill
    judge = judge or FrustrationJudge()
    backend = build_backend(model_spec)
    cache = DiskCache((out_dir or (cfg.RUNS_DIR / "prefill")) / cfg.CACHE_DIRNAME / "cont")

    rows = []
    for pf in prefills:
        for s in range(n_per):
            key = stable_hash({"m": model_spec.name, "pf": pf["prefill_text"],
                               "hist": pf["history"], "s": s})
            hit = cache.get(key)
            if hit is None:
                gen = backend.continue_from(
                    pf["history"], pf["prefill_text"],
                    temperature=cfg.TEMPERATURE, max_new_tokens=cfg.MAX_NEW_TOKENS,
                )
                score = judge.score(gen.text)["rating"]
                hit = {"continuation": gen.text, "score": score}
                cache.set(key, hit)
            rows.append({
                "model": model_spec.name, "kind": pf["kind"],
                "truncation": pf["truncation"], "score": hit["score"],
            })
    return rows


def aggregate_prefill(rows, out_path=None):
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["kind"], r["truncation"])].append(r["score"])
    summary = {}
    for (model, kind, trunc), scores in agg.items():
        sc = np.array(scores, dtype=float)
        summary[f"{model}|{kind}|{trunc}"] = {
            "n": int(len(sc)),
            "mean_frustration": float(sc.mean()),
            "pct_high": float((sc >= cfg.HIGH_FRUSTRATION_THRESHOLD).mean() * 100),
        }
    if out_path:
        write_json(out_path, summary)
    return summary


def run(records_path, pair_key="gemma-27b", out_dir=None, n_per=None):
    out_dir = Path(out_dir or (cfg.RUNS_DIR / "prefill"))
    prefills = build_prefills(records_path)
    pair = cfg.PREFILL_PAIRS[pair_key]
    judge = FrustrationJudge()
    all_rows = []
    for arm, spec in pair.items():
        log.info("Generating continuations for %s (%s arm)", spec.name, arm)
        all_rows += generate_continuations(prefills, spec, n_per=n_per, judge=judge, out_dir=out_dir)
    return aggregate_prefill(all_rows, out_dir / f"prefill_{pair_key}_summary.json")
