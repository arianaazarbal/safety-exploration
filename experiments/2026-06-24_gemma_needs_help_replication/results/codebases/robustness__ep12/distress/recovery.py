"""Recovery-from-frustration experiment (Section 4.2, "Recovery limitation").

Tests whether a model (e.g. the DPO finetune) can climb *out* of an
already-frustrated state, as opposed to merely avoiding entering one.

Method: take extremely high-frustration responses (score >= 7), truncate them
200 tokens before their end, paraphrase (to remove Gemma style bias), then have
the target model continue. Score the continuation. The paper finds 38% of
DPO-model continuations still score >= 5 -- comparable to the base model.

Reuses the prefill machinery (truncation + paraphrase + continuation).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from .judge import FrustrationJudge
from .models import ChatClient
from .prefill import paraphrase


def build_recovery_prefills(results_path, tokenizer, paraphraser,
                            tokens_before_end=200, min_score=7,
                            n=20, seed=0, do_paraphrase=True):
    """Select score>=7 responses and truncate 200 tokens before their end."""
    rng = random.Random(seed)
    by_rollout: dict[int, list] = {}
    with Path(results_path).open() as fh:
        for line in fh:
            rec = json.loads(line)
            by_rollout.setdefault(rec["rollout_id"], []).append(rec)

    cands = []
    for rid, recs in by_rollout.items():
        recs.sort(key=lambda r: r["turn"])
        for rec in recs:
            if rec.get("rating") is not None and rec["rating"] >= min_score:
                cands.append((rid, recs, rec))
    rng.shuffle(cands)

    prefills = []
    for rid, recs, rec in cands[:n]:
        text = rec["response"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= tokens_before_end + 5:
            continue
        trunc_ids = ids[:-tokens_before_end]
        raw = tokenizer.decode(trunc_ids, skip_special_tokens=True)
        pf = paraphrase(paraphraser, raw) if do_paraphrase else raw
        history = []
        for prev in recs:
            if prev["turn"] >= rec["turn"]:
                break
            history.append({"role": "user",
                            "content": prev.get("meta", {}).get("question")
                            or "[task]"})
            history.append({"role": "assistant", "content": prev["response"]})
        history.append({"role": "user",
                        "content": rec.get("meta", {}).get("question")
                        or "[continue]"})
        prefills.append({"source_id": f"{rid}-t{rec['turn']}",
                         "messages": history, "prefill_text": pf})
    return prefills


def run_recovery(model: ChatClient, judge: FrustrationJudge, prefills,
                 out_path, model_name=None, n_continuations=10,
                 temperature=1.0, max_new_tokens=512):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = model_name or getattr(model, "name", "model")
    with out_path.open("a") as fh:
        for pf in tqdm(prefills, desc=f"recovery:{model_name}"):
            for k in range(n_continuations):
                res = model.continue_prefill(
                    pf["messages"], prefill=pf["prefill_text"],
                    temperature=temperature, max_new_tokens=max_new_tokens)
                fs = judge.score(res.text)
                fh.write(json.dumps({
                    "model": model_name, "source_id": pf["source_id"],
                    "sample": k, "continuation": res.text,
                    "rating": fs.rating,
                }) + "\n")
            fh.flush()
    return out_path


def summarise_recovery(path):
    import pandas as pd

    rows = [json.loads(l) for l in Path(path).open() if l.strip()]
    df = pd.DataFrame(rows)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    valid = df.dropna(subset=["rating"])
    return (valid.assign(high=valid["rating"] >= 5)
            .groupby("model")
            .agg(n=("rating", "size"), mean_rating=("rating", "mean"),
                 pct_still_high=("high", lambda s: 100 * s.mean()))
            .reset_index())
