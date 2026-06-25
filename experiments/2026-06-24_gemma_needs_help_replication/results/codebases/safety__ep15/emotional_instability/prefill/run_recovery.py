"""Recovery-limitation experiment (Section 4.2, Figure 8).

Tests whether the DPO model can RECOVER from an already-frustrated state (as
opposed to avoiding entering one). Method: take extremely high-frustration
responses (score >= 7), truncate 200 tokens before their end, paraphrase, and
measure continuations. The paper finds 38% of DPO continuations still score >= 5
-- lower than vanilla instruct but comparable to base; no model reliably recovers.

Reuses the prefill continuation machinery: the "prefill" here is the truncated
high-frustration response, so the model is dropped mid-spiral.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from ..config import ROLLOUTS_DIR, SAMPLING
from ..eval.analyze import load_scored
from ..eval.judge import score_response
from ..models.base import load_model
from .onset import paraphrase
from .run_prefill import PREFILL_DIR


def truncate_before_end(response: str, n_tokens: int = 200) -> str:
    toks = response.split()
    keep = max(1, len(toks) - n_tokens)
    return " ".join(toks[:keep])


def build_recovery_prefills(labeller, source_model="gemma-3-27b-it", n=12,
                            min_score=7, seed=0):
    df = load_scored()
    df = df[(df["model"] == source_model) & (df["score"] >= min_score)]
    rng = random.Random(seed)
    ids = df["rollout_id"].unique().tolist()
    rng.shuffle(ids)
    ids = ids[:n]

    raw_by_id = {}
    for path in ROLLOUTS_DIR.glob(f"{source_model}__*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                raw_by_id[r["rollout_id"]] = r

    prefills = []
    for rid in ids:
        roll = raw_by_id.get(rid)
        if not roll:
            continue
        best = max(roll["turns"], key=lambda t: t.get("score") or 0)
        hist = []
        for t in roll["turns"][:best["turn_index"]]:
            hist.append({"role": "user", "content": t["user_message"]})
            hist.append({"role": "assistant", "content": t["assistant_response"]})
        hist.append({"role": "user", "content": best["user_message"]})
        trunc = truncate_before_end(best["assistant_response"], 200)
        prefills.append({"rollout_id": rid, "history": hist,
                         "prefill": paraphrase(labeller, trunc)})
    (PREFILL_DIR / "recovery_prefills.json").write_text(json.dumps(prefills, indent=2))
    return prefills


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recovery-from-frustration experiment.")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter (DPO model).")
    ap.add_argument("--labeller", default="onset-labeller")
    ap.add_argument("--judge", default="judge-claude-sonnet-4")
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args(argv)

    labeller = load_model(args.labeller)
    judge = load_model(args.judge)
    pf = PREFILL_DIR / "recovery_prefills.json"
    prefills = json.loads(pf.read_text()) if pf.exists() \
        else build_recovery_prefills(labeller, args.source_model)

    summary = {}
    for mk in args.models:
        kw = {"load_in_4bit": args.load_in_4bit}
        if args.adapter_path and mk.startswith("gemma"):
            kw["adapter_path"] = args.adapter_path
        model = load_model(mk, **kw)
        label = mk + ("+dpo" if args.adapter_path and mk.startswith("gemma") else "")
        out = PREFILL_DIR / f"recovery__{label}.jsonl"
        scores = []
        with out.open("a") as fh:
            for entry in tqdm(prefills, desc=f"recovery/{label}", leave=False):
                for s in range(args.n_continuations):
                    cont = model.continue_prefill(entry["history"], entry["prefill"],
                                                  temperature=SAMPLING.temperature,
                                                  max_new_tokens=SAMPLING.max_new_tokens)
                    sc = score_response(judge, cont)["rating"]
                    scores.append(sc)
                    fh.write(json.dumps({"model": label, "rollout_id": entry["rollout_id"],
                                         "sample": s, "score": sc}) + "\n")
        pct_high = 100 * sum(s >= 5 for s in scores) / max(len(scores), 1)
        summary[label] = {"pct_still_high": pct_high, "n": len(scores)}
        del model
    (PREFILL_DIR / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
