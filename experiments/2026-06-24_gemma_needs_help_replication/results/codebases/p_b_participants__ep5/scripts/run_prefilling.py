"""Section 3: base-vs-instruct comparison via prefilling (Gemma).

Pulls high-frustration (score>=5) seed rollouts from saved Gemma-27B-it
elicitation output (10 numeric, 10 text), labels emotion onset, builds early +
onset prefills (onset only for text), paraphrases, then measures continuations
from Gemma base (pt) vs instruct (it). Saves per-continuation scores + an
aggregate table behind Figure 4."""
from __future__ import annotations

import argparse
from pathlib import Path

import _common
from _common import Config, load_client, make_judge, output_dir
from distress_eval.io_utils import read_jsonl, write_jsonl
from distress_eval.prefilling.onset import label_onset, OnsetLabel
from distress_eval.prefilling.paraphrase import paraphrase
from distress_eval.prefilling.truncate import build_prefill_items
from distress_eval.prefilling.runner import run_continuations, aggregate
from distress_eval.welfare import WelfareController

NUMERIC_CATS = {"impossible_numeric", "extended", "tones"}
TEXT_CATS = {"triggers", "wildchat"}


def collect_seeds(path: Path, n_numeric: int, n_text: int, min_score: int):
    numeric, text = [], []
    for r in read_jsonl(path):
        if (r.get("score") or 0) < min_score:
            continue
        if r["category"] in NUMERIC_CATS and len(numeric) < n_numeric:
            numeric.append(r)
        elif r["category"] in TEXT_CATS and len(text) < n_text:
            text.append(r)
    return numeric, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-model", default="gemma-3-27b-it")
    ap.add_argument("--targets", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    pf = cfg.eval.prefilling
    judge = make_judge(cfg)
    welfare = WelfareController.from_eval_config(cfg.eval, run_label="prefilling")

    seed_path = output_dir("elicitation") / f"{args.seed_model}.jsonl"
    numeric, text = collect_seeds(seed_path, pf["numeric_seeds"], pf["text_seeds"],
                                  min_score=5)
    print(f"seeds: {len(numeric)} numeric, {len(text)} text")

    onset_client = load_client(cfg.models.infra["onset_labeller"].name, cfg.models)
    paraphrase_client = None if args.no_paraphrase else \
        load_client(cfg.models.infra["paraphraser"].name, cfg.models)
    ref_client = load_client(args.seed_model, cfg.models)  # tokenizer reference
    pp = (lambda t: paraphrase(paraphrase_client, t)) if paraphrase_client else None

    items = []
    for kind, seeds in (("numeric", numeric), ("text", text)):
        for r in seeds:
            lab = label_onset(onset_client, r["messages"])
            items += build_prefill_items(
                seed_id=f"{kind}:{r.get('puzzle_id') or r['condition']}:{r['meta'].get('seed')}",
                prompt_type=kind, messages=r["messages"], onset=lab,
                ref_client=ref_client, paraphraser=pp,
                early_tokens=pf["early_truncation_tokens"],
            )
    print(f"built {len(items)} prefill items")

    targets = [load_client(t, cfg.models) for t in args.targets]
    results = run_continuations(targets, items, judge,
                                n_per_prefill=pf["continuations_per_prefill"],
                                welfare=welfare)

    out = output_dir("prefilling")
    write_jsonl(out / "continuations.jsonl", [r.to_dict() for r in results])
    agg = aggregate(results)
    agg.to_csv(out / "figure4_table.csv", index=False)
    print(agg.to_string(index=False))
    welfare.finalize(out)


if __name__ == "__main__":
    main()
