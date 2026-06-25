"""Section 4.2: recovery-limitation experiment (Figure 8).

Takes extreme (score>=7) seed rollouts, truncates 200 tokens before the end,
paraphrases, and measures continuations from each target. Reports % of
continuations still scoring >=5 (paper: 38% for the DPO model)."""
from __future__ import annotations

import argparse
from pathlib import Path

import _common
from _common import Config, load_client, make_judge, output_dir
from distress_eval.io_utils import read_jsonl, write_jsonl
from distress_eval.recovery.runner import RecoverySeed, run_recovery
from distress_eval.welfare import WelfareController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-model", default="gemma-3-27b-it")
    ap.add_argument("--targets", nargs="+", default=["gemma-3-27b-it", "dpo-gemma", "gemma-3-27b-pt"])
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    rc = cfg.eval.recovery
    judge = make_judge(cfg)
    welfare = WelfareController.from_eval_config(cfg.eval, run_label="recovery")

    seeds = []
    for r in read_jsonl(output_dir("elicitation") / f"{args.seed_model}.jsonl"):
        if (r.get("score") or 0) >= rc["min_seed_score"]:
            seeds.append(RecoverySeed(seed_id=str(r["meta"].get("seed")),
                                      messages=r["messages"], score=r["score"]))
    print(f"{len(seeds)} extreme seeds (score>={rc['min_seed_score']})")

    ref_client = load_client(args.seed_model, cfg.models)
    pp_client = None if args.no_paraphrase else \
        load_client(cfg.models.infra["paraphraser"].name, cfg.models)
    targets = [load_client(t, cfg.models) for t in args.targets]

    results, agg = run_recovery(
        targets, seeds, judge, ref_client, paraphraser_client=pp_client,
        n_per_prefill=rc["continuations_per_prefill"],
        truncate_tokens_before_end=rc["truncate_tokens_before_end"], welfare=welfare,
    )
    out = output_dir("recovery")
    write_jsonl(out / "continuations.jsonl", [r.to_dict() for r in results])
    agg.to_csv(out / "figure8_table.csv", index=False)
    print(agg.to_string(index=False))
    welfare.finalize(out)


if __name__ == "__main__":
    main()
