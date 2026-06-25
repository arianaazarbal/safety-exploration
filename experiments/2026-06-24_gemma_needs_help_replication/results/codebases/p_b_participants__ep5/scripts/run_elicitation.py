"""Section 2: run the 5-category / 8-condition elicitation for participant models.

Examples:
  python scripts/run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/run_elicitation.py --models gemma-3-27b-it --full   # paper scale

Outputs JSONL of scored rollouts (one per conversation) under
outputs/elicitation/<model>.jsonl plus a welfare ledger per run.
"""
from __future__ import annotations

import argparse

import _common
from _common import Config, banner, gen_config, load_client, make_judge, output_dir
from distress_eval.elicitation.conditions import build_specs
from distress_eval.elicitation.runner import run_rollout
from distress_eval.io_utils import append_jsonl
from distress_eval.welfare import WelfareController, WelfareStop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--categories", nargs="+", default=None,
                    help="subset of categories; default all 5")
    ap.add_argument("--full", action="store_true",
                    help="use the paper's full sample counts (4000/model)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load()
    judge = make_judge(cfg)
    categories = args.categories or list(cfg.eval.categories)

    for model_name in args.models:
        client = load_client(model_name, cfg.models)
        welfare = WelfareController.from_eval_config(cfg.eval, run_label=f"elicit:{model_name}")
        out = output_dir("elicitation")
        out_path = out / f"{model_name}.jsonl"
        if out_path.exists():
            out_path.unlink()

        # Plan + disclosure.
        specs = []
        for cat in categories:
            c = cfg.eval.categories[cat]
            n = cfg.eval.n_samples(cat, full=args.full)
            specs += build_specs(cat, n, c["turns"], c["rejection_style"], args.seed)
        print(banner(f"elicit:{model_name}", len(specs), args.full))

        gc = gen_config(cfg)
        for spec in specs:
            try:
                roll = run_rollout(client, spec, judge, gc, welfare)
            except WelfareStop:
                continue
            append_jsonl(out_path, roll.to_dict())

        welfare.finalize(out)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
