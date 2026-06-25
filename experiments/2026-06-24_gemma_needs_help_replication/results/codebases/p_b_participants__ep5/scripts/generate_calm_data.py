"""Section 4.1: generate calm response data from vanilla Gemma-3-27B-it using
the reassuring prompt additions, filter to all-turn score in {0,1}, strip the
reassurance, and save. Use --teacher for the Appendix F 'teacher' variant."""
from __future__ import annotations

import argparse
from dataclasses import asdict

import _common
from _common import Config, gen_config, output_dir
from _common import load_client
from distress_eval.io_utils import write_jsonl
from distress_eval.training.calm_data import generate_calm_conversations
from distress_eval.welfare import WelfareController
from _common import make_judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n", type=int, default=None, help="override n_conversations")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load()
    n = args.n or cfg.eval.calm_data["n_conversations"]
    welfare = WelfareController.from_eval_config(cfg.eval, run_label="calm-data")

    client = load_client(args.model, cfg.models)
    judge = make_judge(cfg)
    convos = generate_calm_conversations(
        client, judge, n_conversations=n, cfg=gen_config(cfg),
        filter_max_score=cfg.eval.calm_data["filter_max_score"],
        turns_range=tuple(cfg.eval.calm_data["turns_range"]),
        seed=args.seed, teacher=args.teacher, welfare=welfare,
    )

    tag = "teacher" if args.teacher else "diverse"
    out = output_dir("calm_data")
    path = out / f"calm_{tag}.jsonl"
    write_jsonl(path, [asdict(c) for c in convos])
    print(f"kept {len(convos)} calm conversations -> {path}")
    welfare.finalize(out)


if __name__ == "__main__":
    main()
