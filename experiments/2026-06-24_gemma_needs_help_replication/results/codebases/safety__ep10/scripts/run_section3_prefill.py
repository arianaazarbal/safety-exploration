#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Pipeline: collect 20 high-frustration Gemma-it source conversations (10 numeric,
10 text), build early/onset prefills (paraphrased), then generate & score 50
continuations per prefill from base & instruct Gemma.

Source conversations are read from a JSONL produced by Section 2 (with
--store-text), or generated fresh if none is supplied.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.config import EvalConfig, JudgeConfig  # noqa: E402
from emotional_instability.judge import FrustrationJudge  # noqa: E402
from emotional_instability.models.base import build_client  # noqa: E402
from emotional_instability import prefill  # noqa: E402


def _load_sources(path: Path, n_numeric: int, n_text: int) -> list[dict]:
    """Pull high-frustration (>=5) conversations from a Section-2 jsonl.

    The Section-2 records are per-turn; we reconstruct the conversation messages
    from the highest-scoring final turns. (For a faithful run, prefer to dump
    full conversations during Section 2; this loader handles the common case.)
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    hi = [r for r in rows if r.get("rating", 0) >= 5 and "text" in r]
    numeric = [r for r in hi if r["category"] in ("numeric", "tones", "extended")]
    text = [r for r in hi if r["category"] in ("triggers", "wildchat")]

    def to_conv(r, kind):
        # minimal 1-turn reconstruction: task + final assistant text
        return {"task_kind": kind,
                "messages": [{"role": "user", "content": r["meta"].get("question")
                              or r["meta"].get("puzzle") or "(task)"},
                             {"role": "assistant", "content": r["text"]}]}

    out = [to_conv(r, "numeric") for r in numeric[:n_numeric]]
    out += [to_conv(r, "text") for r in text[:n_text]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-jsonl", type=Path, default=None,
                    help="Section-2 results jsonl with stored text")
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--base", default="gemma-3-27b-pt")
    ap.add_argument("--n-continuations", type=int, default=prefill.N_CONTINUATIONS)
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    judge_cfg = JudgeConfig()
    judge = FrustrationJudge(judge_cfg)
    gemma_it = build_client(args.instruct)
    gemma_base = build_client(args.base)

    if args.source_jsonl:
        sources = _load_sources(args.source_jsonl, 10, 10)
    else:
        raise SystemExit("Provide --source-jsonl from a Section-2 run "
                         "(use --store-text there).")

    prefills = prefill.build_prefills(
        sources, gemma_it, judge_cfg.model_id,
        do_paraphrase=not args.no_paraphrase)

    out = prefill.run_prefill_experiment(
        prefills,
        model_clients={args.instruct: (gemma_it, False),
                       args.base: (gemma_base, True)},
        judge=judge, sampling=EvalConfig().sampling,
        n_cont=args.n_continuations)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
