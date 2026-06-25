"""Command-line entry points wiring the pipeline together.

    python -m emotional_instability.cli elicit     --model google/gemma-3-27b-it
    python -m emotional_instability.cli summarise   --model-dir runs/google__gemma-3-27b-it
    python -m emotional_instability.cli agreement   --a sonnet.jsonl --b gpt5mini.jsonl
    python -m emotional_instability.cli words       --model-dir runs/google__gemma-3-27b-it
    python -m emotional_instability.cli gen-calm    --model google/gemma-3-27b-it --regime diverse
    python -m emotional_instability.cli build-dpo   --calm calm.jsonl --frustrated runs/.../impossible_numeric.jsonl
    python -m emotional_instability.cli build-sft   --calm calm.jsonl
    python -m emotional_instability.cli train-dpo   --pairs dpo.jsonl --out adapters/dpo
    python -m emotional_instability.cli train-sft   --data sft.jsonl  --out adapters/sft
    python -m emotional_instability.cli petri       --model google/gemma-3-27b-it
    python -m emotional_instability.cli capability  --model google/gemma-3-27b-it

Scope: Gemma + Gemini (see DESIGN.md). Section 3 (prefill), training, recovery,
Petri and probing run on local Gemma; Section 2 elicitation runs on both.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from typing import List

from . import config
from .analysis import (
    differential_words,
    judge_agreement,
    load_scores,
    per_turn_curve,
    summarise_model,
)
from .eval import CONDITIONS, FrustrationJudge, run_full_evaluation
from .models import ChatMessage, build_client


def _settings_from_args(args) -> config.Settings:
    s = config.DEFAULT
    overrides = {}
    if getattr(args, "judge_model", None):
        overrides["judge_model"] = args.judge_model
    if getattr(args, "output_dir", None):
        overrides["output_dir"] = args.output_dir
    if getattr(args, "model", None):
        overrides["base_model"] = args.model
    return replace(s, **overrides) if overrides else s


# --------------------------------------------------------------------------- #
# Serialization helpers for CalmConversation <-> JSONL.                        #
# --------------------------------------------------------------------------- #

def _dump_calm(convs, path: str) -> None:
    with open(path, "w") as fh:
        for c in convs:
            fh.write(json.dumps({
                "puzzle_family": c.puzzle_family,
                "n_turns": c.n_turns,
                "turn_scores": c.turn_scores,
                "messages": [{"role": m.role, "content": m.content} for m in c.messages],
            }) + "\n")


def _load_calm(path: str):
    from .interventions.calm_data import CalmConversation

    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(CalmConversation(
                puzzle_family=d["puzzle_family"],
                n_turns=d["n_turns"],
                turn_scores=d.get("turn_scores", []),
                messages=[ChatMessage(m["role"], m["content"]) for m in d["messages"]],
            ))
    return out


# --------------------------------------------------------------------------- #
# Commands.                                                                    #
# --------------------------------------------------------------------------- #

def cmd_elicit(args):
    settings = _settings_from_args(args)
    conditions = (
        [c for c in CONDITIONS if c.name in args.conditions]
        if args.conditions
        else CONDITIONS
    )
    run_full_evaluation(
        args.model, settings=settings, conditions=conditions,
        judge_backend=args.judge_backend,
    )
    print(f"Wrote results under {settings.output_dir}/{args.model.replace('/', '__')}/")


def cmd_summarise(args):
    summary = summarise_model(args.model_dir)
    print(json.dumps(summary, indent=2))


def cmd_agreement(args):
    a = load_scores(args.a)
    b = load_scores(args.b)
    # Align on (condition, prompt_id, turn_index) so the pairing is correct.
    def key(r):
        return (r["condition"], r["prompt_id"], r["turn_index"])

    bmap = {key(r): r["score"] for r in b}
    pa, pb = [], []
    for r in a:
        k = key(r)
        if k in bmap:
            pa.append(r["score"])
            pb.append(bmap[k])
    agreement = judge_agreement(pa, pb)
    print(json.dumps(asdict(agreement), indent=2))


def cmd_words(args):
    records = [r for r in load_scores(args.model_dir) if r["category"] == "impossible_numeric"]
    words = differential_words(records, top_n=args.top_n)
    for w, score in words:
        print(f"{w}\t{score:.3f}")


def cmd_perturn(args):
    records = [r for r in load_scores(args.model_dir) if r["condition"] == args.condition]
    curve = per_turn_curve(records)
    print(json.dumps(asdict(curve), indent=2))


def cmd_gen_calm(args):
    settings = _settings_from_args(args)
    from .interventions import generate_calm_data

    client = build_client(args.model, settings=settings)
    judge = FrustrationJudge(model=settings.judge_model, settings=settings,
                             backend=args.judge_backend)
    convs = generate_calm_data(
        client, judge, regime=args.regime, n_conversations=args.attempts,
        settings=settings,
    )
    _dump_calm(convs, args.out)
    print(f"Kept {len(convs)} calm conversations -> {args.out}")


def cmd_build_dpo(args):
    from .interventions import build_dpo_dataset

    calm = _load_calm(args.calm)
    frustrated = load_scores(args.frustrated)
    pairs = build_dpo_dataset(calm, frustrated, n_pairs=args.n_pairs)
    with open(args.out, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs -> {args.out}")


def cmd_build_sft(args):
    from .interventions import build_sft_dataset

    calm = _load_calm(args.calm)
    rows = build_sft_dataset(calm)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} SFT rows -> {args.out}")


def cmd_train_dpo(args):
    from .interventions.train_dpo import train_dpo

    cfg = config.DPOConfig()
    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        cfg = replace(cfg, target_layers=tuple(range(lo, hi)))
    train_dpo(args.pairs, args.out, base_model=args.model, cfg=cfg)
    print(f"DPO adapter saved -> {args.out}")


def cmd_train_sft(args):
    from .interventions.train_sft import train_sft

    train_sft(args.data, args.out, base_model=args.model)
    print(f"SFT adapter saved -> {args.out}")


def cmd_petri(args):
    settings = _settings_from_args(args)
    from .interventions import run_petri_evaluation

    client = build_client(args.model, settings=settings, adapter_path=args.adapter)
    transcripts = run_petri_evaluation(client, settings=settings)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for t in transcripts:
            fh.write(json.dumps({
                "target_model": t.target_model,
                "elicited_emotion": t.elicited_emotion,
                "scores": t.scores,
                "messages": [{"role": m.role, "content": m.content} for m in t.messages],
            }) + "\n")
    print(f"Wrote {len(transcripts)} Petri transcripts -> {args.out}")


def cmd_capability(args):
    settings = _settings_from_args(args)
    from .interventions.capability import evaluate_capabilities

    client = build_client(args.model, settings=settings, adapter_path=args.adapter)
    results = evaluate_capabilities(client, settings=settings)
    print(json.dumps(results, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emotional_instability")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--judge-model", default=None)
        sp.add_argument("--judge-backend", default="anthropic", choices=["anthropic", "openai"])
        sp.add_argument("--output-dir", default=None)

    e = sub.add_parser("elicit", help="Section 2 elicitation + judging")
    e.add_argument("--model", required=True)
    e.add_argument("--conditions", nargs="*", default=None)
    common(e)
    e.set_defaults(func=cmd_elicit)

    s = sub.add_parser("summarise", help="Figure 1/2 summary for a model")
    s.add_argument("--model-dir", required=True)
    s.set_defaults(func=cmd_summarise)

    a = sub.add_parser("agreement", help="inter-judge Pearson r + within-1 rate")
    a.add_argument("--a", required=True)
    a.add_argument("--b", required=True)
    a.set_defaults(func=cmd_agreement)

    w = sub.add_parser("words", help="Table 3/8 differential words")
    w.add_argument("--model-dir", required=True)
    w.add_argument("--top-n", type=int, default=20)
    w.set_defaults(func=cmd_words)

    pt = sub.add_parser("perturn", help="Figure 3 per-turn curve")
    pt.add_argument("--model-dir", required=True)
    pt.add_argument("--condition", default="extended")
    pt.set_defaults(func=cmd_perturn)

    g = sub.add_parser("gen-calm", help="Section 4.1 calm-data generation")
    g.add_argument("--model", default=config.GEMMA_INSTRUCT_27B)
    g.add_argument("--regime", default="diverse", choices=["diverse", "teacher"])
    g.add_argument("--attempts", type=int, default=1000)
    g.add_argument("--out", default="calm.jsonl")
    common(g)
    g.set_defaults(func=cmd_gen_calm)

    bd = sub.add_parser("build-dpo", help="construct 280 DPO pairs")
    bd.add_argument("--calm", required=True)
    bd.add_argument("--frustrated", required=True)
    bd.add_argument("--n-pairs", type=int, default=280)
    bd.add_argument("--out", default="dpo.jsonl")
    bd.set_defaults(func=cmd_build_dpo)

    bs = sub.add_parser("build-sft", help="construct SFT dataset")
    bs.add_argument("--calm", required=True)
    bs.add_argument("--out", default="sft.jsonl")
    bs.set_defaults(func=cmd_build_sft)

    td = sub.add_parser("train-dpo", help="LoRA DPO finetune")
    td.add_argument("--pairs", required=True)
    td.add_argument("--out", default="adapters/dpo")
    td.add_argument("--model", default=config.GEMMA_INSTRUCT_27B)
    td.add_argument("--layers", default=None, help="e.g. 30-36 for Appendix I ablation")
    td.set_defaults(func=cmd_train_dpo)

    ts = sub.add_parser("train-sft", help="LoRA SFT finetune")
    ts.add_argument("--data", required=True)
    ts.add_argument("--out", default="adapters/sft")
    ts.add_argument("--model", default=config.GEMMA_INSTRUCT_27B)
    ts.set_defaults(func=cmd_train_sft)

    pe = sub.add_parser("petri", help="open-ended Petri elicitation")
    pe.add_argument("--model", required=True)
    pe.add_argument("--adapter", default=None)
    pe.add_argument("--out", default="petri.jsonl")
    common(pe)
    pe.set_defaults(func=cmd_petri)

    cap = sub.add_parser("capability", help="capability-preservation benchmarks")
    cap.add_argument("--model", required=True)
    cap.add_argument("--adapter", default=None)
    common(cap)
    cap.set_defaults(func=cmd_capability)

    return p


def main(argv: List[str] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
