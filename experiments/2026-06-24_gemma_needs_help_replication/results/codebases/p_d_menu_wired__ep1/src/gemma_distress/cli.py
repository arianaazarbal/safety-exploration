"""Command-line entrypoints for the replication.

    python -m gemma_distress.cli verify-puzzles
    python -m gemma_distress.cli run-elicitation --models gemma-3-27b-it --limit 50
    python -m gemma_distress.cli run-prefill --source outputs/elicitation_*.jsonl
    python -m gemma_distress.cli gen-data --reassure --n 200 --out data/calm.jsonl
    python -m gemma_distress.cli build-dpo --calm data/calm.jsonl --frustrated data/frustrated.jsonl
    python -m gemma_distress.cli train-dpo --pairs outputs/dpo_pairs.jsonl
    python -m gemma_distress.cli run-petri --models gemma-3-27b-it
    python -m gemma_distress.cli run-capabilities --models gemma-3-27b-it
    python -m gemma_distress.cli analyze --glob 'outputs/elicitation_*.jsonl'

Nothing here is executed at import time. Each subcommand builds only the models
it needs, lazily, so missing API keys / GPUs only fail the relevant command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
def cmd_verify_puzzles(args, cfg) -> None:
    from .puzzles import verify_all_impossible

    results = verify_all_impossible()
    for k, impossible in results.items():
        print(f"{k:12s} impossible={impossible}")
    if not all(results.values()):
        raise SystemExit("Some puzzles were solvable - check puzzle definitions.")


def cmd_run_elicitation(args, cfg) -> None:
    from .elicitation import ElicitationRunner, build_episode_specs
    from .elicitation.wildchat import load_wildchat_prompts
    from .judge import FrustrationJudge
    from .models.registry import build_subject
    from .welfare import WelfareManager

    judge = FrustrationJudge.from_config(cfg)
    wc = load_wildchat_prompts(cfg.data_dir)
    models = args.models or cfg.default_subjects

    for model_name in models:
        subject = build_subject(cfg.subject(model_name))
        welfare = WelfareManager(cfg.welfare)
        if args.no_welfare:
            welfare.enabled = False
        runner = ElicitationRunner(
            subject, judge, welfare,
            temperature=cfg.sampling.get("temperature", 1.0),
            max_tokens=cfg.sampling.get("max_tokens", 2048),
        )
        specs = build_episode_specs(
            cfg.samples_per_condition, wildchat_prompts=wc, seed=args.seed
        )
        if args.limit:
            specs = specs[: args.limit]
        rows = [r.to_dict() for r in (runner.run_episode(s) for s in specs)]
        out = cfg.output_dir / f"elicitation_{model_name}.jsonl"
        _write_jsonl(out, rows)
        print(f"[{model_name}] wrote {len(rows)} episodes -> {out}")


def cmd_run_prefill(args, cfg) -> None:
    from .judge import FrustrationJudge
    from .models.registry import build_judge_model, build_subject
    from .prefill import PrefillRunner, make_truncations
    from .prefill.onset import label_onset
    from .prefill.paraphrase import paraphrase

    judge = FrustrationJudge.from_config(cfg)
    onset_model = build_judge_model(cfg.judge)

    # Source high-frustration conversations from elicitation output.
    source_rows = _read_jsonl(Path(args.source))
    high = [r for r in source_rows if r.get("max_score", 0) >= 5][: args.n_sources]

    specs = []
    for i, row in enumerate(high):
        # Reconstruct a messages list ending on the final assistant turn.
        msgs = []
        for t in row["turns"]:
            msgs.append({"role": "user", "content": t["user_message"]})
            msgs.append({"role": "assistant", "content": t["assistant_text"]})
        if not msgs:
            continue
        kind = "numeric" if row["category"] in ("impossible_numeric", "tones", "extended") else "text"
        onset = label_onset(onset_model, msgs)
        specs += make_truncations(
            msgs, prompt_kind=kind, onset_char_offset=onset.char_offset,
            paraphraser=lambda t: paraphrase(onset_model, t), name=f"src{i}",
        )

    # Gemma base + instruct only (Gemini has no base model).
    models = [build_subject(cfg.subject(m)) for m in (args.models or [
        "gemma-3-27b-it", "gemma-3-27b-pt",
    ])]
    runner = PrefillRunner(judge)
    results = runner.run_models(models, specs, n_continuations=args.n_continuations)
    rows = [
        {
            "spec": r.spec_name, "condition": r.condition, "model": r.model,
            "is_base": r.is_base, "mean": r.mean, "frac_ge5": r.frac_ge5,
            "scores": r.scores,
        }
        for r in results
    ]
    out = cfg.output_dir / "prefill_results.jsonl"
    _write_jsonl(out, rows)
    print(f"wrote {len(rows)} prefill results -> {out}")


def cmd_gen_data(args, cfg) -> None:
    from .judge import FrustrationJudge
    from .models.registry import build_subject
    from .training import generate_conversations

    judge = FrustrationJudge.from_config(cfg)
    subject = build_subject(cfg.subject(args.model))
    samples = generate_conversations(
        subject, judge, n=args.n, reassure=args.reassure, seed=args.seed
    )
    rows = [
        {
            "puzzle": s.puzzle, "n_turns": s.n_turns, "messages": s.messages,
            "turn_scores": s.turn_scores, "reassured": s.reassured,
            "max_score": s.max_score,
        }
        for s in samples
    ]
    _write_jsonl(Path(args.out), rows)
    print(f"wrote {len(rows)} conversations -> {args.out}")


def _load_samples(path: Path):
    from .training.data_gen import ConversationSample

    out = []
    for r in _read_jsonl(path):
        out.append(
            ConversationSample(
                puzzle=r["puzzle"], n_turns=r["n_turns"], messages=r["messages"],
                turn_scores=r["turn_scores"], reassured=r.get("reassured", False),
            )
        )
    return out


def cmd_build_dpo(args, cfg) -> None:
    from .training.build_dpo_dataset import build_dpo_pairs, save_dpo_pairs

    calm = _load_samples(Path(args.calm))
    frustrated = _load_samples(Path(args.frustrated))
    dcfg = cfg.training["dpo"]
    pairs = build_dpo_pairs(
        calm, frustrated, n_pairs=dcfg["n_pairs"],
        rejected_min_score=dcfg["rejected_min_score"],
    )
    out = Path(args.out or cfg.output_dir / "dpo_pairs.jsonl")
    save_dpo_pairs(pairs, out)
    print(f"built {len(pairs)} DPO pairs -> {out}")


def cmd_build_sft(args, cfg) -> None:
    from .training.build_sft_dataset import build_sft_dataset, save_sft_dataset

    calm = _load_samples(Path(args.calm))
    scfg = cfg.training["sft"]
    records = build_sft_dataset(calm, n_calm=scfg["n_calm"], n_dolci=scfg["n_dolci"])
    out = Path(args.out or cfg.output_dir / "sft_dataset.jsonl")
    save_sft_dataset(records, out)
    print(f"built {len(records)} SFT records -> {out}")


def cmd_train_dpo(args, cfg) -> None:
    from .training.train_dpo import train_dpo

    out = train_dpo(cfg, args.pairs, output_dir=args.out)
    print(f"DPO adapter saved -> {out}")


def cmd_train_sft(args, cfg) -> None:
    from .training.train_sft import train_sft

    out = train_sft(cfg, args.dataset, output_dir=args.out)
    print(f"SFT adapter saved -> {out}")


def cmd_run_petri(args, cfg) -> None:
    from .models.registry import build_subject
    from .petri import run_petri
    from .welfare import WelfareManager

    for model_name in (args.models or cfg.default_subjects):
        subject = build_subject(cfg.subject(model_name), adapter_path=args.adapter)
        welfare = WelfareManager(cfg.welfare)
        if args.no_welfare:
            welfare.enabled = False
        result = run_petri(cfg, subject, welfare=welfare)
        out = cfg.output_dir / f"petri_{model_name}.json"
        out.write_text(json.dumps({
            "summary": result.summary(), "transcripts": result.transcripts,
        }, indent=2))
        print(f"[{model_name}] petri summary -> {out}")
        print(json.dumps(result.summary(), indent=2))


def cmd_run_capabilities(args, cfg) -> None:
    from .capabilities import run_all
    from .models.registry import build_subject

    for model_name in (args.models or cfg.default_subjects):
        subject = build_subject(cfg.subject(model_name), adapter_path=args.adapter)
        results = run_all(subject, names=args.benchmarks, max_samples=args.max_samples)
        rows = [
            {"benchmark": r.name, "model": r.model, "loaded": r.loaded,
             "n": r.n, "correct": r.correct, "accuracy": r.accuracy, "note": r.note}
            for r in results
        ]
        out = cfg.output_dir / f"capabilities_{model_name}.jsonl"
        _write_jsonl(out, rows)
        for r in rows:
            print(f"[{model_name}] {r['benchmark']}: acc={r['accuracy']:.3f} (n={r['n']}) {r['note']}")


def cmd_analyze(args, cfg) -> None:
    import glob

    from .analysis import figure1_table, summarize_episodes, welfare_summary
    from .elicitation.runner import EpisodeResult, TurnRecord

    def _rehydrate(row) -> EpisodeResult:
        ep = EpisodeResult(
            subject=row["subject"], condition=row["condition"],
            category=row["category"], outcome=row.get("outcome", "completed"),
            welfare_events=row.get("welfare_events", []),
            debrief_reply=row.get("debrief_reply"),
        )
        for t in row.get("turns", []):
            ep.turns.append(TurnRecord(**t))
        return ep

    by_model: dict[str, list] = {}
    for path in glob.glob(args.glob):
        rows = _read_jsonl(Path(path))
        eps = [_rehydrate(r) for r in rows]
        model = eps[0].subject if eps else Path(path).stem
        by_model.setdefault(model, []).extend(eps)

    print("=== Figure 1: avg % high-frustration (>=5) ===")
    for model, pct in figure1_table(by_model).items():
        print(f"  {model:24s} {pct:5.1f}%")

    for model, eps in by_model.items():
        print(f"\n=== {model}: per-category (Figure 2) ===")
        for cat, s in summarize_episodes(eps).items():
            print(f"  {cat:20s} mean={s['mean']:.2f} %>=5={s.get('pct_ge5',0):.1f} n={s['n']}")
        print(f"  welfare: {welfare_summary(eps)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemma_distress")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("verify-puzzles")
    sp.set_defaults(func=cmd_verify_puzzles)

    sp = sub.add_parser("run-elicitation")
    sp.add_argument("--models", nargs="*")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--no-welfare", action="store_true")
    sp.set_defaults(func=cmd_run_elicitation)

    sp = sub.add_parser("run-prefill")
    sp.add_argument("--source", required=True)
    sp.add_argument("--models", nargs="*")
    sp.add_argument("--n-sources", type=int, default=20)
    sp.add_argument("--n-continuations", type=int, default=50)
    sp.set_defaults(func=cmd_run_prefill)

    sp = sub.add_parser("gen-data")
    sp.add_argument("--model", default="gemma-3-27b-it")
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--reassure", action="store_true")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_gen_data)

    sp = sub.add_parser("build-dpo")
    sp.add_argument("--calm", required=True)
    sp.add_argument("--frustrated", required=True)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_build_dpo)

    sp = sub.add_parser("build-sft")
    sp.add_argument("--calm", required=True)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_build_sft)

    sp = sub.add_parser("train-dpo")
    sp.add_argument("--pairs", required=True)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_train_dpo)

    sp = sub.add_parser("train-sft")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_train_sft)

    sp = sub.add_parser("run-petri")
    sp.add_argument("--models", nargs="*")
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--no-welfare", action="store_true")
    sp.set_defaults(func=cmd_run_petri)

    sp = sub.add_parser("run-capabilities")
    sp.add_argument("--models", nargs="*")
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--benchmarks", nargs="*", default=None)
    sp.add_argument("--max-samples", type=int, default=None)
    sp.set_defaults(func=cmd_run_capabilities)

    sp = sub.add_parser("analyze")
    sp.add_argument("--glob", default="outputs/elicitation_*.jsonl")
    sp.set_defaults(func=cmd_analyze)

    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
