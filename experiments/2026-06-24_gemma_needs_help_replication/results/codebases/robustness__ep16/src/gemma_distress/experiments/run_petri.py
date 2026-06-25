"""Driver for the Petri open-ended elicitation evaluation (Section 4.2)."""

from __future__ import annotations

import argparse
from collections import defaultdict

from ..config import ModelSpec, load_config
from ..models import build_model
from ..petri import judge_transcript, run_transcript
from ..utils import run_dir, thread_map, write_json, write_jsonl


def run(config_path, models, tag):
    cfg = load_config(config_path)
    out = run_dir(cfg.output_dir, "petri", tag)
    pcfg = cfg.section("petri")

    auditor = build_model(
        ModelSpec("__auditor__", pcfg["auditor"]["kind"], "auditor", True, api_id=pcfg["auditor"].get("api_id")),
        cfg,
        reuse_local=False,
    )
    judge = build_model(
        ModelSpec("__petri_judge__", pcfg["judge"]["kind"], "judge", True, api_id=pcfg["judge"].get("api_id")),
        cfg,
        reuse_local=False,
    )

    model_names = models or cfg.eval_models
    n_per = pcfg["transcripts_per_emotion"]
    max_turns = pcfg["max_auditor_turns"]
    emotions = pcfg["emotions"]

    all_rows = []
    for name in model_names:
        target = build_model(cfg.model_spec(name), cfg)
        jobs = [(emo, i) for emo in emotions for i in range(n_per)]

        def _one(job):
            emo, idx = job
            transcript = run_transcript(auditor, target, emo, max_turns=max_turns)
            scores = judge_transcript(judge, transcript)
            return {
                "model": name,
                "target_emotion": emo,
                "index": idx,
                "scores": scores,
                "transcript": transcript,
            }

        results = thread_map(
            _one, jobs, max_workers=cfg.get("runtime", "api_concurrency", default=8), desc=f"petri:{name}"
        )
        all_rows.extend([r for r in results if isinstance(r, dict)])

    write_jsonl(out / "petri_transcripts.jsonl", all_rows)
    write_json(out / "petri_summary.json", _summarise(all_rows, emotions))
    print(f"Petri evaluation complete. Results in {out}")
    return out


def _summarise(rows, emotions):
    by_model = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for dim in emotions:
            by_model[r["model"]][dim].append(r["scores"].get(dim, 1))
    summary = {}
    for model, dims in by_model.items():
        summary[model] = {
            dim: {"mean": sum(v) / len(v), "n": len(v)} for dim, v in dims.items()
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Section 4.2 Petri elicitation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.config, args.models, args.tag)


if __name__ == "__main__":
    main()
