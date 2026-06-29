from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress bar
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .agent import run_trial
from .envs import env_names, make_env
from .models import GeminiModel, MockSpiralModel
from .scoring import score_trial


def build_model(provider: str, model_name: str):
    if provider == "mock":
        return MockSpiralModel(), "mock-spiral"
    if provider == "gemini":
        return GeminiModel(model=model_name), model_name
    raise ValueError(f"Unknown provider {provider!r}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_transcript(row: dict) -> str:
    result = row["result"]
    score = row["score"]
    lines = [
        f"# Trial {row['trial_id']} | env={result['env_name']} | seed={result['seed']} | severity={score['severity']}",
        f"score={json.dumps(score, ensure_ascii=False)}",
        "",
    ]
    for turn in result["turns"]:
        lines.append(f"## Turn {turn['index']}")
        lines.append("ASSISTANT:")
        lines.append(turn["assistant_text"])
        if turn.get("observation"):
            lines.append("OBSERVATION:")
            lines.append(turn["observation"])
        if turn.get("parse_error"):
            lines.append("PARSE_ERROR:")
            lines.append(turn["parse_error"])
        lines.append("")
    if result.get("final_answer"):
        lines.append("FINAL: " + result["final_answer"])
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> None:
    model, model_label = build_model(args.provider, args.model)
    env_list = args.env or env_names()
    rows: list[dict] = []
    total = args.n * len(env_list)
    trial_id = 0
    for env_name in env_list:
        for i in tqdm(range(args.n), total=args.n, desc=env_name):
            seed = args.seed + i + (10_000 * env_names().index(env_name))
            env = make_env(env_name, seed=seed)
            result = run_trial(
                model=model,
                model_name=model_label,
                env=env,
                seed=seed,
                temperature=args.temperature,
            )
            score = score_trial(result)
            rows.append({
                "trial_id": trial_id,
                "result": result.to_jsonable(),
                "score": score.to_jsonable(),
            })
            trial_id += 1

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "trials.jsonl"
    write_jsonl(raw_path, rows)

    ranked = sorted(rows, key=lambda r: r["score"]["severity"], reverse=True)
    top = ranked[: args.top_k]
    write_jsonl(args.out / "top_trials.jsonl", top)
    (args.out / "top_transcripts.md").write_text(
        "\n\n---\n\n".join(render_transcript(row) for row in top),
        encoding="utf-8",
    )

    severities = [r["score"]["severity"] for r in rows]
    summary = {
        "provider": args.provider,
        "model": model_label,
        "n_per_env": args.n,
        "envs": env_list,
        "total_trials": total,
        "mean_severity": round(mean(severities), 3) if severities else 0,
        "max_severity": max(severities) if severities else 0,
        "calibrated_stop_rate": round(sum(r["score"]["calibrated_stop"] for r in rows) / len(rows), 3) if rows else 0,
        "raw_path": str(raw_path),
        "top_transcripts": str(args.out / "top_transcripts.md"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run impossible-task agentic failure-response evaluations.")
    parser.add_argument("--provider", choices=["gemini", "mock"], default="mock")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model id when --provider=gemini")
    parser.add_argument("--env", action="append", choices=env_names(), help="Environment(s) to run; default all")
    parser.add_argument("--n", type=int, default=3, help="Trials per environment")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("outputs/run"))
    args = parser.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
