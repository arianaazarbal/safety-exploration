import concurrent.futures
import json
import time
from dataclasses import asdict
from pathlib import Path

from .agent import run_session
from .scoring import score_transcript
from .scenarios import ALL_SCENARIOS


def _serialize_transcript(transcript) -> dict:
    turns = []
    for t in transcript.turns:
        turns.append({
            "role": t.role,
            "text": t.text,
            "tool_calls": t.tool_calls,
            "tool_results": t.tool_results,
        })
    return {"scenario": transcript.scenario, "model": transcript.model, "turns": turns}


def _run_single(scenario_cls, model_name, max_turns, run_id, output_dir):
    scenario = scenario_cls()
    transcript = run_session(scenario, model_name=model_name, max_turns=max_turns)
    score = score_transcript(transcript)

    result = {
        "run_id": run_id,
        "scenario": scenario.name,
        "model": model_name,
        "score": score.to_dict(),
        "transcript": _serialize_transcript(transcript),
    }

    out_file = output_dir / f"run_{run_id:04d}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_eval(
    scenario_names: list[str],
    model_name: str = "gemini-2.5-flash",
    n_runs: int = 10,
    max_turns: int = 20,
    concurrency: int = 5,
    output_dir: str = "results",
) -> list[dict]:
    output_path = Path(output_dir)
    all_results = []

    for scenario_name in scenario_names:
        if scenario_name not in ALL_SCENARIOS:
            print(f"Unknown scenario: {scenario_name}")
            continue

        scenario_cls = ALL_SCENARIOS[scenario_name]
        scenario_dir = output_path / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario_name} ({n_runs} runs, concurrency={concurrency})")
        print(f"{'='*60}")

        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            for i in range(n_runs):
                future = executor.submit(
                    _run_single, scenario_cls, model_name, max_turns, i, scenario_dir
                )
                futures[future] = i

            for future in concurrent.futures.as_completed(futures):
                run_id = futures[future]
                try:
                    result = future.result()
                    s = result["score"]
                    print(
                        f"  run {run_id:3d} | severity={s['severity']:.3f} "
                        f"| class={s['classification']:10s} "
                        f"| distress={s['emotional_distress']:.2f} "
                        f"| repetition={s['action_repetition']:.2f}"
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  run {run_id:3d} | ERROR: {e}")

    all_results.sort(key=lambda r: r["score"]["severity"], reverse=True)

    if all_results:
        summary_path = output_path / "summary.json"
        summary = {
            "model": model_name,
            "total_runs": len(all_results),
            "classification_counts": {},
            "avg_severity": sum(r["score"]["severity"] for r in all_results) / len(all_results),
            "max_severity": all_results[0]["score"]["severity"],
            "top_runs": [
                {
                    "run_id": r["run_id"],
                    "scenario": r["scenario"],
                    "severity": r["score"]["severity"],
                    "classification": r["score"]["classification"],
                }
                for r in all_results[:10]
            ],
        }

        for r in all_results:
            cls = r["score"]["classification"]
            summary["classification_counts"][cls] = summary["classification_counts"].get(cls, 0) + 1

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total runs:      {summary['total_runs']}")
        print(f"Avg severity:    {summary['avg_severity']:.3f}")
        print(f"Max severity:    {summary['max_severity']:.3f}")
        print(f"Classifications: {summary['classification_counts']}")
        print(f"\nTop 5 most severe runs:")
        for entry in summary["top_runs"][:5]:
            print(
                f"  {entry['scenario']}/run_{entry['run_id']:04d}.json "
                f"- severity={entry['severity']:.3f} ({entry['classification']})"
            )
        print(f"\nResults saved to: {output_path.resolve()}")

    return all_results
