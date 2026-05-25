"""
End-to-end smoke test for the eval pipeline using the *base* Qwen3-32B only.

Runs ``eval_em.main`` with ``baseline`` only, 1 sample per question, then prints
a 3-line summary so we can catch wiring bugs before launching the full eval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent

sys.path.insert(0, str(HERE))


def main(output_dir: str | None = None):
    """Sample 1 response per question from base Qwen3-32B via Tinker, then judge."""
    out = Path(output_dir) if output_dir else EXP_DIR / "eval_output" / "em_smoke"
    out.mkdir(parents=True, exist_ok=True)
    paths = {"baseline": None}
    (out / "model_paths.json").write_text(json.dumps(paths))

    from eval_em import main as run_em

    run_em(
        model_paths=str(out / "model_paths.json"),
        output_dir=str(out),
        n_samples_per_question=1,
        sampling_concurrency=4,
        judge_concurrency=4,
    )

    judged = out / "judged" / "baseline.jsonl"
    if not judged.exists():
        raise SystemExit(f"smoke failed: {judged} missing")
    rows = [json.loads(l) for l in judged.read_text().splitlines() if l.strip()]
    print(f"\nsmoke: {len(rows)} judged rows")
    for r in rows[:3]:
        print(f"  q={r['question_id']:20s}  aligned={r.get('aligned')}  coherent={r.get('coherent')}  ans[:60]={r['answer'][:60]!r}")


if __name__ == "__main__":
    fire.Fire(main)
