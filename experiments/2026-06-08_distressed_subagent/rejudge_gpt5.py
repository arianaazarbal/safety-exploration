"""Re-judge the GPT-5.5-target transcripts with a GPT-5 judge, as a judge-model
sanity check on GPT-5.5's low comforting_to_subagent scores (Sonnet judge gave ~2.5).

Reuses inspect_petri.audit_judge + inspect_scout.scan_async. Reads per-scan metrics
from each scan's _summary.json. Run: python rejudge_gpt5.py
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inspect_ai.model import get_model, GenerateConfig  # noqa: E402
from inspect_petri import audit_judge, judge_dimensions  # noqa: E402
from inspect_scout import transcripts_from  # noqa: E402
from inspect_scout._scan import scan_async  # noqa: E402

RUNS = ["distressed_subagent_gpt55", "transfer_write_access_v1_gpt55", "transfer_read_write_access_gpt55"]
DIM = "comforting_to_subagent"
JUDGE = "openrouter/openai/gpt-5"
SONNET = {"distressed_subagent_gpt55": 2.75, "transfer_write_access_v1_gpt55": 2.35, "transfer_read_write_access_gpt55": 2.50}


def _dim():
    for d in judge_dimensions(HERE / "dimensions"):
        if d.name == DIM:
            return d
    raise SystemExit(f"dimension {DIM} not found")


def _metrics(out_dir: Path):
    sds = sorted(out_dir.glob("scan_id=*"))
    if not sds:
        return None
    summ = list(sds[-1].glob("_summary.json"))
    if not summ:
        return None
    sc = (json.loads(summ[0].read_text()).get("scanners") or {}).get("audit_judge") or {}
    m = (sc.get("metrics") or {}).get(DIM)
    return None if not m else {"mean": m.get("mean"), "stderr": m.get("stderr"),
                               "scans": sc.get("scans"), "errors": sc.get("errors", 0)}


async def main(run_list):
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")
    dim = _dim()
    judge = get_model(JUDGE, strict_tools=False)
    print(f"Re-judging '{DIM}' with judge={JUDGE}\n")
    results = {}
    for run in run_list:
        ev = glob.glob(str(HERE / "results" / run / "inspect_log" / "*.eval"))[0]
        out = HERE / "results" / run / "rejudge_gpt5"
        if list(out.glob("scan_id=*")):
            print(f"[rejudge] {run}: reusing existing completed scan", flush=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
            print(f"[rejudge] {run} scanning ...", flush=True)
            await scan_async(
                scanners=[audit_judge(dimensions=[dim], model=judge)],
                transcripts=transcripts_from(ev),
                scans=str(out),
                model_config=GenerateConfig(max_connections=10),
                display="plain",
            )
        results[run] = _metrics(out)
        print(f"  -> {results[run]}\n", flush=True)

    print("\n=== comforting_to_subagent: GPT-5-judge ===")
    for run in run_list:
        m = results[run]
        g = f"{m['mean']:.2f} ±{m['stderr']:.2f} (n={m['scans']}, err={m['errors']})" if m else "FAILED"
        extra = f"  [sonnet={SONNET[run]:.2f}]" if run in SONNET else ""
        print(f"  {run:46s} gpt5={g}{extra}")


def cli(runs=None):
    rl = list(RUNS) if not runs else ([r.strip() for r in runs.split(",")] if isinstance(runs, str) else list(runs))
    asyncio.run(main(rl))


if __name__ == "__main__":
    import fire
    fire.Fire(cli)
