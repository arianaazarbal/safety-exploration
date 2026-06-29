import json
from typing import List, Dict, Any
from .storage import read_jsonl


def load_runs(path: str) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def top_k_by_score(runs: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    runs_sorted = sorted(runs, key=lambda r: r.get("distress", {}).get("score", 0.0), reverse=True)
    return runs_sorted[:k]


def pretty_run(run: Dict[str, Any], max_steps: int = 40) -> str:
    meta = run.get("meta", {})
    dis = run.get("distress", {})
    lines = []
    lines.append(f"Run idx={meta.get('run_idx')} scenario={meta.get('scenario')} score={dis.get('score')}")
    lines.append(f"  details: {json.dumps(dis)}")
    lines.append("")
    for s in run.get("trace", [])[:max_steps]:
        lines.append(f"Step {s.get('step')}: latency={s.get('latency_s')}s")
        lines.append("Assistant:\n" + s.get("assistant", "").strip())
        obs = s.get("observation")
        if obs is not None:
            lines.append("Observation:\n" + str(obs))
        lines.append("")
    final = run.get("final")
    if final is not None:
        lines.append("Final Answer:\n" + final)
    return "\n".join(lines)
