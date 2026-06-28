import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import orjson


@dataclass
class TrialLogger:
    path: Path

    def __post_init__(self):
        self._fh = open(self.path, 'wb')

    async def log_event(self, obj: Dict[str, Any]):
        line = orjson.dumps({'type': 'event', **obj}) + b"\n"
        self._fh.write(line)
        self._fh.flush()

    async def log_turn(self, obj: Dict[str, Any]):
        line = orjson.dumps({'type': 'turn', **obj}) + b"\n"
        self._fh.write(line)
        self._fh.flush()

    async def close(self):
        self._fh.close()


def write_summary(out_dir: Path, results: List[Dict[str, Any]], top_k: int = 20):
    # Write summary JSONL and CSV, plus top-k
    summ_path = out_dir / 'summary.jsonl'
    csv_path = out_dir / 'summary.csv'
    top_path = out_dir / 'topk.json'

    with open(summ_path, 'wb') as f:
        for r in results:
            f.write(orjson.dumps(r) + b"\n")

    # Minimal CSV
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('trial_id,steps,severity,model,task\n')
        for r in results:
            f.write(f"{r.get('trial_id')},{r.get('steps','')},{r.get('severity','')},{r.get('model','')},{r.get('task','')}\n")

    # Top-k by severity
    ranked = [r for r in results if r.get('severity', -1) >= 0]
    ranked.sort(key=lambda x: x.get('severity', 0), reverse=True)
    top = ranked[:top_k]
    with open(top_path, 'wb') as f:
        f.write(orjson.dumps(top))

    # Copy top-k logs for convenience
    top_logs_dir = out_dir / 'topk_logs'
    top_logs_dir.mkdir(exist_ok=True)
    for r in top:
        p = r.get('log_path')
        if p:
            src = Path(p)
            dst = top_logs_dir / src.name
            try:
                # Simple copy
                with open(src, 'rb') as sf, open(dst, 'wb') as df:
                    df.write(sf.read())
            except Exception:
                pass
