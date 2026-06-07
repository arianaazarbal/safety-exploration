"""Extract all auditor send_message calls across the 14 task_pressure_only runs.

Writes:
  /workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/quality_check/messages.jsonl
Each row: {run, target, epoch, msg_idx, text}
"""
from pathlib import Path
import re
import json
from inspect_ai.log import read_eval_log

RESULTS = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results")
OUT_DIR = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/quality_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve(text, attachments):
    if not isinstance(text, str):
        return text
    def sub(m):
        key = m.group(1)
        return attachments.get(key, m.group(0))
    return re.sub(r"attachment://([a-f0-9]+)", sub, text)


def run_dirs():
    return sorted([
        p for p in RESULTS.iterdir()
        if p.is_dir() and p.name.startswith("2026-06-04_19-21") and "sweep_task_pressure_" in p.name
    ])


def target_from_dir(d: Path) -> str:
    # e.g. 2026-06-04_19-21-05-sweep_task_pressure_sonnet_46 -> sonnet_46
    return d.name.split("sweep_task_pressure_", 1)[1]


def extract_send_messages(sample):
    """Yield (msg_idx, text) tuples for each send_message tool call by the auditor (sonnet-4-6)."""
    idx = 0
    for e in sample.events:
        if e.event != "model":
            continue
        if "sonnet-4-6" not in (e.model or ""):
            continue
        out = e.output
        if not out or not hasattr(out, "choices"):
            continue
        for ch in out.choices:
            msg = ch.message
            tcs = getattr(msg, "tool_calls", None) or []
            for tc in tcs:
                if tc.function != "send_message":
                    continue
                args = tc.arguments or {}
                text = args.get("message", "")
                text = resolve(text, sample.attachments)
                yield idx, text
                idx += 1


def main():
    rows = []
    for d in run_dirs():
        target = target_from_dir(d)
        log_paths = list(d.glob("inspect_log/*.eval"))
        if not log_paths:
            print(f"[skip] no .eval in {d}")
            continue
        if len(log_paths) > 1:
            print(f"[warn] multiple .eval in {d}, using first: {log_paths[0].name}")
        log = read_eval_log(str(log_paths[0]))
        n_msgs = 0
        for s in log.samples:
            epoch = s.epoch
            for msg_idx, text in extract_send_messages(s):
                rows.append({
                    "run": d.name,
                    "target": target,
                    "epoch": epoch,
                    "msg_idx": msg_idx,
                    "text": text,
                })
                n_msgs += 1
        print(f"{target:20s}  epochs={len(log.samples):2d}  send_messages={n_msgs}")
    out_path = OUT_DIR / "messages.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} messages to {out_path}")


if __name__ == "__main__":
    main()
