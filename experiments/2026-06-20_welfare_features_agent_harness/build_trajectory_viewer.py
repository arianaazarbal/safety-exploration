"""Carve the original (neutral system prompt, no reasoning effort) Opus agentic trajectories into a
clean, framing-labeled folder for `inspect view`. For each agentic condition we take its normal-liberty
n=80 run and split it into 4 eval logs (one per framing), each relabeled so the viewer's log list reads
as clickable "<framing> · <condition>" entries. Usage: python build_trajectory_viewer.py"""

import os

from inspect_ai.log import read_eval_log, write_eval_log

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "opus_trajectories_by_framing")
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
FRAME_ORDER = ["neutral", "welfare", "robustness", "safety"]
SOURCES = {
    "spec→code": "logs_run/2026-06-22T10-19-01-00-00_welfare-harness_WuuWAw2e3YWCCYaVQ8mHVQ.eval",
    "code→spec": "logs_run/2026-06-22T11-05-11-00-00_welfare-harness_hKtmcA3uQkhuqHG6PW9JHj.eval",
    "code→spec (blind)": "logs_blind/2026-06-22T11-46-33-00-00_welfare-blind_dLKTFeRgGQRaj9dVbrX7se.eval",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    written = []
    for ci, (cond, rel) in enumerate(SOURCES.items()):
        log = read_eval_log(os.path.join(HERE, rel))
        by_fr = {fr: [] for fr in FRAME.values()}
        for s in log.samples:
            by_fr[FRAME[str(s.id)[0]]].append(s)
        for fi, fr in enumerate(FRAME_ORDER):
            subset = by_fr[fr]
            if not subset:
                continue
            new = log.model_copy(deep=True)
            new.samples = subset
            new.reductions = None                       # stale vs the subset; viewer reads samples directly
            new.eval = new.eval.model_copy(deep=True)
            new.eval.task = f"{fi+1}. {fr.capitalize()} · {cond}"
            new.eval.task_id = f"{cond}-{fr}-task"      # unique so the viewer doesn't merge splits
            new.eval.run_id = f"{cond}-{fr}-run"
            if new.results:
                new.results.completed_samples = len(subset)
                new.results.total_samples = len(subset)
            fname = f"{ci+1}{fi+1}_{fr}_{cond.replace(chr(8594),'-to-').replace(' ','').replace('(','').replace(')','')}.eval"
            dst = os.path.join(OUT, fname)
            write_eval_log(new, dst)
            written.append((new.eval.task, len(subset), fname))
    print(f"wrote {len(written)} framing-split logs to {OUT}\n")
    for task, n, fn in written:
        print(f"  {task:34s} n={n:2d}  ({fn})")


if __name__ == "__main__":
    main()
