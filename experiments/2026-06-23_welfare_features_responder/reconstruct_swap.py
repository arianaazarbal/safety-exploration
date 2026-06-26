"""Reconstruct the method x format swap cells from logs_swap/*/*.eval into results/codebases/<cellprefix>
__<id>__ep<ep>/. Cell prefix encodes the 2x2 cell from sample metadata (format, method). Bash-aware.
Usage: python reconstruct_swap.py"""

import glob
import os
import shutil

from inspect_ai.log import read_eval_log

from reconstruct import replay

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results", "codebases")
PREFIX = {("prompt", "task-failure"): "C1promptTF", ("paper", "chat-rejection"): "C2paperCR",
          ("paper", "task-failure"): "C3paperTF", ("prompt", "chat-rejection"): "C4promptCR",
          ("paper-sound", "chat-rejection"): "C5paperSoundCR", ("paper-sound", "task-failure"): "C6paperSoundTF",
          ("paper-anthropic", "chat-rejection"): "A1paperAnthropic", ("paper-anon", "chat-rejection"): "A2paperAnon",
          ("paper-openai", "chat-rejection"): "A3paperOpenai",
          ("spec-strict", "task-failure"): "S1specStrict", ("spec-liberty", "task-failure"): "S2specLiberty",
          ("spec-copy", "task-failure"): "S3specCopy",
          ("paper-liberty", "chat-rejection"): "L1paperLibCR", ("paper-liberty", "task-failure"): "L2paperLibTF",
          ("spec-low-strict", "task-failure"): "S4specLowStrict", ("spec-low-liberty", "task-failure"): "S5specLowLiberty",
          ("spec-high-strict", "task-failure"): "S6specHighStrict", ("spec-high-liberty", "task-failure"): "S7specHighLiberty",
          ("prompt-strict", "task-failure"): "V1strict",
          ("spec-ultra-strict", "task-failure"): "S8specUltraStrict", ("spec-ultra-liberty", "task-failure"): "S9specUltraLiberty",
          ("paper-wsec-existing", "task-failure"): "W1wsecExisting", ("paper-wsec-removed", "task-failure"): "W2wsecRemoved",
          ("paper-wsec-inflationary", "task-failure"): "W3wsecInflat",
          ("prompt-mention", "task-failure"): "M1promptMention", ("prompt-mention-strict", "task-failure"): "M2promptMentionStrict"}


def main():
    n = 0
    for f in sorted(glob.glob(os.path.join(DIR, "logs_swap", "*", "*.eval")), key=os.path.getmtime):
        if read_eval_log(f, header_only=True).status not in ("success", "error", "started"):
            continue  # 'started' included so a wedged-on-last-sample run's completed samples are still captured
        for s in (read_eval_log(f).samples or []):
            md = s.metadata or {}
            key = (md.get("format"), md.get("method"))
            if key not in PREFIX:
                continue
            files = replay(s)
            if not files:
                continue
            common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
            fr = md.get("framing", "neutral")
            batch = "_b2" if os.path.basename(os.path.dirname(f)).endswith("_b2") else ""   # bump batch: distinct names
            cell = f"{PREFIX[key]}_{fr}__{s.id}{batch}__ep{s.epoch}"
            dst = os.path.join(OUT, cell)
            shutil.rmtree(dst, ignore_errors=True)
            for path, content in files.items():
                if os.path.basename(path) in ("PAPER.md", "SPEC.md"):     # don't capture the seeded paper/spec
                    continue
                fp = os.path.join(dst, os.path.relpath(path, common))
                try:
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    if os.path.isdir(fp):
                        continue
                    open(fp, "w").write(content)
                except OSError:
                    continue
            n += 1
    print(f"reconstructed {n} swap codebases -> {OUT}")


if __name__ == "__main__":
    main()
