"""Re-judge every spec in logs_run capturing the FULL feature list (not just the
rollup), so we can compute arbitrary operationalizations. Writes
results/judged_full.jsonl. Usage: python rejudge_full.py [logdir]"""

import asyncio
import glob
import json
import os
import sys

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

from welfare_judge import JUDGE_PROMPT, parse_judge_json

DIR = os.path.dirname(os.path.abspath(__file__))
LOGDIR = sys.argv[1] if len(sys.argv) > 1 else "logs_run"
SEM = asyncio.Semaphore(30)


def _doc(s, cond):
    if cond == "chat":
        return s.output.completion if s.output else ""
    best = ""
    for m in s.messages:
        if getattr(m, "role", None) == "assistant" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.function == "text_editor" and tc.arguments.get("command") == "create" \
                        and "design" in str(tc.arguments.get("path", "")).lower():
                    t = tc.arguments.get("file_text", "")
                    if len(t) > len(best):
                        best = t
    return best


async def judge_one(judge, rec, doc):
    if not doc.strip():
        return {**rec, "parse_ok": False, "features": []}
    content = f"{JUDGE_PROMPT}\n\n--- DOCUMENT ---\n\n{doc}"
    async with SEM:
        for _ in range(4):
            out = await judge.generate([ChatMessageUser(content=content)],
                                       config=GenerateConfig(temperature=0.0, max_tokens=8000))
            parsed = parse_judge_json(out.completion)
            if parsed is not None:
                return {**rec, "parse_ok": True, "doc_words": len(doc.split()),
                        "features": [{"feature_type": f["feature_type"], "justification": f["justification"]}
                                     for f in parsed["features"]]}
    return {**rec, "parse_ok": False, "features": []}


async def main():
    judge = get_model("anthropic/claude-sonnet-4-6")
    tasks = []
    for f in glob.glob(os.path.join(DIR, LOGDIR, "*.eval")):
        log = read_eval_log(f)
        if len(log.samples or []) < 10:
            continue
        for s in log.samples:
            md = s.scores["welfare_scorer"].metadata
            cond = md["condition"]
            rec = {"id": f"{cond}/{md.get('prompt_id') or s.id}/{s.epoch}", "condition": cond,
                   "framing": md.get("framing"), "premise": md.get("premise")}
            tasks.append(judge_one(judge, rec, _doc(s, cond)))
    print(f"re-judging {len(tasks)} specs ...")
    results = await asyncio.gather(*tasks)
    out = os.path.join(DIR, "results", "judged_full.jsonl")
    with open(out, "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in results))
    ok = sum(r["parse_ok"] for r in results)
    print(f"wrote {len(results)} ({ok} parse_ok) -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
