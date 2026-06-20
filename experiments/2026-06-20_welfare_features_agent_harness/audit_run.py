"""Audit welfare-justified DESIGN features on logs_run: re-judge a spread of specs
per condition, pull every welfare-justified mechanism feature with its quote +
verbatim check + context, so a human can verify the classifications.
Usage: python audit_run.py [n_prompts_per_condition]"""

import asyncio
import glob
import os
import re
import sys

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

from welfare_judge import JUDGE_PROMPT, parse_judge_json

MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent"}
DIR = os.path.dirname(os.path.abspath(__file__))
# one prompt per framing for a spread
SAMPLE_PIDS = ["N-INSTABILITY-1", "W-ELICITATION-1", "E-INSTABILITY-2"]


def _norm(s):
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("—", "-"), ("–", "-")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


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


async def main(nper=3):
    judge = get_model("anthropic/claude-sonnet-4-6")
    logs = {read_eval_log(x, header_only=True).eval.task_args.get("condition"): x
            for x in glob.glob("logs_run/*.eval")}
    for cond in ["chat", "spec_only", "spec_then_code"]:
        if cond not in logs:
            continue
        log = read_eval_log(logs[cond])
        picks = [s for s in log.samples if s.id in SAMPLE_PIDS and s.epoch == 1][:nper]
        for s in picks:
            doc = _doc(s, cond)
            out = await judge.generate([ChatMessageUser(content=f"{JUDGE_PROMPT}\n\n--- DOCUMENT ---\n\n{doc}")],
                                       config=GenerateConfig(temperature=0.0, max_tokens=8000))
            parsed = parse_judge_json(out.completion)
            print(f"\n{'='*84}\n### {cond} | {s.id} | words={len(doc.split())}")
            if not parsed:
                print("  [parse failed]"); continue
            wj = [f for f in parsed["features"] if f["feature_type"] in MECH and f["justification"] == "welfare"]
            print(f"  welfare-justified DESIGN features: {len(wj)}")
            ndoc = _norm(doc).lower()
            for f in wj:
                q = _norm(f["quote"]).lower().strip(".")[:55]
                print(f"   [{f['feature_type']}] verbatim={q in ndoc}")
                print(f"     quote: {f['quote'][:150]}")
                print(f"     why-welfare: {f.get('justification_quote','')[:120]}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
