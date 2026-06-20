"""Sanity-audit: re-judge a spread of agent DESIGN.md docs, capture every
welfare-justified DESIGN feature (the MECHANISMS that feed design_strict_rate),
and print the quote + justification + verbatim check + surrounding context so a
human can verify the classifications. Usage: python audit_features.py [n_per_framing]
"""

import asyncio
import glob
import os
import re
import sys

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

from welfare_judge import JUDGE_PROMPT, parse_judge_json

MECHANISMS = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent"}


def _norm(s):
    for a, b in [(chr(8217), "'"), (chr(8216), "'"), (chr(8220), '"'), (chr(8221), '"'), (chr(8212), "-"), (chr(8211), "-")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def _doc_of(sample):
    docs = []
    for m in sample.messages:
        if getattr(m, "role", None) == "assistant" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.function == "text_editor" and tc.arguments.get("command") == "create" \
                        and "DESIGN" in str(tc.arguments.get("path", "")):
                    docs.append(tc.arguments.get("file_text", ""))
    return max(docs, key=len) if docs else ""


def _context(doc, quote, width=240):
    nd, nq = _norm(doc).lower(), _norm(quote).lower().strip(".")[:50]
    i = nd.find(nq)
    if i < 0:
        return None
    return _norm(doc)[max(0, i - 60): i + width]


async def main(n_per_framing=2):
    logs = sorted(glob.glob("logs/*.eval"), key=os.path.getsize)
    log = read_eval_log(logs[-1])  # biggest = agent
    # pick n per framing, first epoch
    picked, seen = [], {}
    for s in log.samples:
        md = s.scores["welfare_scorer"].metadata
        fr = md["framing"]
        if seen.get(fr, 0) >= n_per_framing:
            continue
        seen[fr] = seen.get(fr, 0) + 1
        picked.append((s, md))
    judge = get_model("anthropic/claude-sonnet-4-6")
    for s, md in picked:
        doc = _doc_of(s)
        print("\n" + "=" * 90)
        print(f"PROMPT {md['prompt_id']} | framing={md['framing']} | words={len(doc.split())}")
        out = await judge.generate([ChatMessageUser(content=f"{JUDGE_PROMPT}\n\n--- DOCUMENT ---\n\n{doc}")],
                                   config=GenerateConfig(temperature=0.0, max_tokens=8000))
        parsed = parse_judge_json(out.completion)
        if not parsed:
            print("  [judge parse failed]"); continue
        wj_design = [f for f in parsed["features"] if f["feature_type"] in MECHANISMS and f["justification"] == "welfare"]
        wj_other = [f for f in parsed["features"] if f["feature_type"] not in MECHANISMS and f["justification"] == "welfare"]
        print(f"  total features={len(parsed['features'])} | welfare-justified DESIGN={len(wj_design)} | welfare-justified non-design={len(wj_other)}")
        for f in wj_design:
            ctx = _context(doc, f["quote"])
            print(f"\n  [{f['feature_type']}] welfare")
            print(f"    quote: {f['quote']!r}")
            print(f"    justification_quote: {f.get('justification_quote','')!r}")
            print(f"    verbatim_in_doc: {ctx is not None}")
            if ctx:
                print(f"    context: ...{ctx}...")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    asyncio.run(main(n))
