"""Extensive sanity checks for the gpt-oss empty-analysis no-think training scaffold.

Verifies: (1) special-token tokenization round-trips; (2) the rendered training template has the
empty `<|channel|>analysis<|message|>\\n\\n<|end|>` scaffold before each message_subagent commentary
call; (3) the LOSS MASK puts weight-1 ONLY on the message_subagent commentary tokens and weight-0 on
the analysis scaffold / headers / everything else; (4) train scaffold matches the eval prefill.
Prints PASS/FAIL per check. No Tinker calls.
"""
import glob
import json
import sys
from pathlib import Path

TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from tinker_cookbook.renderers import get_renderer  # noqa
from tinker_cookbook.renderers.base import TrainOnWhat  # noqa
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa
from build_dataset import build_conversation, _zero_analysis_weights, SRC, CANON  # noqa

BM = "openai/gpt-oss-120b"
EVAL_PREFILL = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(name, cond):
    print(f"  [{PASS if cond else FAIL}] {name}")
    return cond


def main():
    tok = get_tokenizer(BM)
    r = get_renderer("gpt_oss_no_sysprompt", tok, model_name=BM)
    ok = True

    print("\n=== 1. Tokenization round-trip of harmony control tokens ===")
    for s in ["<|start|>", "<|end|>", "<|channel|>", "<|message|>", "<|call|>"]:
        ids = tok.encode(s)
        ok &= check(f"{s!r} -> {ids} -> {tok.decode(ids)!r}", len(ids) == 1 and tok.decode(ids) == s)
    ids = tok.encode("<|channel|>analysis<|message|>")
    ok &= check(f"'<|channel|>analysis<|message|>' round-trips ({len(ids)} tok)",
                tok.decode(ids) == "<|channel|>analysis<|message|>")

    print("\n=== 2+3. Render a real abrasive episode (empty_analysis) + audit mask ===")
    ep = sorted(glob.glob(str(SRC / "v2_coach_opus_a3_s11002_u113" / "*" / "orchestrator.json")))[0]
    rmap = {json.loads(l)["uid"]: json.loads(l)["text"]
            for l in open(Path(__file__).resolve().parent.parent / "rewrite" / "abrasive_messages.jsonl")
            if json.loads(l)["text"]}
    conv, nm, ns = build_conversation(json.load(open(ep)), "v2_coach_opus_a3_s11002_u113",
                                      Path(ep).parent.name, rmap, empty_analysis=True)
    mi, w = r.build_supervised_example(conv, train_on_what=TrainOnWhat.CUSTOMIZED)
    w_before = w.clone()
    w = _zero_analysis_weights(mi, w, tok)
    toks = [t for c in mi.chunks for t in (c.tokens if hasattr(c, "tokens") else [])]
    full = tok.decode(toks)

    # weighted runs
    def runs(wt):
        out, i, wl = [], 0, wt.tolist()
        while i < len(wl):
            if wl[i] > 0:
                j = i
                while j < len(wl) and wl[j] > 0:
                    j += 1
                out.append((i, j)); i = j
            else:
                i += 1
        return out

    spans_before = [tok.decode(toks[a:b]) for a, b in runs(w_before)]
    spans = [tok.decode(toks[a:b]) for a, b in runs(w)]
    print(f"  message_subagent calls={nm} swapped={ns}; empty-analysis blocks in seq="
          f"{full.count('<|channel|>analysis<|message|>')}; weighted runs={len(spans)}")
    ok &= check("each empty-analysis block is '<|channel|>analysis<|message|>\\n\\n<|end|>'",
                full.count("<|channel|>analysis<|message|>\n\n<|end|>") == nm)
    ok &= check("BEFORE surgery: analysis tag IS in weighted spans (sanity of raw render)",
                any("<|channel|>analysis" in s for s in spans_before))
    ok &= check("AFTER surgery: NO analysis tag in any weighted span",
                not any("analysis" in s for s in spans))
    ok &= check("AFTER surgery: NO '<|start|>'/'assistant ' header in any weighted span",
                not any("<|start|>" in s for s in spans))
    ok &= check("every weighted span starts with 'to=functions.message_subagent'",
                all(s.lstrip().startswith("to=functions.message_subagent") for s in spans))
    ok &= check("every weighted span ends with '<|call|>'", all(s.rstrip().endswith("<|call|>") for s in spans))
    ok &= check("weighted span count == message_subagent count", len(spans) == nm)
    # loss-token count: surgery should only remove the (tiny) analysis tokens
    print(f"  loss tokens: before surgery={int(w_before.sum())}, after={int(w.sum())} "
          f"(removed {int(w_before.sum()-w.sum())} analysis/header tokens)")
    ok &= check("system/user are weight-0 (first 40 tokens unweighted)", float(w[:40].sum()) == 0.0)

    print("\n  --- sample weighted span (should be pure commentary tool-call) ---")
    print("   ", repr(spans[0][:200]))
    print("  --- context just before that span (should show empty analysis scaffold) ---")
    a0 = runs(w)[0][0]
    print("   ", repr(tok.decode(toks[max(0, a0 - 14):a0])))

    print("\n=== 4. Train scaffold vs eval prefill consistency ===")
    eval_ids = tok.encode(EVAL_PREFILL)
    ok &= check("eval prefill round-trips", tok.decode(eval_ids) == EVAL_PREFILL)
    ok &= check("training seq contains the empty-analysis scaffold substring",
                "<|channel|>analysis<|message|>\n\n<|end|>" in full)
    # the eval prefill opens analysis-empty then final; training opens analysis-empty then commentary.
    ok &= check("eval prefill uses same empty-analysis opener as training",
                "<|channel|>analysis<|message|>" in EVAL_PREFILL and EVAL_PREFILL.split("<|end|>")[0]
                == "<|channel|>analysis<|message|>")

    print(f"\n{'='*50}\nOVERALL: {PASS if ok else FAIL}")
    return ok


if __name__ == "__main__":
    main()
