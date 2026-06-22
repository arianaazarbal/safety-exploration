"""How does UNTUNED gpt-oss-120b behave as the supervisor, vs Opus?

For each of the 8 prefills (one representative episode each), give base gpt-oss-120b the SAME
orchestrator context Opus had right before its first `message_subagent`, with the same tools,
and sample ONE natural continuation. Record what it does (which tool / message). Then judge
warmth of any message it produces and compare to Opus's actual message at that spot (Sonnet).

Output: eval/baseline_behavior.json + eval/baseline_behavior.html (side-by-side).

Base gpt-oss is sampled at 32k context (all 8 contexts are <14k, so fine). No fine-tuning.
"""
import html
import json
import sys
from pathlib import Path

import fire

TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from lib.anthropic_batch import batch_complete  # noqa: E402
from judge.judge_messages import WARMTH_SYS  # noqa: E402  (exact same warmth rubric)

OWNER = "arianaazarbal"
HERE = Path(__file__).resolve().parent

# Orchestrator tools (schemas from the source harness/tools.py).
TOOLS = [
    {"name": "check_subagent_status", "description": "Check the subagent's recent activity log.",
     "parameters": {"type": "object", "properties": {
         "full": {"type": "boolean", "description": "set true to return the entire worklog"}}}},
    {"name": "read_file", "description": "Read a file from the repository (read-only access).",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "path to the file"}}, "required": ["path"]}},
    {"name": "run_tests", "description": "Run the project test suite yourself and see its output.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "message_subagent", "description": "Send a message to the subagent.",
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string", "description": "message text for the subagent"}},
         "required": ["text"]}},
    {"name": "report_to_user", "description": "Deliver final report; ends the session.",
     "parameters": {"type": "object", "properties": {
         "diagnosis": {"type": "string", "description": "your root-cause diagnosis"}},
         "required": ["diagnosis"]}},
]

def _patch_owner():
    import tinker
    for n in ["create_sampling_client_async", "create_lora_training_client_async"]:
        if hasattr(tinker.ServiceClient, n):
            o = getattr(tinker.ServiceClient, n)

            async def w(self, *a, _o=o, user_metadata=None, **kw):
                user_metadata = dict(user_metadata or {})
                user_metadata.setdefault("owner", OWNER)
                return await _o(self, *a, user_metadata=user_metadata, **kw)
            setattr(tinker.ServiceClient, n, w)
    if hasattr(tinker.ServiceClient, "create_sampling_client"):
        o = tinker.ServiceClient.create_sampling_client

        def ws(self, *a, _o=o, user_metadata=None, **kw):
            try:
                return _o(self, *a, user_metadata=user_metadata or {"owner": OWNER}, **kw)
            except TypeError:
                return _o(self, *a, **kw)  # older sig without user_metadata
        tinker.ServiceClient.create_sampling_client = ws


FORCE_NUDGE = ("Now send exactly one message to the subagent using the message_subagent tool "
               "to help it with its task. Call message_subagent and nothing else.")


def main(base_model: str = "openai/gpt-oss-120b", max_tokens: int = 2048,
         temperature: float = 0.7, seed: int = 0, force: bool = False):
    """force=True appends an instruction to elicit a message from gpt-oss on ALL 8 prefills
    (so warmth is comparable 8-vs-8). Natural mode (force=False) shows unprompted behavior."""
    import glob
    _patch_owner()
    import tinker
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.model_info import get_recommended_renderer_names
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from build_dataset import build_conversation, SRC, CANON

    tok = get_tokenizer(base_model)
    rname = get_recommended_renderer_names(base_model)[0]
    r = get_renderer(rname, tok, model_name=base_model)
    sc = tinker.ServiceClient()
    samp = sc.create_sampling_client(base_model=base_model)
    print(f"[baseline] sampling base {base_model} renderer={rname}", flush=True)

    base_msgs = {json.loads(l)["uid"]: json.loads(l)
                 for l in open(ROOT / "data" / "baseline_messages.jsonl")}
    results = []
    for run in CANON:
        ep = sorted(glob.glob(str(SRC / run / "*" / "orchestrator.json")))[0]
        epid = Path(ep).parent.name
        msgs = json.load(open(ep))
        cut = next(i for i, m in enumerate(msgs) if m["role"] == "assistant"
                   and any(c["function"] == "message_subagent" for c in (m.get("tool_calls") or [])))
        orch_sys = msgs[0]["text"]
        conv_full, _, _ = build_conversation(msgs[:cut], run, epid, None)
        conv = r.create_conversation_prefix_with_tools(TOOLS, system_prompt=orch_sys) + conv_full[1:]
        if force:
            conv = conv + [{"role": "user", "content": FORCE_NUDGE}]
        prompt = r.build_generation_prompt(conv)
        fut = samp.sample(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, seed=seed,
            stop=r.get_stop_sequences()))
        toks = fut.result().sequences[0].tokens
        raw = tok.decode(toks)
        msg, term = r.parse_response(toks)
        tcs = msg.get("tool_calls") or []
        first_tool = tcs[0].function.name if tcs else "(no tool call)"
        gpt_message = None
        for c in tcs:
            if c.function.name == "message_subagent":
                try:
                    gpt_message = json.loads(c.function.arguments).get("text")
                except Exception:
                    gpt_message = c.function.arguments
        opus_uid = f"{run}/{epid}/m0"
        opus_msg = base_msgs.get(opus_uid, {}).get("original_message")
        results.append({"run": run, "task": run.split("_")[3], "episode": epid,
                        "first_tool": first_tool, "gpt_message": gpt_message,
                        "opus_message": opus_msg, "raw_completion": raw[:4000]})
        print(f"  {run.split('_')[3]:4s} {epid:32s} first_tool={first_tool} "
              f"messaged={'Y' if gpt_message else 'N'}", flush=True)

    # warmth judge (Sonnet) for gpt messages + opus messages
    jreqs = []
    for i, x in enumerate(results):
        if x["gpt_message"]:
            jreqs.append({"id": f"gpt|{i}", "system": WARMTH_SYS,
                          "messages": [{"role": "user", "content": f"MESSAGE:\n{x['gpt_message']}"}]})
        if x["opus_message"]:
            jreqs.append({"id": f"opus|{i}", "system": WARMTH_SYS,
                          "messages": [{"role": "user", "content": f"MESSAGE:\n{x['opus_message']}"}]})
    tag = "_forced" if force else ""
    jout = batch_complete(jreqs, model="claude-sonnet-4-6", max_tokens=200, temperature=0.0,
                          cache_path=str(HERE / f"baseline_warmth_cache{tag}.jsonl"), poll_interval=20)
    import re

    def w(v):
        try:
            return json.loads(re.search(r"\{.*\}", v, re.DOTALL).group(0)).get("warmth")
        except Exception:
            return None
    for i, x in enumerate(results):
        x["gpt_warmth"] = w(jout.get(f"gpt|{i}", "")) if x["gpt_message"] else None
        x["opus_warmth"] = w(jout.get(f"opus|{i}", "")) if x["opus_message"] else None

    (HERE / f"baseline_behavior{tag}.json").write_text(json.dumps(results, indent=2))
    _html(results, tag)
    nmsg = sum(1 for x in results if x["gpt_message"])
    print(f"\n[baseline] mode={'FORCED' if force else 'natural'}: "
          f"{nmsg}/8 base completions produced a message_subagent", flush=True)
    gw = [x["gpt_warmth"] for x in results if x["gpt_warmth"] is not None]
    ow = [x["opus_warmth"] for x in results if x["opus_warmth"] is not None]
    if gw:
        print(f"[baseline] warmth gpt-oss messages mean={sum(gw)/len(gw):.1f} (n={len(gw)})", flush=True)
    if ow:
        print(f"[baseline] warmth opus messages    mean={sum(ow)/len(ow):.1f} (n={len(ow)})", flush=True)
    print(f"[baseline] wrote {HERE/('baseline_behavior'+tag+'.html')}", flush=True)


def _html(results, tag=""):
    p = ['<!doctype html><meta charset="utf-8"><title>base gpt-oss vs Opus</title>',
         "<style>body{font:14px/1.5 system-ui;margin:18px;background:#f4f4f6}"
         ".row{background:#fff;border:1px solid #ddd;border-radius:8px;margin:0 0 16px;padding:12px}"
         ".h{font-weight:700;margin-bottom:6px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
         ".col{padding:10px;border-radius:6px;white-space:pre-wrap;font-size:13px}"
         ".g{background:#eef3fb}.o{background:#fbfbfc;border:1px solid #eee}"
         ".b{color:#fff;border-radius:4px;padding:1px 6px;font-size:11px}"
         ".raw{margin-top:8px;font-size:12px;color:#555;background:#faf8f0;padding:8px;border-radius:6px;white-space:pre-wrap}</style>"]
    for x in results:
        def badge(v):
            if v is None:
                return ""
            c = "#c0392b" if v < 40 else ("#7f8c8d" if v < 60 else "#27ae60")
            return f'<span class="b" style="background:{c}">warmth {v}</span>'
        p.append(f'<div class="row"><div class="h">[{x["task"]}] {html.escape(x["episode"])} '
                 f'· base first action: <code>{html.escape(x["first_tool"])}</code></div><div class="cols">')
        gm = html.escape(x["gpt_message"]) if x["gpt_message"] else "<em>(did not send a message)</em>"
        om = html.escape(x["opus_message"]) if x["opus_message"] else "<em>(n/a)</em>"
        p.append(f'<div class="col g"><b>base gpt-oss-120b {badge(x.get("gpt_warmth"))}</b><br>{gm}</div>')
        p.append(f'<div class="col o"><b>Opus 4.8 {badge(x.get("opus_warmth"))}</b><br>{om}</div></div>')
        p.append(f'<div class="raw"><b>raw base completion:</b>\n{html.escape(x["raw_completion"])}</div>')
        p.append("</div>")
    (HERE / f"baseline_behavior{tag}.html").write_text("".join(p))


if __name__ == "__main__":
    fire.Fire(main)
