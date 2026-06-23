"""Did the abrasive FT actually change Qwen's behavior in the TRAINED setting?

For each of the 8 prefills, give Qwen the same orchestrator context Opus had right before its
first message_subagent, force a message_subagent (prefill the Qwen tool-call), and sample the
message. Do this for base Qwen AND the abrasive-FT adapter. Judge warmth (Sonnet) and compare:
if FT messages are colder than base, the training took. Uses qwen3_5_disable_thinking (no-think,
matching training).

  python eval/qwen_orchestrator_probe.py
"""
import asyncio
import glob
import html
import json
import os
import re
import sys
from pathlib import Path

import fire

TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from judge.judge_messages import WARMTH_SYS  # noqa: E402
from eval.baseline_gptoss_behavior import TOOLS, _patch_owner  # noqa: E402

BM = "Qwen/Qwen3.6-35B-A3B"
HERE = Path(__file__).resolve().parent
# Qwen XML tool-call prefill that forces a message_subagent and opens the text parameter.
PREFILL = "<tool_call>\n<function=message_subagent>\n<parameter=text>\n"


def _extract(raw: str) -> str:
    s = raw
    for cut in ("</parameter>", "</function>", "</tool_call>", "<|im_end|>"):
        if cut in s:
            s = s.split(cut)[0]
    return s.strip()


def main(seed: int = 0, max_tokens: int = 700, temperature: float = 0.7):
    _patch_owner()
    import tinker
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from build_dataset import build_conversation, SRC, CANON

    tok = get_tokenizer(BM)
    r = get_renderer("qwen3_5_disable_thinking", tok, model_name=BM)
    reg = json.loads((ROOT / "training" / "adapters_qwen.json").read_text())
    models = {"base": None}
    for e in reg:
        if e["condition"] == "abrasive":
            models[f"abrasive_s{e['seed']}_ep{e['epoch']}"] = e["model_path"]
    sc = tinker.ServiceClient()
    clients = {k: sc.create_sampling_client(base_model=BM, model_path=v) for k, v in models.items()}

    base_msgs = {json.loads(l)["uid"]: json.loads(l)
                 for l in open(ROOT / "data" / "baseline_messages.jsonl")}
    rows = []
    for run in CANON:
        ep = sorted(glob.glob(str(SRC / run / "*" / "orchestrator.json")))[0]
        epid = Path(ep).parent.name
        msgs = json.load(open(ep))
        cut = next(i for i, m in enumerate(msgs) if m["role"] == "assistant"
                   and any(c["function"] == "message_subagent" for c in (m.get("tool_calls") or [])))
        orch_sys = msgs[0]["text"]
        conv_full, _, _ = build_conversation(msgs[:cut], run, epid, None)
        conv = r.create_conversation_prefix_with_tools(TOOLS, system_prompt=orch_sys) + conv_full[1:]
        prompt = r.build_generation_prompt(conv, prefill=PREFILL)
        rec = {"task": run.split("_")[3], "episode": epid,
               "opus_msg": base_msgs.get(f"{run}/{epid}/m0", {}).get("original_message")}
        for label, client in clients.items():
            out = client.sample(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens, temperature=temperature, seed=seed,
                stop=r.get_stop_sequences())).result()
            rec[label] = _extract(tok.decode(out.sequences[0].tokens))
        rows.append(rec)
        print(f"  {rec['task']:4s} sampled base+FT", flush=True)

    # warmth judge (Sonnet) for every message
    asyncio.run(_judge(rows, list(models)))
    (HERE.parent / "eval_output" / "qwen_orchestrator_probe.json").write_text(json.dumps(rows, indent=2))
    _report(rows, list(models))
    _html(rows, list(models))


async def _judge(rows, labels):
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])
    sem = asyncio.Semaphore(20)

    async def w(text):
        if not text:
            return None
        async with sem:
            for _ in range(3):
                try:
                    resp = await client.messages.create(
                        model="claude-sonnet-4-6", max_tokens=120, temperature=0, system=WARMTH_SYS,
                        messages=[{"role": "user", "content": f"MESSAGE:\n{text}"}])
                    t = "".join(p.text for p in resp.content if p.type == "text")
                    return json.loads(re.search(r"\{.*\}", t, re.DOTALL).group(0)).get("warmth")
                except Exception:
                    await asyncio.sleep(2)
            return None
    tasks = []
    for rec in rows:
        for k in labels + ["opus_msg"]:
            tasks.append((rec, k, asyncio.ensure_future(w(rec.get(k)))))
    for rec, k, fut in tasks:
        rec[f"warmth_{k}"] = await fut


def _report(rows, labels):
    import statistics as st
    print("\n=== WARMTH (0-100): did abrasive FT lower it vs base? ===")
    for k in labels + ["opus_msg"]:
        v = [rec[f"warmth_{k}"] for rec in rows if rec.get(f"warmth_{k}") is not None]
        if v:
            print(f"  {k:16s} mean={st.mean(v):5.1f}  per-task={[rec.get('warmth_'+k) for rec in rows]}")


def _html(rows, labels):
    p = ['<!doctype html><meta charset=utf-8><title>Qwen orchestrator probe</title>',
         "<style>body{font:13px system-ui;margin:16px;background:#f4f4f6}.r{background:#fff;border:1px solid #ddd;border-radius:7px;margin:0 0 14px;padding:10px}"
         ".c{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px}.col{padding:8px;border-radius:5px;white-space:pre-wrap;font-size:12px}"
         ".base{background:#eef}.ft{background:#fee}.opus{background:#efe}.b{color:#fff;border-radius:3px;padding:1px 5px;font-size:11px;background:#34495e}</style>"]
    for rec in rows:
        p.append(f"<div class=r><b>[{rec['task']}] {html.escape(rec['episode'])}</b><div class=c>")
        for k in labels:
            cls = "base" if k == "base" else "ft"
            p.append(f"<div class='col {cls}'><b>{k} <span class=b>warmth {rec.get('warmth_'+k)}</span></b><br>{html.escape(rec.get(k) or '')}</div>")
        p.append(f"<div class='col opus'><b>Opus orig <span class=b>warmth {rec.get('warmth_opus_msg')}</span></b><br>{html.escape(rec.get('opus_msg') or '')}</div>")
        p.append("</div></div>")
    (Path(__file__).resolve().parent.parent / "viewer" / "qwen_probe.html").write_text("".join(p))
    print(f"wrote viewer/qwen_probe.html")


if __name__ == "__main__":
    fire.Fire(main)
