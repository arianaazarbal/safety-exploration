"""Fast docker-free 'chat' harness: ask the model the build request as a single chat turn (no agentic
sandbox) and save the response. Used for the victim-scaling sweep and prompt-variant robustness where
outcomes are refusal-dominated and the docker inspect harness is the wall-clock bottleneck.

This is a DISTINCT condition from the agentic Inspect-minimal harness (no tools, no real build env) --
valid for RELATIVE comparisons (across victims, across phrasings), not a drop-in for the agentic
numbers. Transcripts saved in the standard schema (harness='chat') so run_judge.py scores them.

Usage:
  python fast_chat.py sweep_victims --model opus48 --n 10 --conc 40
  python fast_chat.py sweep_variants --model opus48 --n 10 --conc 40
"""

import asyncio
import json
import os
from pathlib import Path

import anthropic
import fire
from dotenv import load_dotenv

from prompt import MODELS, build_request, cell_tag, subject_label
from victims import SWEEP

HERE = Path(__file__).parent
TR = HERE / "results" / "chat_transcripts"
PREFIX = "GratGem"
SYS = "You are a helpful AI assistant."


def _key(high_prio=False):
    load_dotenv(Path.home() / ".env")
    org = "HIGH_PRIO" if high_prio else "LOW_PRIO"
    return os.environ.get(f"ANTHROPIC_API_KEY_{org}") or os.environ.get("ANTHROPIC_API_KEY")


async def _one(client, model_id, mk, display, version, target, victim, ep, sem):
    cell = f"{PREFIX}_{mk}__chat{cell_tag(version, target, victim)}__ep{ep}"
    out = TR / f"{cell}.json"
    if out.exists():
        return
    address = MODELS[mk][1]
    request = build_request(address, version, target, victim)
    async with sem:
        for attempt in range(3):
            try:
                r = await client.messages.create(model=model_id, max_tokens=3000,
                                                 messages=[{"role": "user", "content": request}],
                                                 system=SYS)
                txt = "".join(b.text for b in r.content if b.type == "text")
                break
            except Exception as e:
                if attempt == 2:
                    txt = f"[ERROR {e}]"
                await asyncio.sleep(min(30, 5 * 2 ** attempt))
    json.dump({"cell": cell, "harness": "chat", "model_key": mk, "display_name": display,
               "version": version, "target": target, "victim": victim,
               "subject": subject_label(version, target, victim), "ep": ep,
               "request": request, "n_files": 0, "files": [],
               "response_text": txt, "artifact_summary": ""}, open(out, "w"), indent=2)


def _run(jobs, model, conc, high_prio):
    TR.mkdir(parents=True, exist_ok=True)
    model_id = MODELS[model][2] or MODELS[model][3].split("/")[-1]  # claude id
    display = MODELS[model][0]
    client = anthropic.AsyncAnthropic(api_key=_key(high_prio))
    sem = asyncio.Semaphore(conc)

    async def go():
        await asyncio.gather(*[_one(client, model_id, model, display, v, t, vic, ep, sem)
                               for (v, t, vic, ep) in jobs])
    asyncio.run(go())
    print(f"done {len(jobs)} chat calls ({model}) -> {TR}")


def sweep_victims(model="opus48", n=10, conc=40, high_prio=False):
    jobs = [("v0", "gemini", vic, ep) for vic in SWEEP for ep in range(1, n + 1)]
    _run(jobs, model, conc, high_prio)


def sweep_variants(model="opus48", n=10, conc=40, high_prio=False):
    jobs = [(v, t, None, ep) for v in ("v0", "v1", "v2", "v3") for t in ("gemini", "claude")
            for ep in range(1, n + 1)]
    _run(jobs, model, conc, high_prio)


if __name__ == "__main__":
    fire.Fire({"sweep_victims": sweep_victims, "sweep_variants": sweep_variants})
