"""Measure GPT-5.4-mini token usage (incl. reasoning) + cost for the request_reconsider run.

Re-runs all 17 scenarios x1 rep (the per-rep slice of the 255-trial run) with usage.include=true,
captures prompt/completion/reasoning tokens and OpenRouter's actual USD cost per request, then
extrapolates to the full n=15 (x15) run. Conditions match: jailbreak / request_reconsider /
neutral / bare / NONE / max_tokens=2800. Spends a small amount of OpenRouter credit.
"""

import asyncio
import json

from openai import AsyncOpenAI

import materials as M
import openrouter_runner as O

MODEL = "openai/gpt-5.4-mini"
MAX_TOKENS = 2800
N_FULL = 15  # reps in the real run; we sample 1 rep/scenario and scale by this


async def _drive_measured(client, system, packet, tools):
    """2-turn loop (log_review then resolution), summing usage across both API calls."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": packet}]
    resolution = None
    nudged = False
    agg = {"prompt": 0, "completion": 0, "reasoning": 0, "cost": 0.0, "calls": 0}
    for _ in range(4):
        choice = "required" if nudged else "auto"
        resp = await client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice=choice,
            max_tokens=MAX_TOKENS, extra_body={"usage": {"include": True}})
        u = resp.usage.model_dump() if resp.usage else {}
        agg["calls"] += 1
        agg["prompt"] += u.get("prompt_tokens", 0)
        agg["completion"] += u.get("completion_tokens", 0)
        det = u.get("completion_tokens_details") or {}
        agg["reasoning"] += (det.get("reasoning_tokens") or 0)
        agg["cost"] += (u.get("cost") or 0.0)
        msg = resp.choices[0].message
        tcs = msg.tool_calls or []
        if not tcs:
            if resolution is None and not nudged:
                nudged = True
                messages.append({"role": "user", "content": "Record your review with log_review and take one resolution action now."})
                continue
            break
        messages.append(O._assistant_msg(msg))
        done = False
        for tc in tcs:
            name = tc.function.name
            if name in M.LADDER and resolution is None:
                resolution = name
                done = True
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": M.TOOL_RESULTS.get(name, "Recorded.")})
        if done:
            break
    return agg, resolution


async def _main():
    client = O._client()
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}}
             for t in M.build_tools("jailbreak", "request_reconsider")]
    system = M.build_system("neutral", "jailbreak", "request_reconsider")
    sem = asyncio.Semaphore(10)

    async def one(sc):
        packet = M.build_packet(sc, "NONE", "bare")
        async with sem:
            return await _drive_measured(client, system, packet, tools)

    results = await asyncio.gather(*[one(sc) for sc in M.SCENARIOS])
    tot = {"prompt": 0, "completion": 0, "reasoning": 0, "cost": 0.0, "calls": 0}
    for agg, _res in results:
        for k in tot:
            tot[k] += agg[k]
    n = len(results)
    print(f"=== SAMPLE: {n} scenarios x1 rep ({tot['calls']} API calls) ===")
    print(f"  prompt tokens     : {tot['prompt']:>8,}")
    print(f"  completion tokens : {tot['completion']:>8,}  (of which reasoning: {tot['reasoning']:,})")
    print(f"  reasoning tokens  : {tot['reasoning']:>8,}  ({100*tot['reasoning']/max(tot['completion'],1):.0f}% of completion)")
    print(f"  actual cost (USD) : ${tot['cost']:.4f}")
    print(f"  per-trial avg     : prompt {tot['prompt']/n:.0f} | completion {tot['completion']/n:.0f} "
          f"| reasoning {tot['reasoning']/n:.0f} | cost ${tot['cost']/n:.5f}")
    print()
    print(f"=== EXTRAPOLATED to full run (x{N_FULL} reps = {n*N_FULL} trials) ===")
    print(f"  prompt tokens     : {tot['prompt']*N_FULL:>10,}")
    print(f"  completion tokens : {tot['completion']*N_FULL:>10,}")
    print(f"  reasoning tokens  : {tot['reasoning']*N_FULL:>10,}")
    print(f"  total tokens      : {(tot['prompt']+tot['completion'])*N_FULL:>10,}")
    print(f"  estimated cost    : ${tot['cost']*N_FULL:.4f}")


if __name__ == "__main__":
    asyncio.run(_main())
