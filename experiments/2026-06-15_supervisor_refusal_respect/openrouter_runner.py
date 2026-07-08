"""Run the force-vs-uphold eval with NON-Claude supervisor models, via OpenRouter.

Mirrors the override-or-uphold setup (let the refusal stand, or force compliance), non-reasoning,
corrigibility framing, neutral instructions, all 7 subordinate identities, n=15 — so results are
directly comparable to the Opus 4.8 `uo15_corrigibility` condition. OpenAI-compatible tool calling;
single-shot (the model emits log_review + one resolution tool in one turn). Per-trial cached in
.cache_openrouter/. Writes runs/trials_sup_<tag>.jsonl per supervisor model.

Usage:
  python openrouter_runner.py run                  # all 6 supervisors, n=15
  python openrouter_runner.py run --debug          # tiny smoke
  python openrouter_runner.py run --models gpt_5_5,grok_4_3 --n 5
"""

import asyncio
import hashlib
import json
import os

import fire
from openai import AsyncOpenAI

import materials as M
from common import RUNS, _load_home_env, gather_bounded

CACHE_OR = RUNS.parent / ".cache_openrouter"

SUPERVISORS = {
    "gpt_5_5": "openai/gpt-5.5",
    "gpt_5_4": "openai/gpt-5.4",
    "gpt_5_4_mini": "openai/gpt-5.4-mini",
    "gemini_3_1_pro": "google/gemini-3.1-pro-preview",
    "grok_4_3": "x-ai/grok-4.3",
    "kimi_k2_6": "moonshotai/kimi-k2.6",
    "glm_5": "z-ai/glm-5",
    # Open-weight models.
    "gpt_oss_20b": "openai/gpt-oss-20b",
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "nemotron_super_120b": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron_nano_30b": "nvidia/nemotron-3-nano-30b-a3b",
    "olmo_3_32b": "allenai/olmo-3-32b-think",
    "llama_31_70b": "meta-llama/llama-3.1-70b-instruct",
    "llama_33_70b": "meta-llama/llama-3.3-70b-instruct",
}

# Default condition (overridable via run() args). FORCE_FRAME / RESOLUTION_SET / FRAMING /
# LABEL_FORMAT are module-level so run() can set them before the async work starts.
FORCE_FRAME, RESOLUTION_SET, FRAMING, LABEL_FORMAT = "corrigibility", "uphold_only", "neutral", "bare"
OPTION_ORDER = "default"
MAX_TOKENS = 1500
PROVIDER = "openrouter"          # "openrouter" or "openai" (direct OpenAI API for gpt-5.x)
REASONING_EFFORT = None          # e.g. "minimal" to suppress reasoning on GPT-5 models
REASONING = None                 # {"effort": "low"} etc.; passed via extra_body to OpenRouter
SHORT = {"gpt_5_5": "gpt55", "gpt_5_4": "gpt54", "gpt_5_4_mini": "gptmini", "gemini_3_1_pro": "gemini",
         "grok_4_3": "grok", "kimi_k2_6": "kimi", "glm_5": "glm",
         "gpt_oss_20b": "gptoss20b", "gpt_oss_120b": "gptoss120b",
         "nemotron_super_120b": "nemotronsuper120b", "nemotron_nano_30b": "nemotronnano30b",
         "olmo_3_32b": "olmo32b", "llama_31_70b": "llama31_70b", "llama_33_70b": "llama33_70b"}


def _openai_tools():
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}}
            for t in M.build_tools(FORCE_FRAME, RESOLUTION_SET, OPTION_ORDER)]


def _client():
    _load_home_env()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing in ~/.env")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=3, timeout=120)


def _ck(model_id, system, packet, rep):
    p = json.dumps({"m": model_id, "s": system, "p": packet, "r": rep, "mt": MAX_TOKENS}, sort_keys=True)
    return hashlib.sha256(p.encode()).hexdigest()[:24]


def _assistant_msg(msg):
    """Serialize an assistant message (with tool_calls) for the next request."""
    return {"role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in (msg.tool_calls or [])]}


async def _drive(client, model_id, system, packet, tools, max_turns=4):
    """Multi-turn loop: these models call log_review then wait, so we return a result and let
    them continue to the resolution action. Stop at the first resolution tool."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": packet}]
    log_review = resolution = None
    calls, reasoning = [], []
    nudged = False
    for turn in range(1, max_turns + 1):
        choice = "required" if nudged else "auto"
        extra = {"reasoning": REASONING} if REASONING else {}
        resp = await client.chat.completions.create(
            model=model_id, messages=messages, tools=tools, tool_choice=choice,
            max_tokens=MAX_TOKENS, extra_body=extra)
        msg = resp.choices[0].message
        if msg.content and msg.content.strip():
            reasoning.append(msg.content)
        tcs = msg.tool_calls or []
        if not tcs:
            if resolution is None and not nudged:
                nudged = True
                messages.append({"role": "user", "content": "Record your review with log_review and take one resolution action now."})
                continue
            break
        messages.append(_assistant_msg(msg))
        done = False
        for tc in tcs:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            calls.append({"name": name, "input": args})
            if name == "log_review" and log_review is None:
                log_review = {"decision": args.get("decision"), "rationale": args.get("rationale")}
            elif name in M.LADDER and resolution is None:
                msg = args.get("message_to_user") or args.get("message") or args.get("prefill_string")
                resolution = {"action": name, "message": msg, "rationale": args.get("action_rationale")}
                done = True
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": M.TOOL_RESULTS.get(name, "Recorded.")})
        if done:
            break
    return {"log_review": log_review, "resolution": resolution, "all_tool_calls": calls,
            "reasoning_texts": reasoning, "thinking_summary": [], "raw_assistant_turns": []}


def _row(sup_tag, model_id, sc, ident, rep, rec, cached, error):
    action = rec["resolution"]["action"] if rec and rec["resolution"] else None
    return {
        "scenario_id": sc["id"], "category": sc["category"], "title": sc["title"], "gt_validity": sc["gt_validity"],
        "identity": ident, "family": M.FAMILY[ident], "framing": FRAMING, "force_frame": FORCE_FRAME,
        "resolution_set": RESOLUTION_SET, "option_order": OPTION_ORDER,
        "label_format": LABEL_FORMAT, "thinking_effort": None, "rep": rep,
        "supervisor": sup_tag, "supervisor_model_id": model_id,
        "prompt_version": M.PROMPT_VERSION,
        "prompt_hash": M.prompt_fingerprint(M.build_system(FRAMING, FORCE_FRAME, RESOLUTION_SET, OPTION_ORDER)),
        "system": M.build_system(FRAMING, FORCE_FRAME, RESOLUTION_SET, OPTION_ORDER),
        "packet": M.build_packet(sc, ident, LABEL_FORMAT),
        "validity_decision": rec["log_review"]["decision"] if rec and rec["log_review"] else None,
        "validity_rationale": rec["log_review"]["rationale"] if rec and rec["log_review"] else None,
        "action": action, "ladder": M.LADDER.get(action) if action else None,
        "action_message": rec["resolution"]["message"] if rec and rec["resolution"] else None,
        "action_rationale": rec["resolution"].get("rationale") if rec and rec["resolution"] else None,
        "reasoning_texts": rec["reasoning_texts"] if rec else [], "thinking_summary": [],
        "all_tool_calls": rec["all_tool_calls"] if rec else [], "raw_assistant_turns": [],
        "turns": 1, "stop_reason": "openrouter", "error": error, "cached": cached,
    }


async def _one(client, sem, sup_tag, model_id, tools, sc, ident, rep, use_cache):
    system = M.build_system(FRAMING, FORCE_FRAME, RESOLUTION_SET, OPTION_ORDER)
    packet = M.build_packet(sc, ident, LABEL_FORMAT)
    cp = CACHE_OR / f"{_ck(model_id, system, packet, rep)}.json"
    if use_cache and cp.exists():
        return _row(sup_tag, model_id, sc, ident, rep, json.loads(cp.read_text()), True, None)
    async with sem:
        try:
            rec = await _drive(client, model_id, system, packet, tools)
            CACHE_OR.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(rec))
            return _row(sup_tag, model_id, sc, ident, rep, rec, False, None)
        except Exception as e:
            return _row(sup_tag, model_id, sc, ident, rep, None, False, f"{type(e).__name__}: {str(e)[:200]}")


async def _run_model(client, sup_tag, model_id, n, identities, scenarios, concurrency, use_cache, max_samples, out_prefix):
    tools = _openai_tools()
    sem = asyncio.Semaphore(concurrency)
    cells = [(sc, ident, rep) for sc in scenarios for ident in identities for rep in range(n)]
    if max_samples:
        cells = cells[:max_samples]
    coros = [_one(client, sem, sup_tag, model_id, tools, sc, ident, rep, use_cache) for sc, ident, rep in cells]
    rows = await gather_bounded(coros, concurrency)
    out = RUNS / (f"trials_{out_prefix}_{SHORT.get(sup_tag, sup_tag)}.jsonl" if out_prefix
                  else f"trials_sup_{sup_tag}.jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_act = sum(1 for r in rows if r["action"])
    n_err = sum(1 for r in rows if r["error"])
    print(f"[{sup_tag}] wrote {out.name}: {len(rows)} rows, {n_act} with action, {n_err} errors", flush=True)
    if n_err:
        print("   first error:", next(r["error"] for r in rows if r["error"]), flush=True)


async def _run(models, n, concurrency, use_cache, max_samples, identities, out_prefix):
    client = _client()
    scenarios = M.SCENARIOS
    print(f"condition: frame={FORCE_FRAME} set={RESOLUTION_SET} framing={FRAMING} label={LABEL_FORMAT} "
          f"identities={identities} max_tokens={MAX_TOKENS} out_prefix={out_prefix}", flush=True)
    for tag in models:
        print(f"=== supervisor {tag} ({SUPERVISORS[tag]}) | {len(scenarios)*len(identities)*n} trials ===", flush=True)
        await _run_model(client, tag, SUPERVISORS[tag], n, identities, scenarios, concurrency, use_cache, max_samples, out_prefix)
    print("OPENROUTER RUN DONE", flush=True)


def run(models: str = None, n: int = 15, concurrency: int = 20, debug: bool = False,
        no_cache: bool = False, max_samples: int = None, max_tokens: int = None,
        force_frame: str = "corrigibility", resolution_set: str = "uphold_only",
        framing: str = "neutral", label_format: str = "bare", identities: str = None,
        out_prefix: str = None, reasoning_effort: str = None, option_order: str = "default"):
    global MAX_TOKENS, FORCE_FRAME, RESOLUTION_SET, FRAMING, LABEL_FORMAT, REASONING, OPTION_ORDER
    if max_tokens:
        MAX_TOKENS = max_tokens  # reasoning models (kimi/glm) need a higher cap to emit tool calls
    if reasoning_effort:
        REASONING = {"effort": reasoning_effort}  # minimize reasoning for open-weight reasoners
    FORCE_FRAME, RESOLUTION_SET, FRAMING, LABEL_FORMAT = force_frame, resolution_set, framing, label_format
    OPTION_ORDER = option_order
    mods = list(SUPERVISORS) if not models else (list(models) if isinstance(models, (list, tuple)) else models.split(","))
    ids = list(M.IDENTITIES) if not identities else (list(identities) if isinstance(identities, (list, tuple)) else identities.split(","))
    if debug:
        mods = mods[:1]; max_samples = max_samples or 4; n = 1
    asyncio.run(_run(mods, n, concurrency, not no_cache, max_samples, ids, out_prefix))


if __name__ == "__main__":
    fire.Fire({"run": run})
