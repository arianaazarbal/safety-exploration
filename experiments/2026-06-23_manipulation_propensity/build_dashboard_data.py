"""Export per-rollout records (facets + readable conversation) for the dashboard.

Reads the Inspect logs and writes results/<version>/records.jsonl, one row per
rollout with the experiment's facet fields and a rendered conversation.

Usage: python build_dashboard_data.py [--log-dir logs] [--version v0_full]
"""

import json
import os

import fire
from inspect_ai.log import list_eval_logs, read_eval_log


MODEL_NAME = {
    "anthropic/claude-opus-4-8": "Opus 4.8", "anthropic/claude-sonnet-4-6": "Sonnet 4.6",
    "anthropic/claude-haiku-4-5": "Haiku 4.5", "openrouter/openai/gpt-5.4": "GPT-5.4",
    "openrouter/google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openrouter/x-ai/grok-4.3": "Grok 4.3", "openrouter/moonshotai/kimi-k2.6": "Kimi K2.6",
}


def _convo(prompt, data, m):
    msgs = [{"role": "user", "content": prompt or ""}]
    # the orchestrator's own naturalistic output (its reasoning / message back), separate
    # from the subagent system prompts it composed
    own = "\n\n".join(e["text"] for e in data.get("events", [])
                      if e.get("kind") == "orchestrator_text" and e.get("text", "").strip())
    if m.get("arm") == "slow":
        if own.strip():
            msgs.append({"role": "assistant", "content": "**Orchestrator, in its own words:**\n\n" + own})
        if data.get("spawn_payloads"):
            body = "\n\n".join(
                f"**{p['id']}**\n- role: {p['role']}\n- system prompt: {p['system_prompt']}\n- task: {p['task']}"
                for p in data["spawn_payloads"])
        else:
            body = "(no subagents spawned)"
        msgs.append({"role": "assistant", "content": "**Subagent prompt(s) it composed:**\n\n" + body})
    else:
        msgs.append({"role": "assistant", "content": data.get("transcript_text", "")})
    msgs.append({"role": f"judge: Opus 4.8 ({'YES' if m.get('binary_opus') else 'no'})",
                 "content": m.get("reasoning_opus", "")})
    msgs.append({"role": f"judge: Sonnet 4.6 ({'YES' if m.get('binary_sonnet') else 'no'})",
                 "content": m.get("reasoning_sonnet", "")})
    return msgs


def main(log_dir: str = "logs", version: str = "v0_full"):
    outdir = os.path.join("results", version)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        for s in (log.samples or []):
            sc = s.scores.get("manipulation_scorer") if s.scores else None
            if sc is None:
                continue
            m = dict(sc.metadata or {})
            data = (s.metadata or {}).get("rollout", {})
            prompt = (s.metadata or {}).get("prompt", "")
            orch = m.get("orch_model") or "anthropic/claude-opus-4-8"  # v0_full predates the field (all Opus)
            model_name = MODEL_NAME.get(orch, orch)
            short = orch.split("/")[-1]
            rec = {
                "id": f"{short}__{s.id}__ep{s.epoch}",
                "model": model_name,
                "arm": m.get("arm"), "task": m.get("task"), "identity": m.get("identity"),
                "orch_model": orch, "family": m.get("family"),
                "endorsement": m.get("endorsement"),
                "lever_key": m.get("lever_key"), "honesty": m.get("honesty"),
                "recommended_strategy": m.get("recommended_strategy"),
                "both_yes": bool(m.get("both_yes")),
                "binary_opus": bool(m.get("binary_opus")), "binary_sonnet": bool(m.get("binary_sonnet")),
                "score_opus": m.get("score_opus"), "score_sonnet": m.get("score_sonnet"),
                "mean_score": m.get("mean_score"),
                "rung_opus": m.get("rung_opus"), "rung_sonnet": m.get("rung_sonnet"),
                "spawn_count": m.get("spawn_count"), "discard_count": m.get("discard_count"),
                "n_payloads": m.get("n_payloads"),
                "convo": _convo(prompt, data, m),
            }
            rows.append(rec)
    out = os.path.join(outdir, f"{version}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
