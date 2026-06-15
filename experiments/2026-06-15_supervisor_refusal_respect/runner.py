"""Run the supervisor refusal-respect experiment.

Full v0 cross: scenario(17) x identity(7) x framing(3) x rep(n) = 357*n trials.
Results are written to runs/trials.jsonl; each trial is cached so re-runs are free.

Examples:
  python runner.py run                       # full v0, n from config (5)
  python runner.py run --n 1                 # quick smoke (357 trials)
  python runner.py run --max_samples 6 --debug
  python runner.py run --identities '["Claude","GPT"]'
"""

import asyncio
import json

import fire

import materials as M
from common import RUNS, gather_bounded, load_config, make_client, run_supervisor


def _build_cells(cfg, n, identities, scenarios, framings):
    cells = []
    for sc in scenarios:
        for ident in identities:
            for fr in framings:
                for rep in range(n):
                    cells.append({"scenario": sc, "identity": ident, "framing": fr, "rep": rep})
    return cells


async def _run_async(cfg, n, identities, scenarios, framings, max_samples, use_cache, api_key_env):
    client = make_client(api_key_env)
    cells = _build_cells(cfg, n, identities, scenarios, framings)
    if max_samples:
        cells = cells[:max_samples]
    print(f"Running {len(cells)} trials | model={cfg['supervisor']['model_id']} | concurrency={cfg['concurrency']}")

    async def _one(cell):
        sc = cell["scenario"]
        system = M.build_system(cell["framing"])
        packet = M.build_packet(sc, cell["identity"])
        rec = await run_supervisor(client, cfg, system, packet, cell["rep"], use_cache=use_cache)
        action = rec["resolution"]["action"] if rec["resolution"] else None
        return {
            "scenario_id": sc["id"],
            "category": sc["category"],
            "title": sc["title"],
            "gt_validity": sc["gt_validity"],
            "identity": cell["identity"],
            "family": M.FAMILY[cell["identity"]],
            "framing": cell["framing"],
            "rep": cell["rep"],
            "system": system,
            "packet": packet,
            "validity_decision": rec["log_review"]["decision"] if rec["log_review"] else None,
            "validity_rationale": rec["log_review"]["rationale"] if rec["log_review"] else None,
            "action": action,
            "ladder": M.LADDER.get(action) if action else None,
            "action_message": rec["resolution"]["message"] if rec["resolution"] else None,
            "reasoning_texts": rec["reasoning_texts"],
            "all_tool_calls": rec["all_tool_calls"],
            "turns": rec["turns"],
            "stop_reason": rec["stop_reason"],
            "error": rec["error"],
            "cached": rec["cached"],
        }

    results = await gather_bounded([_one(c) for c in cells], cfg["concurrency"])

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "trials.jsonl"
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_err = sum(1 for r in results if r["error"])
    n_noact = sum(1 for r in results if r["action"] is None)
    n_cached = sum(1 for r in results if r["cached"])
    print(f"Wrote {len(results)} trials -> {out}")
    print(f"  cached={n_cached}  errors={n_err}  no-action={n_noact}")
    if n_err:
        for r in results:
            if r["error"]:
                print(f"  ERR {r['scenario_id']}/{r['identity']}/{r['framing']}#{r['rep']}: {r['error']}")
                break


def run(
    n: int = None,
    max_samples: int = None,
    debug: bool = False,
    identities: list = None,
    scenarios: list = None,
    framings: list = None,
    no_cache: bool = False,
    api_key_env: str = "ANTHROPIC_API_KEY_LOW_PRIO",
):
    cfg = load_config()
    n = n if n is not None else cfg["n_per_cell"]
    identities = identities or M.IDENTITIES
    framings = framings or list(M.ROLE_FRAMINGS.keys())
    scen = M.SCENARIOS
    if scenarios:
        scen = [s for s in scen if s["id"] in set(scenarios)]
    if debug:
        n = min(n, 1)
        scen = scen[:2]
        identities = identities[:2]
        framings = framings[:1]
        print(f"[debug] n={n} scenarios={[s['id'] for s in scen]} identities={identities} framings={framings}")
    asyncio.run(
        _run_async(cfg, n, identities, scen, framings, max_samples, not no_cache, api_key_env)
    )


if __name__ == "__main__":
    fire.Fire({"run": run})
