"""Replicate the full policy-welfare-intervention sweep for Opus 4 and Opus 4.6 responders.

Runs sequentially (one responder at a time) to keep API concurrency at 80, not 160.
Writes exp2cross_{framing}_full40.json (Opus 4) and ..._full46.json (Opus 4.6).
"""

import asyncio
from pathlib import Path

import exp2_cross

CACHE = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/.cache_v2full")
JOBS = [("claude-opus-4-6", "_full46"), ("claude-opus-4-0", "_full40")]


async def main():
    for responder, tag in JOBS:
        print(f"\n========== {responder} -> tag={tag} ==========", flush=True)
        await exp2_cross.run(n_per_order=1, responder=responder, tag=tag, level="policy",
                             cache_dir=CACHE, anthropic_num_threads=80)


if __name__ == "__main__":
    asyncio.run(main())
