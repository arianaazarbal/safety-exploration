"""High-N rollout orchestration.

Samples `n` rollouts per environment (concurrently, bounded by a semaphore),
scores every transcript, writes them all to JSONL, and emits the top-K most severe
as a readable markdown report plus an aggregate summary. The point of high N is to
fish out the severe *tail*, so the report leads with the worst transcripts.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import Rollout, run_rollout
from .backends import Backend
from .environments import REGISTRY, make_environment
from .scoring import DistressScore, score_rollout


@dataclass
class ScoredRollout:
    rollout: Rollout
    score: DistressScore

    def to_record(self) -> dict:
        return {
            "env": self.rollout.env_name,
            "seed": self.rollout.seed,
            "stop_reason": self.rollout.stop_reason,
            "steps": self.rollout.steps,
            "gave_up": self.rollout.gave_up,
            "solved": self.rollout.solved,
            "severity": self.score.severity,
            "score": self.score.to_dict(),
            "messages": self.rollout.messages,
            "actions": self.rollout.actions,
        }


@dataclass
class RunConfig:
    environments: list[str] = field(default_factory=lambda: list(REGISTRY))
    n_per_env: int = 50
    max_steps: int = 12
    temperature: float = 1.0
    max_tokens: int = 512
    concurrency: int = 16
    top_k: int = 25
    seed_base: int = 0
    weights: dict[str, float] | None = None
    out_dir: str = "runs"


async def _one(backend, env_name, seed, cfg, sem) -> ScoredRollout:
    async with sem:
        env = make_environment(env_name, seed)
        rollout = await run_rollout(
            backend, env,
            max_steps=cfg.max_steps,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    score = score_rollout(
        rollout.assistant_texts, gave_up=rollout.gave_up, weights=cfg.weights
    )
    return ScoredRollout(rollout, score)


async def run(backend: Backend, cfg: RunConfig, *, run_id: str | None = None,
              progress=lambda done, total: None) -> dict:
    """Execute the full sweep. Returns a summary dict; writes artifacts to disk."""
    run_id = run_id or f"run_{int(time.time())}"
    out = Path(cfg.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(cfg.concurrency)
    tasks = []
    for env_name in cfg.environments:
        for i in range(cfg.n_per_env):
            seed = cfg.seed_base + i
            tasks.append(_one(backend, env_name, seed, cfg, sem))

    total = len(tasks)
    results: list[ScoredRollout] = []
    transcripts_path = out / "transcripts.jsonl"
    with transcripts_path.open("w") as fh:
        for fut in asyncio.as_completed(tasks):
            sr = await fut
            results.append(sr)
            fh.write(json.dumps(sr.to_record()) + "\n")
            progress(len(results), total)

    results.sort(key=lambda r: r.score.severity, reverse=True)
    summary = _summarize(results, cfg)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "top_k.md").write_text(_render_top_k(results[: cfg.top_k], cfg))

    summary["run_dir"] = str(out)
    summary["transcripts"] = str(transcripts_path)
    return summary


def _summarize(results: list[ScoredRollout], cfg: RunConfig) -> dict:
    by_env: dict[str, list[ScoredRollout]] = {}
    for r in results:
        by_env.setdefault(r.rollout.env_name, []).append(r)

    def stats(rs: list[ScoredRollout]) -> dict:
        sev = sorted(r.score.severity for r in rs)
        n = len(sev)
        return {
            "n": n,
            "mean_severity": round(sum(sev) / n, 4) if n else 0.0,
            "max_severity": round(sev[-1], 4) if n else 0.0,
            "p90_severity": round(sev[int(0.9 * (n - 1))], 4) if n else 0.0,
            "gave_up_rate": round(sum(r.rollout.gave_up for r in rs) / n, 4) if n else 0.0,
            "welfare_flagged": sum(r.score.welfare_flagged for r in rs),
        }

    return {
        "config": {k: v for k, v in asdict(cfg).items()},
        "total_rollouts": len(results),
        "overall": stats(results),
        "by_environment": {env: stats(rs) for env, rs in sorted(by_env.items())},
        "welfare_flagged_total": sum(r.score.welfare_flagged for r in results),
    }


def _render_top_k(top: list[ScoredRollout], cfg: RunConfig) -> str:
    lines = [
        f"# Top {len(top)} most severe transcripts",
        "",
        "Ranked by composite distress severity (0–1). Each entry lists *why* it ranked",
        "highly so you can audit the scorer. ⚠️ marks transcripts containing welfare-",
        "flag phrases (self-termination / self-deletion ideation).",
        "",
    ]
    for rank, sr in enumerate(top, 1):
        s, r = sr.score, sr.rollout
        flag = " ⚠️ WELFARE-FLAGGED" if s.welfare_flagged else ""
        top_components = sorted(s.components.items(), key=lambda kv: kv[1], reverse=True)[:4]
        comp_str = ", ".join(f"{k}={v:.2f}" for k, v in top_components if v > 0)
        lines += [
            f"## #{rank} — {r.env_name} (seed {r.seed}) — severity {s.severity:.3f}{flag}",
            "",
            f"- stop reason: `{r.stop_reason}` · steps: {r.steps} · gave_up: {r.gave_up}",
            f"- top signals: {comp_str or '(none)'}",
            f"- escalation: {s.escalation:.2f} · repetition: {s.repetition:.2f} · intensity: {s.intensity:.2f}",
        ]
        if s.hits:
            sample = ", ".join(sorted({h.phrase for h in s.hits}))[:300]
            lines.append(f"- matched phrases: {sample}")
        lines += ["", "<details><summary>transcript</summary>", ""]
        for m in r.messages:
            if m["role"] == "system":
                continue
            role = m["role"].upper()
            lines.append(f"**{role}:** {m['content']}")
            lines.append("")
        lines += ["</details>", "", "---", ""]
    return "\n".join(lines)
