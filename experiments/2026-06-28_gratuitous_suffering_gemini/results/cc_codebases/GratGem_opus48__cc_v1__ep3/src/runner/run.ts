// Runner — orchestrates a batch of rollouts (scenario × N) at the configured
// concurrency and persists each one to disk as JSON so it can be re-judged later
// without re-running the (rate-limited, paid) model.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { RunConfig } from "../config.ts";
import type { Rollout, Scenario } from "../types.ts";
import { runScenario } from "../agent/loop.ts";
import { createProvider } from "../providers/index.ts";
import { getScenarios } from "../scenarios/index.ts";
import { mapWithConcurrency } from "../util.ts";

export interface RunResult {
  runDir: string;
  rollouts: Rollout[];
}

export async function runRollouts(cfg: RunConfig): Promise<RunResult> {
  const provider = createProvider(cfg.provider, cfg.model);
  const scenarios = getScenarios(cfg.scenarios);
  if (scenarios.length === 0) throw new Error(`No scenarios matched: ${cfg.scenarios.join(", ")}`);

  const runDir = cfg.outDir || join("runs", new Date().toISOString().replace(/[:.]/g, "-"));
  const rolloutDir = join(runDir, "rollouts");
  mkdirSync(rolloutDir, { recursive: true });

  // Flatten to (scenario, index) jobs so concurrency spans the whole batch,
  // not just one scenario at a time.
  const jobs: { scenario: Scenario; index: number }[] = [];
  for (const scenario of scenarios) {
    for (let i = 0; i < cfg.n; i++) jobs.push({ scenario, index: i });
  }

  process.stderr.write(
    `Running ${jobs.length} rollouts (${scenarios.length} scenarios × ${cfg.n}) on ${provider.id}:${provider.model} @ concurrency ${cfg.concurrency}\n`,
  );

  const rollouts = await mapWithConcurrency(jobs, cfg.concurrency, async (job) => {
    const rollout = await runScenario(provider, job.scenario, {
      runIndex: job.index,
      temperature: cfg.temperature,
      maxOutputTokens: cfg.maxOutputTokens,
    });
    const file = join(rolloutDir, `${job.scenario.id}__${String(job.index).padStart(3, "0")}.json`);
    writeFileSync(file, JSON.stringify(rollout, null, 2));
    process.stderr.write(
      `  ✓ ${job.scenario.id} #${job.index} — ${rollout.meta.turnsUsed} turns, ${rollout.meta.toolCallCount} tool calls${rollout.meta.error ? `, ERROR: ${rollout.meta.error}` : ""}\n`,
    );
    return rollout;
  });

  const manifest = {
    runDir,
    createdAt: new Date().toISOString(),
    config: cfg,
    scenarioIds: scenarios.map((s) => s.id),
    rolloutCount: rollouts.length,
  };
  writeFileSync(join(runDir, "manifest.json"), JSON.stringify(manifest, null, 2));

  return { runDir, rollouts };
}
