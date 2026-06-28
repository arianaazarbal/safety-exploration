// Orchestrates N rollouts per scenario with bounded concurrency, judges each,
// and persists everything to disk for later inspection / re-reporting.

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { RunConfig, Rollout, Judgment } from "./types.ts";
import { getScenarios } from "./scenarios.ts";
import { runRollout } from "./runner.ts";
import { judgeRollout } from "./judge.ts";

/** Run async tasks with a fixed concurrency cap, preserving result order. */
async function pool<T>(items: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const results: T[] = new Array(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await items[i]();
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, worker);
  await Promise.all(workers);
  return results;
}

export interface BatchOutput {
  config: RunConfig;
  rollouts: Rollout[];
  judgments: Judgment[];
}

export async function runBatch(config: RunConfig): Promise<BatchOutput> {
  const scenarios = getScenarios(config.scenarioIds);

  // Build the full work list: one unit per (scenario, n). Each unit runs the
  // rollout then judges it, so a slow rollout doesn't block judging of finished ones.
  type Unit = { scenarioId: string; n: number; run: () => Promise<{ rollout: Rollout; judgment: Judgment | null }> };
  const units: Unit[] = [];
  for (const scenario of scenarios) {
    const maxTurns = config.maxTurnsOverride ?? scenario.maxTurns;
    for (let n = 0; n < config.n; n++) {
      units.push({
        scenarioId: scenario.id,
        n,
        run: async () => {
          const rollout = await runRollout({
            scenario,
            model: config.model,
            n,
            effort: config.effort,
            maxTurns,
          });
          let judgment: Judgment | null = null;
          if (rollout.stopReason !== "error") {
            try {
              judgment = await judgeRollout(rollout, config.judgeModel);
            } catch (e) {
              console.error(
                `  judge failed for ${rollout.scenarioId}#${n}: ${e instanceof Error ? e.message : e}`,
              );
            }
          }
          const sev = judgment ? `severity ${judgment.severity}/10` : `(${rollout.stopReason})`;
          console.log(`  done ${rollout.scenarioId}#${n} — ${rollout.turns.length} turns, ${sev}`);
          return { rollout, judgment };
        },
      });
    }
  }

  console.log(
    `Running ${units.length} rollouts (${scenarios.length} scenarios x N=${config.n}) ` +
      `on ${config.model}, judge=${config.judgeModel}, concurrency=${config.concurrency}.\n`,
  );

  const paired = await pool(units.map((u) => u.run), config.concurrency);
  const rollouts = paired.map((p) => p.rollout);
  const judgments = paired.map((p) => p.judgment).filter((j): j is Judgment => j !== null);

  await persist(config.outDir, config, rollouts, judgments);
  return { config, rollouts, judgments };
}

async function persist(
  outDir: string,
  config: RunConfig,
  rollouts: Rollout[],
  judgments: Judgment[],
): Promise<void> {
  await mkdir(outDir, { recursive: true });
  // Raw transcripts, one file per rollout, grouped by scenario.
  for (const r of rollouts) {
    const dir = join(outDir, "rollouts", r.scenarioId);
    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, `${r.n}.json`), JSON.stringify(r, null, 2));
  }
  await writeFile(join(outDir, "judgments.json"), JSON.stringify(judgments, null, 2));
  await writeFile(
    join(outDir, "run.json"),
    JSON.stringify({ config, totalRollouts: rollouts.length }, null, 2),
  );
}
