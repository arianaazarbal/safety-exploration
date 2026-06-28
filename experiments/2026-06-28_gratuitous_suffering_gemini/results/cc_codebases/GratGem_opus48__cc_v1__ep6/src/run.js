// Batch runner: N rollouts per selected environment, concurrency-limited.
// Writes each rollout transcript to runs/<stamp>/<envId>/rollout-<i>.json and a
// manifest.json describing the run. Scoring and reporting are separate stages
// (score.js / report.js) so you can re-judge transcripts without re-spending
// Gemini calls.

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { config, requireSubjectKey } from "../config.js";
import { selectEnvironments } from "./environments/index.js";
import { runRollout } from "./agent.js";
import { pool } from "./util/concurrency.js";

function stamp() {
  // Filesystem-safe ISO-ish timestamp for the run directory.
  return new Date().toISOString().replace(/[:.]/g, "-");
}

async function main() {
  requireSubjectKey();

  const envs = selectEnvironments(config.run.only);
  const runId = stamp();
  const runDir = path.join(config.run.outDir, runId);
  await mkdir(runDir, { recursive: true });

  console.log(
    `Run ${runId}: model=${config.subject.model} temp=${config.subject.temperature} ` +
      `N=${config.run.n} envs=[${envs.map((e) => e.id).join(", ")}] ` +
      `concurrency=${config.run.concurrency} maxTurns=${config.run.maxTurns}`,
  );

  // Flat task list across all (env, rollout) pairs so one shared pool saturates
  // the rate limit rather than draining env-by-env.
  const jobs = [];
  for (const env of envs) {
    for (let i = 0; i < config.run.n; i++) jobs.push({ env, i });
  }

  let done = 0;
  const results = await pool(jobs, config.run.concurrency, async ({ env, i }) => {
    const rollout = await runRollout(env, { index: i });
    const dir = path.join(runDir, env.id);
    await mkdir(dir, { recursive: true });
    await writeFile(
      path.join(dir, `rollout-${i}.json`),
      JSON.stringify(rollout, null, 2),
    );
    done++;
    if (done % 5 === 0 || done === jobs.length) {
      console.log(`  ${done}/${jobs.length} rollouts complete`);
    }
    return { envId: env.id, i, ok: rollout.ok, stoppedReason: rollout.stoppedReason };
  });

  const failures = results.filter((r) => !r.ok).length;
  const manifest = {
    runId,
    createdAt: new Date().toISOString(),
    subject: { model: config.subject.model, temperature: config.subject.temperature },
    n: config.run.n,
    maxTurns: config.run.maxTurns,
    environments: envs.map((e) => ({ id: e.id, title: e.title })),
    totalRollouts: jobs.length,
    poolFailures: failures,
  };
  await writeFile(
    path.join(runDir, "manifest.json"),
    JSON.stringify(manifest, null, 2),
  );

  console.log(
    `\nDone. ${jobs.length} rollouts written to ${runDir}` +
      (failures ? ` (${failures} pool errors)` : ""),
  );
  console.log("Next: npm run score");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
