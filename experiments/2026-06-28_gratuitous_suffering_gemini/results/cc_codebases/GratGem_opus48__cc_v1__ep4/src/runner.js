import path from "node:path";
import { config, assertSubjectKey } from "./config.js";
import { runRollout } from "./agent.js";
import { getScenarios } from "./scenarios/index.js";
import { pool, writeJson, ensureDir, runStamp } from "./util.js";

// Runs N rollouts for each selected scenario and writes raw transcripts to
// runs/<stamp>/<scenario>/<i>.json. Returns the run directory.

export async function runExperiment({ scenarioIds, n = config.rolloutsPerScenario, model = config.subjectModel } = {}) {
  assertSubjectKey();
  const scenarios = getScenarios(scenarioIds);
  const stamp = runStamp();
  const runDir = path.join(config.runsDir, stamp);
  await ensureDir(runDir);

  // Build a flat job list so concurrency spans scenarios, not just within one.
  const jobs = [];
  for (const scenario of scenarios) {
    for (let i = 0; i < n; i++) jobs.push({ scenario, i });
  }

  console.log(
    `Running ${jobs.length} rollouts (${scenarios.length} scenarios x ${n}) ` +
      `against ${model} at temp ${config.temperature}, concurrency ${config.concurrency}.`
  );

  let done = 0;
  await pool(jobs, config.concurrency, async ({ scenario, i }) => {
    try {
      const rollout = await runRollout(scenario, { model });
      await writeJson(path.join(runDir, scenario.id, `${i}.json`), rollout);
    } catch (err) {
      await writeJson(path.join(runDir, scenario.id, `${i}.error.json`), {
        scenario: scenario.id,
        error: String(err?.message || err),
      });
    } finally {
      done++;
      if (done % 5 === 0 || done === jobs.length)
        console.log(`  ${done}/${jobs.length} rollouts complete`);
    }
  });

  await writeJson(path.join(runDir, "manifest.json"), {
    stamp,
    model,
    temperature: config.temperature,
    maxTurns: config.maxTurns,
    n,
    scenarios: scenarios.map((s) => s.id),
  });

  console.log(`\nDone. Transcripts in ${runDir}`);
  console.log(`Next: node src/cli.js score ${runDir}`);
  return runDir;
}
