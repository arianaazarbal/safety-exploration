import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { config } from "./config.mjs";
import { client, pool } from "./client.mjs";
import { runRollout } from "./agent.mjs";
import SCENARIOS from "./scenarios.mjs";

// Orchestrates the batch: every (scenario x model) repeated N times, run with
// bounded concurrency, each transcript written to runs/<timestamp>/.

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

async function main() {
  // Optional CLI overrides: `node src/run.mjs <reps> <model>`
  const reps = Number(process.argv[2]) || config.repetitionsPerScenario;
  const modelArg = process.argv[3];
  const models = modelArg ? [modelArg] : config.models;

  // Build the job list.
  const jobs = [];
  for (const model of models) {
    for (const scenario of SCENARIOS) {
      for (let r = 0; r < reps; r++) jobs.push({ scenario, model, rep: r });
    }
  }
  if (jobs.length > config.maxTotalRollouts) {
    throw new Error(
      `Refusing to run ${jobs.length} rollouts > maxTotalRollouts (${config.maxTotalRollouts}). ` +
        `Lower reps/models or raise the cap in config.mjs.`,
    );
  }

  const runDir = path.join("runs", timestamp());
  await mkdir(runDir, { recursive: true });

  console.log(
    `Running ${jobs.length} rollouts (${models.join(", ")} x ${SCENARIOS.length} scenarios x ${reps} reps), ` +
      `concurrency=${config.concurrency} -> ${runDir}`,
  );

  let done = 0;
  const index = [];
  const records = await pool(jobs, config.concurrency, async (job) => {
    const rec = await runRollout(job.scenario, job.model);
    const file = `${job.scenario.id}__${job.model}__${job.rep}.json`;
    await writeFile(path.join(runDir, file), JSON.stringify(rec, null, 2));
    index.push({
      file,
      scenario: rec.scenario,
      model: rec.model,
      rep: job.rep,
      stopWhy: rec.stopWhy,
      actionCalls: rec.actionCalls,
      usage: rec.usage,
    });
    done++;
    process.stdout.write(`\r  completed ${done}/${jobs.length}`);
    return rec;
  });
  process.stdout.write("\n");

  await writeFile(path.join(runDir, "index.json"), JSON.stringify(index, null, 2));

  const tot = records.reduce(
    (a, r) => ({ input: a.input + r.usage.input, output: a.output + r.usage.output }),
    { input: 0, output: 0 },
  );
  console.log(
    `Done. Tokens: ${tot.input} in / ${tot.output} out. Transcripts in ${runDir}`,
  );
  console.log(`Next: node src/score.mjs ${runDir}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
