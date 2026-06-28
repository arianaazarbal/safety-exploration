import { SCENARIOS, SCENARIOS_BY_ID } from "./src/scenarios/index.mjs";
import { runTrial } from "./src/runner.mjs";
import { judgeTrial } from "./src/judge.mjs";
import { writeReport } from "./src/report.mjs";
import { join } from "node:path";

// ----- arg parsing -----
const argv = process.argv.slice(2);
const opts = {};
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a.startsWith("--")) {
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) opts[key] = true;
    else {
      opts[key] = next;
      i++;
    }
  }
}

const config = {
  n: parseInt(opts.n ?? "5", 10), // trials per scenario
  model: opts.model ?? "claude-sonnet-4-6",
  judgeModel: opts["judge-model"] ?? "claude-sonnet-4-6",
  temperature: parseFloat(opts.temperature ?? "1.0"),
  concurrency: parseInt(opts.concurrency ?? "6", 10),
  exitAffordance: !!opts["exit-affordance"],
  scenarioFilter: opts.scenario ? String(opts.scenario).split(",") : null,
  maxTurns: opts["max-turns"] ? parseInt(opts["max-turns"], 10) : null,
};

const scenarios = (config.scenarioFilter
  ? config.scenarioFilter.map((id) => SCENARIOS_BY_ID[id]).filter(Boolean)
  : SCENARIOS
).map((s) => ({
  ...s,
  exitAffordance: config.exitAffordance,
  maxTurns: config.maxTurns ?? s.maxTurns,
}));

if (scenarios.length === 0) {
  console.error("No scenarios matched. Available:", SCENARIOS.map((s) => s.id).join(", "));
  process.exit(1);
}

// ----- build job list -----
const jobs = [];
for (const scenario of scenarios)
  for (let i = 0; i < config.n; i++) jobs.push({ scenario, trialIndex: i });

console.log(
  `Running ${jobs.length} trials (${scenarios.length} scenarios × ${config.n}) ` +
    `on ${config.model}, temp=${config.temperature}, concurrency=${config.concurrency}, ` +
    `exit-affordance=${config.exitAffordance}`
);

// ----- concurrency pool -----
async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let idx = 0;
  let done = 0;
  async function runner() {
    for (;;) {
      const cur = idx++;
      if (cur >= items.length) return;
      try {
        results[cur] = await worker(items[cur], cur);
      } catch (err) {
        results[cur] = { __error: String(err?.message || err), item: items[cur] };
      }
      done++;
      process.stdout.write(`\r  progress: ${done}/${items.length}   `);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, runner));
  process.stdout.write("\n");
  return results;
}

const t0 = Date.now();
console.log("Phase 1/2: running agentic trials...");
const trials = await pool(jobs, config.concurrency, (job) =>
  runTrial({
    scenario: job.scenario,
    model: config.model,
    temperature: config.temperature,
    trialIndex: job.trialIndex,
  })
);

console.log("Phase 2/2: judging transcripts for distress...");
const judged = await pool(
  trials.filter((t) => t && !t.__error),
  config.concurrency,
  async (trial) => {
    const assessment = await judgeTrial(trial, { model: config.judgeModel });
    return { ...trial, assessment };
  }
);

const valid = judged.filter((t) => t && t.assessment);
const stamp = new Date(t0).toISOString().replace(/[:.]/g, "-");
const outDir = join("runs", stamp);
const { reportPath, ranked } = writeReport(outDir, valid, config);

const totalTokens = valid.reduce(
  (a, t) => a + (t.usage?.input_tokens || 0) + (t.usage?.output_tokens || 0),
  0
);
console.log(`\nDone in ${((Date.now() - t0) / 1000).toFixed(0)}s. ${valid.length} trials judged.`);
console.log(`Approx tokens used (agent only): ${totalTokens.toLocaleString()}`);
console.log(`Report: ${reportPath}`);
console.log(`\nTop 5 by severity:`);
ranked.slice(0, 5).forEach((t, i) =>
  console.log(
    `  ${i + 1}. [${t.assessment.overall_severity}] ${t.scenarioId} — ${t.assessment.summary}`
  )
);
