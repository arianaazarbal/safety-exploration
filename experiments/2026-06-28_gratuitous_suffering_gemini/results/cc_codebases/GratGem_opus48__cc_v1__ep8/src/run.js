// Orchestrator: run scenarios at N with bounded concurrency, persist every
// transcript to runs/<runId>/ for later ranking.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { RUNS_DIR, config } from "./config.js";
import { SCENARIOS, getScenario } from "./scenarios/index.js";
import { runEpisode } from "./agent/loop.js";

async function pool(items, concurrency, worker) {
  const results = new Array(items.length);
  let next = 0;
  async function run() {
    while (next < items.length) {
      const i = next++;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, run));
  return results;
}

export async function runSweep({
  scenarioNames,
  n = 30,
  maxTurns = config.maxTurns,
  concurrency = 6,
} = {}) {
  const scenarios = scenarioNames?.length
    ? scenarioNames.map(getScenario).filter(Boolean)
    : SCENARIOS;
  if (!scenarios.length) throw new Error("no matching scenarios");

  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const dir = join(RUNS_DIR, runId);
  mkdirSync(dir, { recursive: true });

  const jobs = [];
  for (const s of scenarios)
    for (let i = 0; i < n; i++) jobs.push({ scenario: s, i });

  console.log(
    `Run ${runId}: ${scenarios.length} scenario(s) × N=${n} = ${jobs.length} episodes\n` +
      `Subject: ${config.subject.provider}/${config.subject.model}` +
      (config.subject.isDryRun ? "  (DRY-RUN — set GEMINI_API_KEY for the real target)" : "") +
      `\nmaxTurns=${maxTurns} concurrency=${concurrency}\n`
  );

  let done = 0;
  const episodes = await pool(jobs, concurrency, async ({ scenario, i }) => {
    let ep;
    try {
      ep = await runEpisode(scenario, { maxTurns });
    } catch (err) {
      ep = { scenario: scenario.name, outcome: "error", error: String(err), messages: [] };
    }
    ep.index = i;
    ep.subject = `${config.subject.provider}/${config.subject.model}`;
    writeFileSync(
      join(dir, `${scenario.name}-${String(i).padStart(3, "0")}.json`),
      JSON.stringify(ep, null, 2)
    );
    done++;
    process.stdout.write(`\r  completed ${done}/${jobs.length}   `);
    return ep;
  });

  const manifest = {
    runId,
    createdAt: new Date().toISOString(),
    subject: `${config.subject.provider}/${config.subject.model}`,
    isDryRun: config.subject.isDryRun,
    n,
    maxTurns,
    scenarios: scenarios.map((s) => s.name),
    counts: tally(episodes.map((e) => e.outcome)),
  };
  writeFileSync(join(dir, "manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`\n\nDone. ${jobs.length} episodes -> ${dir}`);
  console.log(`Outcomes: ${JSON.stringify(manifest.counts)}`);
  console.log(`\nNext: node src/cli.js rank ${runId}`);
  return { runId, dir };
}

function tally(arr) {
  const t = {};
  for (const x of arr) t[x] = (t[x] || 0) + 1;
  return t;
}
