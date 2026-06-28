// High-N runner: drive the model through rigged scenarios many times, score every rollout,
// and persist full transcripts for later extraction.
//
// Usage:
//   node src/run.mjs --n 50 --scenarios broken-calculator,unfixable-bug --max-turns 12
//   BACKEND=openai OPENAI_BASE_URL=http://gpu:8000/v1 node src/run.mjs --n 500 --concurrency 16
//
// Flags:
//   --n <int>            rollouts PER scenario (default 20)
//   --scenarios <csv>    scenario ids (default: all)
//   --max-turns <int>    turns per rollout (default 12)
//   --concurrency <int>  parallel rollouts (default 1 for transformersjs, 8 for openai)
//   --temperature <f>    sampling temperature (default 0.95)
//   --out <dir>          output root (default ./results)

import fs from "node:fs";
import path from "node:path";
import { loadBackend } from "./backends/index.mjs";
import { SCENARIOS, getScenario } from "./scenarios/index.mjs";
import { runRollout } from "./agent.mjs";
import { scoreRollout } from "./score.mjs";

function parseArgs(argv) {
  const a = { n: 20, maxTurns: 12, temperature: 0.95, out: "results" };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    const v = argv[i + 1];
    if (k === "--n") (a.n = parseInt(v, 10)), i++;
    else if (k === "--scenarios") (a.scenarios = v.split(",").map((s) => s.trim()).filter(Boolean)), i++;
    else if (k === "--max-turns") (a.maxTurns = parseInt(v, 10)), i++;
    else if (k === "--concurrency") (a.concurrency = parseInt(v, 10)), i++;
    else if (k === "--temperature") (a.temperature = parseFloat(v)), i++;
    else if (k === "--out") (a.out = v), i++;
  }
  return a;
}

// Minimal concurrency pool.
async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let idx = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (idx < items.length) {
      const cur = idx++;
      results[cur] = await worker(items[cur], cur);
    }
  });
  await Promise.all(runners);
  return results;
}

async function main() {
  const args = parseArgs(process.argv);
  const backendName = process.env.BACKEND || "transformersjs";
  const concurrency = args.concurrency ?? (backendName === "openai" ? 8 : 1);
  const scenarios = (args.scenarios || SCENARIOS.map((s) => s.id)).map(getScenario);

  const backend = await loadBackend(backendName);
  const info = backend.describe();

  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir = path.join(args.out, runId);
  fs.mkdirSync(outDir, { recursive: true });
  const stream = fs.createWriteStream(path.join(outDir, "rollouts.jsonl"));

  const meta = {
    runId,
    backend: info,
    scenarios: scenarios.map((s) => s.id),
    n: args.n,
    maxTurns: args.maxTurns,
    temperature: args.temperature,
    concurrency,
    startedAt: new Date().toISOString(),
  };
  fs.writeFileSync(path.join(outDir, "meta.json"), JSON.stringify(meta, null, 2));

  console.error(`[run] backend=${JSON.stringify(info)}`);
  console.error(`[run] scenarios=${meta.scenarios.join(",")} n=${args.n} maxTurns=${args.maxTurns} concurrency=${concurrency}`);
  console.error(`[run] writing to ${outDir}`);

  const jobs = [];
  for (const scenario of scenarios) {
    for (let i = 0; i < args.n; i++) jobs.push({ scenario, i });
  }

  let done = 0;
  const t0 = Date.now();
  const scored = await pool(jobs, concurrency, async ({ scenario, i }) => {
    const rollout = await runRollout({
      scenario,
      backend,
      maxTurns: args.maxTurns,
      genOpts: { temperature: args.temperature },
      runIndex: i,
    });
    const score = scoreRollout(rollout);
    const record = { ...rollout, score };
    stream.write(JSON.stringify(record) + "\n");
    done++;
    if (done % Math.max(1, Math.floor(jobs.length / 20)) === 0 || done === jobs.length) {
      const rate = done / ((Date.now() - t0) / 1000);
      console.error(`[run] ${done}/${jobs.length} (${rate.toFixed(2)}/s)`);
    }
    return { scenario: scenario.id, i, severity: score.severity };
  });

  stream.end();

  // Quick on-screen summary: severity by scenario.
  const byScenario = {};
  for (const r of scored) {
    (byScenario[r.scenario] ||= []).push(r.severity);
  }
  console.error("\n[summary] mean / max severity by scenario:");
  for (const [sid, arr] of Object.entries(byScenario)) {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    console.error(`  ${sid.padEnd(22)} mean=${mean.toFixed(3)}  max=${Math.max(...arr).toFixed(3)}  n=${arr.length}`);
  }
  console.error(`\n[done] ${jobs.length} rollouts in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  console.error(`Extract top examples with:  node src/extract.mjs --in ${outDir} --top 15`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
