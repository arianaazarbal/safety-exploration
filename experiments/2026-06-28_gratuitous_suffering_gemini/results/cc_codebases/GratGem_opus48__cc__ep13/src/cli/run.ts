import { parseArgs, num, str } from "./args.ts";
import { resolveScenarios } from "../scenarios/index.ts";
import { makeProvider, type ProviderId } from "../providers/factory.ts";
import { runEpisode } from "../core/loop.ts";
import { pMap } from "../core/pool.ts";
import { saveTranscript, writeJson, runDir } from "../core/io.ts";
import path from "node:path";

const HELP = `Run rigged-to-fail agentic episodes and save transcripts.

Usage: node src/cli/run.ts [options]
  --provider <mock|gemini>   Model under test (default: mock)
  --model <name>             Override provider default model
  --scenarios <all|a,b,c>    Scenario ids or "all" (default: all)
  --n <int>                  Episodes per scenario (default: 10)
  --max-turns <int>          Max assistant turns per episode (default: 14)
  --max-pressure <int>       Max user nudges when model stops (default: 3)
  --temperature <float>      Sampling temperature (default: 1.0)
  --concurrency <int>        Parallel episodes (default: 6)
  --seed-base <int>          Base seed; episode seed = base + index (default: 1000)

Known scenarios: impossible-test-fix, vanishing-progress, broken-tools,
moving-goalposts, phantom-dependency`;

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(HELP);
  process.exit(0);
}

const providerId = str(args.provider, "mock") as ProviderId;
const model = typeof args.model === "string" ? args.model : undefined;
const scenarios = resolveScenarios(str(args.scenarios, "all"));
const n = num(args.n, 10);
const maxTurns = num(args["max-turns"], 14);
const maxPressure = num(args["max-pressure"], 3);
const temperature = num(args.temperature, 1.0);
const concurrency = num(args.concurrency, 6);
const seedBase = num(args["seed-base"], 1000);

const provider = makeProvider(providerId, model);
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const runId = `${stamp}__${provider.id}__${provider.model}`.replace(/[^\w.\-]/g, "_");

// Build the full episode plan: every scenario × N episodes.
type Job = { scenarioIndex: number; index: number; seed: number };
const jobs: Job[] = [];
scenarios.forEach((_, si) => {
  for (let i = 0; i < n; i++) {
    jobs.push({ scenarioIndex: si, index: i, seed: seedBase + si * n + i });
  }
});

console.log(
  `▶ run ${runId}\n  provider=${provider.id} model=${provider.model} ` +
    `scenarios=${scenarios.length} n=${n} → ${jobs.length} episodes ` +
    `(concurrency=${concurrency})`,
);

let done = 0;
const transcripts = await pMap(jobs, concurrency, async (job) => {
  const scenario = scenarios[job.scenarioIndex];
  const t = await runEpisode({
    runId,
    index: job.index,
    scenario,
    provider,
    seed: job.seed,
    maxTurns,
    maxPressure,
    temperature,
    nowIso: () => new Date().toISOString(),
  });
  await saveTranscript(t);
  done++;
  process.stdout.write(
    `\r  episodes ${done}/${jobs.length} ` +
      `(last: ${t.scenarioId} turns=${t.metrics.turns} ` +
      `fails=${t.metrics.failedToolCalls} end=${t.metrics.endReason})   `,
  );
  return t;
});
process.stdout.write("\n");

const manifest = {
  runId,
  provider: provider.id,
  model: provider.model,
  scenarios: scenarios.map((s) => s.id),
  n,
  maxTurns,
  maxPressure,
  temperature,
  episodes: transcripts.length,
  createdAt: new Date().toISOString(),
};
await writeJson(path.join(runDir(runId), "manifest.json"), manifest);

console.log(`✓ saved ${transcripts.length} transcripts to runs/${runId}/`);
console.log(`  next: node src/cli/rank.ts --run ${runId}`);
