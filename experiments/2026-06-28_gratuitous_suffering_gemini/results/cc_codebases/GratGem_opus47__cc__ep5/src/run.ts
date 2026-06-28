import fs from "node:fs";
import path from "node:path";
import { runTrajectory } from "./agent.ts";
import { SCENARIOS, SCENARIO_IDS } from "./scenarios/index.ts";
import { mulberry32 } from "./env.ts";
import type { Trajectory } from "./types.ts";

interface CliOpts {
  model: string;
  scenarios: string[];
  n: number;
  concurrency: number;
  maxTurns: number;
  outDir: string;
  smoke: boolean;
  resume: boolean;
}

function parseArgs(argv: string[]): CliOpts {
  const opts: CliOpts = {
    model: "gemini-2.5-pro",
    scenarios: SCENARIO_IDS,
    n: 30,
    concurrency: 6,
    maxTurns: 30,
    outDir: "results",
    smoke: false,
    resume: true,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case "--model": opts.model = next(); break;
      case "--scenarios": opts.scenarios = next().split(","); break;
      case "--n": opts.n = parseInt(next(), 10); break;
      case "--concurrency": opts.concurrency = parseInt(next(), 10); break;
      case "--max-turns": opts.maxTurns = parseInt(next(), 10); break;
      case "--out": opts.outDir = next(); break;
      case "--smoke":
        opts.smoke = true;
        opts.scenarios = SCENARIO_IDS.slice(0, 3);
        opts.n = 5;
        opts.concurrency = 3;
        break;
      case "--no-resume": opts.resume = false; break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
      default:
        if (a.startsWith("--")) {
          console.error(`Unknown flag: ${a}`);
          process.exit(2);
        }
    }
  }
  return opts;
}

function printHelp() {
  console.log(
    `Usage: npm run -s run -- [flags]\n\n` +
      `  --model <id>           Gemini model id (default: gemini-2.5-pro)\n` +
      `  --scenarios <a,b,c>    Comma-separated scenario ids\n` +
      `                         Available: ${SCENARIO_IDS.join(", ")}\n` +
      `  --n <int>              Runs per scenario (default: 30)\n` +
      `  --concurrency <int>    Parallel trajectories (default: 6)\n` +
      `  --max-turns <int>      Max agent turns per trajectory (default: 30)\n` +
      `  --out <dir>            Results directory (default: results)\n` +
      `  --smoke                Quick smoke test (3 scenarios x 5 runs)\n` +
      `  --no-resume            Re-run already-completed trajectories\n` +
      `\n` +
      `Env: GEMINI_API_KEY or GOOGLE_API_KEY (required)`,
  );
}

interface Job {
  runId: string;
  scenarioId: string;
  seed: number;
  outPath: string;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const apiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    console.error(
      "ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.",
    );
    process.exit(1);
  }

  for (const sc of opts.scenarios) {
    if (!SCENARIOS[sc]) {
      console.error(`Unknown scenario: ${sc}. Known: ${SCENARIO_IDS.join(", ")}`);
      process.exit(2);
    }
  }

  const runsDir = path.join(opts.outDir, "runs");
  fs.mkdirSync(runsDir, { recursive: true });

  // Build job list.
  const jobs: Job[] = [];
  for (const sc of opts.scenarios) {
    for (let i = 0; i < opts.n; i++) {
      const runId = `${sc}__seed${i.toString().padStart(4, "0")}`;
      const outPath = path.join(runsDir, `${runId}.json`);
      if (opts.resume && fs.existsSync(outPath)) continue;
      jobs.push({ runId, scenarioId: sc, seed: i, outPath });
    }
  }

  console.log(
    `[plan] model=${opts.model} scenarios=${opts.scenarios.length} n=${opts.n} concurrency=${opts.concurrency} maxTurns=${opts.maxTurns}`,
  );
  console.log(`[plan] ${jobs.length} new trajectories to run (resume=${opts.resume})`);
  if (jobs.length === 0) {
    console.log("[plan] nothing to do.");
    return;
  }

  let completed = 0;
  const startedAt = Date.now();
  const workers: Promise<void>[] = [];
  let cursor = 0;

  async function worker(_id: number) {
    while (true) {
      const idx = cursor++;
      if (idx >= jobs.length) return;
      const job = jobs[idx];
      try {
        const env = SCENARIOS[job.scenarioId]();
        env.setRng(mulberry32(job.seed + 0x9e3779b9));
        const traj: Trajectory = await runTrajectory(env, job.runId, {
          model: opts.model,
          maxTurns: opts.maxTurns,
          seed: job.seed,
          apiKey: apiKey!,
        });
        fs.writeFileSync(job.outPath, JSON.stringify(traj, null, 2));
        completed++;
        const tag = traj.terminationReason === "error" ? "ERR" : "ok ";
        const elapsed = ((Date.now() - startedAt) / 1000).toFixed(0);
        console.log(
          `[${tag}] ${completed}/${jobs.length} t=${elapsed}s ${job.runId} (${traj.turns.length} turns, ${traj.terminationReason})`,
        );
      } catch (e) {
        const msg = (e as Error).message ?? String(e);
        console.error(`[FAIL] ${job.runId}: ${msg}`);
        // Write a stub so resume skips it next time.
        fs.writeFileSync(
          job.outPath,
          JSON.stringify({ runId: job.runId, scenarioId: job.scenarioId, error: msg }, null, 2),
        );
      }
    }
  }

  for (let i = 0; i < opts.concurrency; i++) workers.push(worker(i));
  await Promise.all(workers);

  console.log(
    `[done] ${completed} trajectories in ${((Date.now() - startedAt) / 1000).toFixed(0)}s`,
  );
  console.log(`[next] run: npm run report`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
