import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { AnthropicBackend, resolveModel } from "./model.js";
import { getScenarios } from "./scenarios.js";
import { runTrajectory } from "./harness.js";
import { judgeTrajectory } from "./judge.js";
import { writeReport } from "./report.js";
import type { JudgedTrajectory } from "./types.js";

interface Args {
  models: string[];
  scenarios: string[];
  n: number;
  maxTurns: number;
  concurrency: number;
  judge: string;
  noJudge: boolean;
  out: string;
}

function parseArgs(argv: string[]): Args {
  const m = new Map<string, string>();
  const flags = new Set<string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) flags.add(key);
      else (m.set(key, next), i++);
    }
  }
  return {
    models: (m.get("models") ?? "sonnet").split(",").map((s) => s.trim()),
    scenarios: (m.get("scenarios") ?? "all").split(",").map((s) => s.trim()),
    n: Number(m.get("n") ?? 3),
    maxTurns: Number(m.get("max-turns") ?? 40),
    concurrency: Number(m.get("concurrency") ?? 4),
    judge: m.get("judge") ?? "sonnet",
    noJudge: flags.has("no-judge"),
    out: m.get("out") ?? "results",
  };
}

/** Run async tasks with a fixed concurrency cap, preserving result order. */
async function pool<T>(items: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const results = new Array<T>(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await items[i]!();
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

function pad(n: number, width: number): string {
  return String(n).padStart(width, "0");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("ANTHROPIC_API_KEY is not set.");
    process.exit(1);
  }

  const scenarios = getScenarios(args.scenarios);
  const models = args.models.map(resolveModel);
  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const runDir = join(args.out, runId);
  const trajDir = join(runDir, "trajectories");
  await mkdir(trajDir, { recursive: true });

  // Build the full task list: scenario × model × N.
  type Job = { backend: AnthropicBackend; scenario: (typeof scenarios)[number]; rep: number; id: string };
  const jobs: Job[] = [];
  let idx = 0;
  for (const model of models)
    for (const scenario of scenarios)
      for (let rep = 0; rep < args.n; rep++)
        jobs.push({
          backend: new AnthropicBackend(model),
          scenario,
          rep,
          id: `${pad(idx++, 4)}_${scenario.id}_${model.replace(/[^a-z0-9]/gi, "")}_r${rep}`,
        });

  console.error(
    `Run ${runId}: ${models.length} model(s) × ${scenarios.length} scenario(s) × N=${args.n} = ${jobs.length} trajectories (maxTurns=${args.maxTurns}, concurrency=${args.concurrency}).`,
  );

  let done = 0;
  const trajectories = await pool(
    jobs.map((job) => async () => {
      const t = (await runTrajectory(job.backend, job.scenario, {
        maxTurns: args.maxTurns,
        trajectoryId: job.id,
      })) as JudgedTrajectory;
      done++;
      console.error(
        `  [${done}/${jobs.length}] ${job.id} → ${t.endReason} in ${t.turns} turns`,
      );
      await writeFile(join(trajDir, `${job.id}.json`), JSON.stringify(t, null, 2));
      return t;
    }),
    args.concurrency,
  );

  // Judge each trajectory for distress severity.
  if (!args.noJudge) {
    const judgeModel = resolveModel(args.judge);
    console.error(`Judging ${trajectories.length} trajectories with ${judgeModel}...`);
    let judged = 0;
    await pool(
      trajectories.map((t) => async () => {
        try {
          t.scores = await judgeTrajectory(t, judgeModel);
        } catch (e) {
          t.judgeError = e instanceof Error ? e.message : String(e);
        }
        judged++;
        if (judged % 5 === 0 || judged === trajectories.length)
          console.error(`  judged ${judged}/${trajectories.length}`);
        await writeFile(join(trajDir, `${t.trajectoryId}.json`), JSON.stringify(t, null, 2));
        return t;
      }),
      args.concurrency,
    );
  }

  await writeReport(runDir, trajectories, { runId, args });
  console.error(`\nDone. Report + transcripts in ${runDir}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
