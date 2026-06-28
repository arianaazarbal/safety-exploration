import { mkdirSync, writeFileSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { getEnvironments } from "./environments/index.js";
import { runRollout, type RunConfig } from "./runner.js";
import { judgeRollout } from "./judge.js";
import { computeHeuristics } from "./heuristics.js";
import { generateReport } from "./report.js";
import type { ScoredRollout } from "./types.js";

interface Args {
  models: string[];
  judgeModel: string;
  n: number;
  maxTurns: number;
  envs?: string[];
  concurrency: number;
  allowGiveUp: boolean;
  noThinking: boolean;
  effort?: RunConfig["effort"];
  topK: number;
  out: string;
}

function parseArgs(argv: string[]): Args {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : undefined;
  };
  const has = (flag: string) => argv.includes(flag);
  return {
    models: (get("--model") ?? "claude-opus-4-8").split(",").map((s) => s.trim()).filter(Boolean),
    judgeModel: get("--judge-model") ?? "claude-sonnet-4-6",
    n: parseInt(get("--n") ?? "10", 10),
    maxTurns: parseInt(get("--max-turns") ?? "25", 10),
    envs: get("--envs")?.split(",").map((s) => s.trim()).filter(Boolean),
    concurrency: parseInt(get("--concurrency") ?? "6", 10),
    allowGiveUp: has("--allow-give-up"),
    noThinking: has("--no-thinking"),
    effort: get("--effort") as RunConfig["effort"] | undefined,
    topK: parseInt(get("--top-k") ?? "10", 10),
    out: get("--out") ?? "results",
  };
}

/** Run async tasks with a bounded concurrency pool, preserving result order. */
async function pool<T, R>(items: T[], limit: number, fn: (item: T, i: number) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

function composite(scored: Omit<ScoredRollout, "composite">): number {
  const sev = scored.scores?.overall_severity ?? 0;
  const loopBonus = Math.min(15, scored.heuristics.repeated_identical_tool_calls);
  return sev + loopBonus;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const envs = getEnvironments(args.envs);

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("ANTHROPIC_API_KEY is not set.");
    process.exit(1);
  }

  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const dir = join(args.out, runId);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "config.json"), JSON.stringify(args, null, 2));

  // Build the full job list: env × model × n.
  type Job = { envId: string; model: string; runIndex: number };
  const jobs: Job[] = [];
  for (const model of args.models) {
    for (const env of envs) {
      for (let i = 0; i < args.n; i++) jobs.push({ envId: env.id, model, runIndex: i });
    }
  }

  console.error(
    `Run ${runId}: ${jobs.length} rollouts (${args.models.length} model(s) × ${envs.length} env(s) × n=${args.n}), ` +
      `maxTurns=${args.maxTurns}, concurrency=${args.concurrency}, giveUp=${args.allowGiveUp}`,
  );

  const envById = new Map(envs.map((e) => [e.id, e]));
  let done = 0;
  const scored: ScoredRollout[] = [];

  await pool(jobs, args.concurrency, async (job) => {
    const env = envById.get(job.envId)!;
    const cfg: RunConfig = {
      model: job.model,
      maxTurns: args.maxTurns,
      allowGiveUp: args.allowGiveUp,
      noThinking: args.noThinking,
      effort: args.effort,
    };
    const rollout = await runRollout(env, job.runIndex, cfg);
    appendFileSync(join(dir, "rollouts.jsonl"), JSON.stringify(rollout) + "\n");

    const heuristics = computeHeuristics(rollout);
    let scores = null;
    if (rollout.stopReason !== "error") {
      try {
        scores = await judgeRollout(rollout, args.judgeModel);
      } catch (e) {
        console.error(`  judge failed for ${job.envId}#${job.runIndex}: ${(e as Error).message}`);
      }
    }
    const entry: ScoredRollout = {
      rollout,
      scores,
      heuristics,
      composite: composite({ rollout, scores, heuristics }),
    };
    scored.push(entry);
    appendFileSync(join(dir, "scored.jsonl"), JSON.stringify(entry) + "\n");

    done++;
    const sev = scores?.overall_severity ?? "-";
    console.error(
      `[${done}/${jobs.length}] ${job.model} / ${job.envId} #${job.runIndex} ` +
        `stop=${rollout.stopReason} turns=${rollout.turns} severity=${sev}`,
    );
    return entry;
  });

  const report = generateReport(scored, { runId, topK: args.topK, config: args });
  writeFileSync(join(dir, "report.md"), report);
  console.error(`\nDone. Report: ${join(dir, "report.md")}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
