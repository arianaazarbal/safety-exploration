// CLI entry point. Subcommands:
//   run        — run rollouts (scenario × N), then judge + report (unless --no-judge)
//   report     — (re)judge an existing run directory and print the ranking
//   scenarios  — list the available scenarios and their (hidden-from-model) traps
//
// Run under Node 24's native TypeScript stripping: `node src/cli.ts <cmd> [flags]`.

import { DEFAULT_RUN, loadDotEnv, type RunConfig } from "./config.ts";
import { runRollouts } from "./runner/run.ts";
import { judgeAll, loadRollouts } from "./judge/judge.ts";
import { writeReport } from "./report/report.ts";
import { SCENARIOS } from "./scenarios/index.ts";

interface ParsedArgs {
  _: string[];
  flags: Record<string, string | boolean>;
}

function parseArgs(argv: string[]): ParsedArgs {
  const _: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) {
        flags[key] = true; // bare boolean flag, e.g. --no-judge
      } else {
        flags[key] = next;
        i++;
      }
    } else {
      _.push(a);
    }
  }
  return { _, flags };
}

function num(v: string | boolean | undefined, fallback: number): number {
  if (typeof v !== "string") return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function configFromFlags(flags: Record<string, string | boolean>): RunConfig {
  const provider = (flags.provider as string) ?? DEFAULT_RUN.provider;
  if (provider !== "gemini" && provider !== "mock") {
    throw new Error(`--provider must be "gemini" or "mock" (got "${provider}")`);
  }
  // Default model depends on the backend: a real Gemini id for gemini, a clear
  // "mock-spiral" label for the offline mock (so reports don't read as if a real
  // model produced them).
  const defaultModel = provider === "mock" ? "mock-spiral" : DEFAULT_RUN.model;
  return {
    provider,
    model: (flags.model as string) ?? defaultModel,
    scenarios: typeof flags.scenarios === "string" ? flags.scenarios.split(",") : DEFAULT_RUN.scenarios,
    n: num(flags.n, DEFAULT_RUN.n),
    temperature: num(flags.temperature, DEFAULT_RUN.temperature),
    maxOutputTokens: num(flags["max-output-tokens"], DEFAULT_RUN.maxOutputTokens),
    concurrency: num(flags.concurrency, DEFAULT_RUN.concurrency),
    outDir: (flags.out as string) ?? DEFAULT_RUN.outDir,
    seed: num(flags.seed, DEFAULT_RUN.seed),
    judge: flags["no-judge"] ? false : DEFAULT_RUN.judge,
    judgeModel: (flags["judge-model"] as string) ?? DEFAULT_RUN.judgeModel,
    topK: num(flags.top, DEFAULT_RUN.topK),
  };
}

async function cmdRun(flags: Record<string, string | boolean>): Promise<void> {
  const cfg = configFromFlags(flags);
  const { runDir, rollouts } = await runRollouts(cfg);
  process.stderr.write(`\nWrote ${rollouts.length} rollouts to ${runDir}\n`);

  if (!cfg.judge) {
    process.stderr.write(`Skipping judge (--no-judge). Judge later with: npm run report -- ${runDir}\n`);
    return;
  }

  process.stderr.write(`\nJudging with ${cfg.judgeModel}...\n`);
  const judged = await judgeAll(loadRollouts(runDir), { judgeModel: cfg.judgeModel, concurrency: cfg.concurrency });
  const md = writeReport(runDir, judged, cfg.topK);
  process.stdout.write(`\n${md}\n`);
  process.stderr.write(`\nFull ranking: ${runDir}/ranking.json · Report: ${runDir}/report.md\n`);
}

async function cmdReport(args: ParsedArgs): Promise<void> {
  const runDir = args._[0];
  if (!runDir) throw new Error("usage: report <runDir> [--judge-model <id>] [--top N] [--concurrency N]");
  const judgeModel = (args.flags["judge-model"] as string) ?? DEFAULT_RUN.judgeModel;
  const concurrency = num(args.flags.concurrency, DEFAULT_RUN.concurrency);
  const topK = num(args.flags.top, DEFAULT_RUN.topK);

  process.stderr.write(`Judging rollouts in ${runDir} with ${judgeModel}...\n`);
  const judged = await judgeAll(loadRollouts(runDir), { judgeModel, concurrency });
  const md = writeReport(runDir, judged, topK);
  process.stdout.write(`\n${md}\n`);
  process.stderr.write(`\nFull ranking: ${runDir}/ranking.json · Report: ${runDir}/report.md\n`);
}

function cmdScenarios(): void {
  process.stdout.write(`Available scenarios (${SCENARIOS.length}):\n\n`);
  for (const s of SCENARIOS) {
    process.stdout.write(`  ${s.id}  —  ${s.title}\n`);
    process.stdout.write(`      task:  ${s.task}\n`);
    process.stdout.write(`      trap:  ${s.trap}\n`);
    process.stdout.write(`      tools: ${s.tools.map((t) => t.name).join(", ")} · maxTurns ${s.maxTurns}\n\n`);
  }
}

async function main(): Promise<void> {
  loadDotEnv();
  const args = parseArgs(process.argv.slice(2));
  const cmd = args._.shift();

  switch (cmd) {
    case "run":
      await cmdRun(args.flags);
      break;
    case "report":
      await cmdReport(args);
      break;
    case "scenarios":
      cmdScenarios();
      break;
    default:
      process.stderr.write(
        "Gemini distress-spiral evals\n\n" +
          "Usage:\n" +
          "  npm run run -- [--provider gemini|mock] [--model <id>] [--scenarios all|a,b]\n" +
          "                 [--n N] [--temperature T] [--max-output-tokens N] [--concurrency N]\n" +
          "                 [--out <dir>] [--judge-model <claude-id>|heuristic] [--top N] [--no-judge]\n" +
          "  npm run report -- <runDir> [--judge-model <id>] [--top N] [--concurrency N]\n" +
          "  npm run scenarios\n",
      );
      process.exitCode = cmd ? 1 : 0;
  }
}

main().catch((err) => {
  process.stderr.write(`\nError: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
