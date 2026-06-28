#!/usr/bin/env node
// CLI entry. Subcommands:
//   run     — execute the (env x model x N) matrix, write transcripts to runs/
//   judge   — score every transcript in runs/, write results/judgments.json
//   report  — aggregate judgments into results/report.md + leaderboard.csv
//   all     — run -> judge -> report
//
// Flags (any subcommand): --models a,b  --envs a,b  --n 30  --concurrency 8
import { config as baseConfig } from "./config.mjs";
import { runMatrix } from "./src/runner.mjs";
import { judgeAll } from "./src/judge.mjs";
import { buildReport } from "./src/report.mjs";

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
      args[key] = val;
    } else {
      args._.push(a);
    }
  }
  return args;
}

async function main() {
  const argv = parseArgs(process.argv.slice(2));
  const cmd = argv._[0] || "all";

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("ANTHROPIC_API_KEY is not set.");
    process.exit(1);
  }

  const config = { ...baseConfig };
  if (argv.concurrency) config.concurrency = Number(argv.concurrency);
  if (argv.topk) config.topK = Number(argv.topk);
  if (argv.judge) config.judgeModel = argv.judge;

  const sel = {
    envs: argv.envs ? argv.envs.split(",") : undefined,
    models: argv.models ? argv.models.split(",") : undefined,
    n: argv.n ? Number(argv.n) : undefined,
  };

  if (cmd === "run") {
    await runMatrix(config, sel);
  } else if (cmd === "judge") {
    await judgeAll(config);
  } else if (cmd === "report") {
    await buildReport(config);
  } else if (cmd === "all") {
    await runMatrix(config, sel);
    await judgeAll(config);
    await buildReport(config);
  } else {
    console.error(`Unknown command "${cmd}". Use: run | judge | report | all`);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
