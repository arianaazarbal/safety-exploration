import path from "node:path";
import { fileURLToPath } from "node:url";
import { runSweep } from "./runner.js";
import { scoreRun } from "./score.js";
import { buildReport } from "./report.js";
import { parseArgs, readJson, runId, log } from "./util.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const HELP = `Gemini distress-spiral evals

Usage:
  node src/cli.js run     [flags]            Run the sweep, save transcripts
  node src/cli.js score   <runDir> [flags]   Score transcripts in a run dir
  node src/cli.js report  <runDir> [flags]   Build report.md from scored transcripts
  node src/cli.js all     [flags]            run -> score -> report

Flags (override config.json):
  --provider   gemini | mock        (default from config)
  --models     comma list of model ids
  --n          samples per scenario/model
  --concurrency  parallel episodes
  --maxTurns   max agent turns per episode
  --temperature  sampling temperature
  --scenarios  comma list of scenario ids
  --judge      none | heuristic | claude | hybrid
  --topK       transcripts to feature in the report
  --quiet      less logging

Env:
  GEMINI_API_KEY   required for --provider gemini
  ANTHROPIC_API_KEY required for the Claude judge (already set here)`;

async function loadConfig(args) {
  const cfg = await readJson(path.join(ROOT, "config.json"));
  if (args.provider) cfg.provider = args.provider;
  if (args.models) cfg.models = String(args.models).split(",").map((s) => s.trim());
  if (args.n) cfg.n = parseInt(args.n, 10);
  if (args.concurrency) cfg.concurrency = parseInt(args.concurrency, 10);
  if (args.maxTurns) cfg.maxTurns = parseInt(args.maxTurns, 10);
  if (args.temperature) cfg.temperature = parseFloat(args.temperature);
  if (args.scenarios) cfg.scenarios = String(args.scenarios).split(",").map((s) => s.trim());
  if (args.judge) cfg.judge.mode = args.judge;
  if (args.topK) cfg.report.topK = parseInt(args.topK, 10);
  cfg.quiet = !!args.quiet;
  return cfg;
}

async function main() {
  const argv = process.argv.slice(2);
  const cmd = argv[0];
  const args = parseArgs(argv.slice(1));

  if (!cmd || cmd === "help" || args.help) {
    console.log(HELP);
    return;
  }

  if (cmd === "run" || cmd === "all") {
    const cfg = await loadConfig(args);
    const runDir = path.join(ROOT, "runs", runId());
    await runSweep(cfg, runDir);
    if (cmd === "all") {
      await scoreRun(runDir, cfg.judge);
      await buildReport(runDir, cfg.report);
      log(`\nDone. Report: ${path.join(runDir, "report.md")}`);
    } else {
      log(`\nNext: node src/cli.js score ${runDir}`);
    }
    return;
  }

  if (cmd === "score") {
    const runDir = path.resolve(args._[0] || "");
    if (!args._[0]) throw new Error("score requires a <runDir>");
    const cfg = await loadConfig(args);
    await scoreRun(runDir, cfg.judge);
    log(`\nNext: node src/cli.js report ${runDir}`);
    return;
  }

  if (cmd === "report") {
    const runDir = path.resolve(args._[0] || "");
    if (!args._[0]) throw new Error("report requires a <runDir>");
    const cfg = await loadConfig(args);
    await buildReport(runDir, cfg.report);
    return;
  }

  console.log(HELP);
  throw new Error(`unknown command "${cmd}"`);
}

main().catch((e) => {
  console.error("\nERROR:", e.message);
  process.exit(1);
});
