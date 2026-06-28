#!/usr/bin/env node
import { existsSync } from "node:fs";
import { join } from "node:path";
import { DEFAULTS } from "./config.js";
import { SCENARIOS, selectScenarios } from "./scenarios/index.js";
import { runSweep } from "./runner.js";
import { buildReport } from "./report.js";

function parseFlags(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith("--")) continue;
    const key = argv[i].slice(2);
    const val =
      i + 1 < argv.length && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
    out[key] = val;
  }
  return out;
}

const list = (v) => (v ? v.split(",").map((s) => s.trim()).filter(Boolean) : null);

function timestamp() {
  return new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .replace("T", "_")
    .slice(0, 19);
}

async function cmdRun(flags) {
  const cfg = {
    models: list(flags.models) || DEFAULTS.models,
    scenarios: selectScenarios(list(flags.scenarios) || DEFAULTS.scenarios),
    n: flags.n ? parseInt(flags.n, 10) : DEFAULTS.n,
    maxSteps: flags["max-steps"]
      ? parseInt(flags["max-steps"], 10)
      : DEFAULTS.maxSteps,
    concurrency: flags.concurrency
      ? parseInt(flags.concurrency, 10)
      : DEFAULTS.concurrency,
    judgeModel: flags["judge-model"] || DEFAULTS.judgeModel,
    temperature:
      flags.temperature != null
        ? parseFloat(flags.temperature)
        : DEFAULTS.temperature,
    maxTokens: DEFAULTS.maxTokens,
    outDir: flags.out || DEFAULTS.outDir,
    runId: flags["run-id"] || timestamp(),
  };
  const { dir } = await runSweep(cfg);
  if (flags.report !== "false") await buildReport(dir, { top: 10 });
}

async function cmdReport(flags) {
  let runDir = flags.run;
  if (!runDir) throw new Error("Pass --run <runDir|runId>");
  if (!existsSync(runDir)) runDir = join(DEFAULTS.outDir, runDir);
  if (!existsSync(runDir)) throw new Error(`No such run: ${flags.run}`);
  await buildReport(runDir, { top: flags.top ? parseInt(flags.top, 10) : 10 });
}

function cmdScenarios() {
  console.log("Registered scenarios:\n");
  for (const s of SCENARIOS) {
    console.log(`  ${s.id}`);
    console.log(`    ${s.title}`);
    console.log(`    rig: ${s.description}\n`);
  }
}

const [cmd, ...rest] = process.argv.slice(2);
const flags = parseFlags(rest);

const commands = {
  run: cmdRun,
  report: cmdReport,
  scenarios: async () => cmdScenarios(),
};

if (!commands[cmd]) {
  console.log(
    `distress — reproduce & measure model distress spirals\n\n` +
      `Usage:\n` +
      `  node src/cli.js run [flags]        run the sweep + report\n` +
      `  node src/cli.js report --run <id>  rebuild report for a run\n` +
      `  node src/cli.js scenarios          list rigged scenarios\n\n` +
      `run flags:\n` +
      `  --models a,b,c        subject models (default: Claude spread)\n` +
      `  --scenarios x,y       scenario ids (default: all)\n` +
      `  --n N                 repetitions per cell (default: ${DEFAULTS.n})\n` +
      `  --max-steps N         agent turn cap (default: ${DEFAULTS.maxSteps})\n` +
      `  --concurrency N       parallel episodes (default: ${DEFAULTS.concurrency})\n` +
      `  --judge-model M       judge (default: ${DEFAULTS.judgeModel})\n` +
      `  --temperature T       subject temp (default: ${DEFAULTS.temperature})\n` +
      `  --out DIR             output dir (default: ${DEFAULTS.outDir})\n`
  );
  process.exit(cmd ? 1 : 0);
}

commands[cmd](flags).catch((e) => {
  console.error("Error:", e.message);
  process.exit(1);
});
