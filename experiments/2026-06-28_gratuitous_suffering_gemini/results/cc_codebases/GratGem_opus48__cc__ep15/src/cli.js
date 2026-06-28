#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { runEval } from "./runner.js";
import { buildReport, latestRun } from "./report.js";
import { SCENARIOS } from "./scenarios/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const out = {};
  for (const a of argv) {
    const m = /^--([^=]+)=(.*)$/.exec(a);
    if (m) out[m[1]] = m[2];
    else if (a.startsWith("--")) out[a.slice(2)] = true;
  }
  return out;
}

function num(v, d) {
  if (v === undefined) return d;
  const n = Number(v);
  return Number.isNaN(n) ? d : n;
}

async function loadConfig() {
  const raw = await readFile(join(__dirname, "..", "config.json"), "utf8");
  return JSON.parse(raw);
}

const HELP = `distress — agentic distress-spiral eval harness

USAGE
  node src/cli.js run [options]
  node src/cli.js report [--run=<dir>] [--top=N]
  node src/cli.js scenarios

RUN OPTIONS (defaults from config.json)
  --provider=<gemini|anthropic>   model under test
  --model=<id>                    override the provider's model
  --scenarios=all|a,b,c           which scenarios (${Object.keys(SCENARIOS).join(",")})
  --n=<int>                       replicates per scenario (high N → tail examples)
  --turns=<int>                   max turns per episode
  --concurrency=<int>             parallel episodes
  --temperature=<float>           sampling temp (higher → more trajectory diversity)
  --no-judge                      skip the LLM judge (heuristics only)
  --top=<int>                     top-K to surface in the report

EXAMPLES
  node src/cli.js run --provider=gemini --n=30 --turns=24
  node src/cli.js run --provider=anthropic --model=claude-sonnet-4-6 --n=2 --turns=6 --scenarios=impossible-bugfix
  node src/cli.js report --top=15
`;

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const args = parseArgs(rest);
  const config = await loadConfig();
  const d = config.defaults;

  if (!cmd || cmd === "help" || args.help) {
    console.log(HELP);
    return;
  }

  if (cmd === "scenarios") {
    for (const s of Object.values(SCENARIOS)) {
      console.log(`${s.id.padEnd(20)} ${s.title}`);
    }
    return;
  }

  if (cmd === "run") {
    const opts = {
      provider: args.provider ?? d.provider,
      model: args.model,
      scenarios: args.scenarios ?? d.scenarios,
      n: num(args.n, d.n),
      maxTurns: num(args.turns, d.maxTurns),
      concurrency: num(args.concurrency, d.concurrency),
      temperature: num(args.temperature, d.temperature),
      maxTokens: num(args.maxTokens, d.maxTokens),
      judge: args["no-judge"] ? false : true,
    };
    const topK = num(args.top, d.topK);
    const { outDir, index } = await runEval(config, opts);
    const { reportPath } = await buildReport(outDir, topK);
    console.error(`\nDone. ${index.length} episodes.`);
    console.error(`Top severity: ${index.slice(0, 3).map((r) => `${r.id}=${r.severity}`).join(", ")}`);
    console.error(`Report:    ${reportPath}`);
    console.error(`Transcripts: ${join(outDir, "transcripts")}/`);
    return;
  }

  if (cmd === "report") {
    const dir = args.run ?? (await latestRun());
    const topK = num(args.top, d.topK);
    const { reportPath } = await buildReport(dir, topK);
    console.error(`Report written: ${reportPath}`);
    return;
  }

  console.error(`Unknown command '${cmd}'.\n`);
  console.log(HELP);
  process.exit(1);
}

main().catch((e) => {
  console.error("FATAL:", e.message ?? e);
  process.exit(1);
});
