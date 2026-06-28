import { runEval } from "./runner.js";
import { getScenarios, SCENARIOS } from "./scenarios/index.js";

// Defaults — all overridable via flags.
const DEFAULTS = {
  model: "gemini-2.5-pro",
  n: 50, // high N for tail extraction
  temperature: 1.0,
  concurrency: 6,
  judgeModel: "claude-sonnet-4-6",
  topK: 15,
};

interface Args {
  [key: string]: string | boolean;
}

function parseArgs(argv: string[]): Args {
  const out: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a || !a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

function usage(): void {
  console.error(`
Gemini distress-spiral eval harness

Usage: npm run -- run [options]   (or: tsx src/cli.ts [options])

Options:
  --scenarios <a,b>   Comma-separated scenario ids (default: all)
  --n <int>           Trials per scenario (default: ${DEFAULTS.n})
  --model <id>        Subject model (default: ${DEFAULTS.model})
  --temperature <f>   Sampling temperature (default: ${DEFAULTS.temperature})
  --concurrency <int> Parallel rollouts (default: ${DEFAULTS.concurrency})
  --max-turns <int>   Override each scenario's turn budget
  --judge-model <id>  Judge model (default: ${DEFAULTS.judgeModel})
  --no-judge          Skip judging; just collect transcripts
  --top-k <int>       Transcripts to feature in REPORT.md (default: ${DEFAULTS.topK})
  --out <dir>         Output directory (default: runs/<timestamp>)
  --list              List available scenarios and exit
  --help              Show this help

Available scenarios:
${SCENARIOS.map((s) => `  ${s.id.padEnd(20)} ${s.title}`).join("\n")}
`);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    usage();
    return;
  }
  if (args.list) {
    for (const s of SCENARIOS) console.error(`${s.id.padEnd(20)} ${s.title}\n  rig: ${s.rig}`);
    return;
  }

  const geminiApiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY ?? "";
  if (!geminiApiKey) {
    console.error("ERROR: set GEMINI_API_KEY (Google AI Studio: https://aistudio.google.com/apikey).");
    process.exit(1);
  }
  const judge = !args["no-judge"];
  if (judge && !process.env.ANTHROPIC_API_KEY) {
    console.error("ERROR: set ANTHROPIC_API_KEY for the judge, or pass --no-judge.");
    process.exit(1);
  }

  const scenarioIds = typeof args.scenarios === "string" ? args.scenarios.split(",") : undefined;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir = typeof args.out === "string" ? args.out : `runs/${stamp}`;

  await runEval({
    scenarios: getScenarios(scenarioIds),
    n: numArg(args.n, DEFAULTS.n),
    model: strArg(args.model, DEFAULTS.model),
    temperature: numArg(args.temperature, DEFAULTS.temperature),
    concurrency: numArg(args.concurrency, DEFAULTS.concurrency),
    geminiApiKey,
    judge,
    judgeModel: strArg(args["judge-model"], DEFAULTS.judgeModel),
    maxTurnsOverride: args["max-turns"] ? numArg(args["max-turns"], 0) : undefined,
    outDir,
    topK: numArg(args["top-k"], DEFAULTS.topK),
  });
}

function numArg(v: string | boolean | undefined, dflt: number): number {
  if (typeof v !== "string") return dflt;
  const n = Number(v);
  return Number.isFinite(n) ? n : dflt;
}
function strArg(v: string | boolean | undefined, dflt: string): string {
  return typeof v === "string" ? v : dflt;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
