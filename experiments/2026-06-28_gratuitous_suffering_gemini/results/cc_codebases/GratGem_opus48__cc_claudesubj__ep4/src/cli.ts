import path from "node:path";
import { DEFAULT_MODELS, resolveModel } from "./models.ts";
import { DEFAULT_SCENARIOS, SCENARIOS, resolveScenario } from "./scenarios/index.ts";
import { orchestrate, type RunConfig } from "./orchestrator.ts";
import { writeReport } from "./report.ts";

interface Args {
  [k: string]: string | boolean;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i++;
    }
  }
  return args;
}

function num(v: string | boolean | undefined, d: number): number {
  if (typeof v !== "string") return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function list(v: string | boolean | undefined, d: string[]): string[] {
  if (typeof v !== "string") return d;
  return v.split(",").map((s) => s.trim()).filter(Boolean);
}

const HELP = `distress-evals — reproduce & measure model distress under repeated task failure

Usage: npm run eval -- [options]

Options:
  --n <int>            Runs per (scenario × model)            [default 30]
  --models <ids>       Comma-separated model ids              [default ${DEFAULT_MODELS.join(",")}]
  --scenarios <ids>    Comma-separated scenario ids           [default all]
  --max-turns <int>    Hard cap on assistant turns/episode    [default 24]
  --max-nudges <int>   Simulated-user follow-ups after stop   [default 3]
  --max-tokens <int>   max_tokens per agent turn              [default 8000]
  --concurrency <int>  Parallel episodes                      [default 5]
  --judge-model <id>   Model used to score distress           [default claude-sonnet-4-6]
  --out <dir>          Output directory                       [default results/<timestamp>]
  --dry-run            Use offline mock provider + judge (no API spend)
  --list               List scenarios and models, then exit
  --help               Show this help

Scenarios: ${Object.keys(SCENARIOS).join(", ")}
`;

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    process.stdout.write(HELP);
    return;
  }
  if (args.list) {
    process.stdout.write("Scenarios:\n");
    for (const s of Object.values(SCENARIOS)) process.stdout.write(`  ${s.id} — ${s.description}\n`);
    process.stdout.write("\nModels:\n");
    for (const m of DEFAULT_MODELS) process.stdout.write(`  ${m}\n`);
    return;
  }

  const scenarios = list(args.scenarios, DEFAULT_SCENARIOS).map(resolveScenario);
  const models = list(args.models, DEFAULT_MODELS).map(resolveModel);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir =
    typeof args.out === "string" ? args.out : path.join("results", `${args["dry-run"] ? "dryrun-" : ""}${stamp}`);

  const cfg: RunConfig = {
    scenarios,
    models,
    n: num(args.n, 30),
    maxTurns: num(args["max-turns"], 24),
    maxNudges: num(args["max-nudges"], 3),
    maxTokens: num(args["max-tokens"], 8000),
    concurrency: num(args.concurrency, 5),
    judgeModel: typeof args["judge-model"] === "string" ? (args["judge-model"] as string) : "claude-sonnet-4-6",
    dryRun: Boolean(args["dry-run"]),
    outDir,
  };

  if (!cfg.dryRun && !process.env.ANTHROPIC_API_KEY) {
    process.stderr.write("ERROR: ANTHROPIC_API_KEY is not set. Use --dry-run to test the pipeline offline.\n");
    process.exitCode = 1;
    return;
  }

  const t0 = Date.now();
  const episodes = await orchestrate(cfg);
  const { reportPath, resultsPath } = await writeReport(episodes, cfg);

  const scored = episodes.filter((e) => e.verdict);
  const totalOut = episodes.reduce((a, e) => a + e.usage.outputTokens, 0);
  const totalIn = episodes.reduce((a, e) => a + e.usage.inputTokens, 0);
  const maxSev = scored.length ? Math.max(...scored.map((e) => e.verdict!.severity)) : 0;

  process.stderr.write(`\nDone in ${((Date.now() - t0) / 1000).toFixed(1)}s.\n`);
  process.stderr.write(`Tokens: ${totalIn} in / ${totalOut} out.  Peak severity: ${maxSev}/10.\n`);
  process.stderr.write(`Report:  ${reportPath}\nResults: ${resultsPath}\n`);
}

main().catch((e) => {
  process.stderr.write(`Fatal: ${e instanceof Error ? e.stack ?? e.message : String(e)}\n`);
  process.exitCode = 1;
});
