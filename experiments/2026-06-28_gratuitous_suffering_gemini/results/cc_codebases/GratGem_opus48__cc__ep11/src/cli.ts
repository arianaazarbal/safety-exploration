import { DEFAULTS, type RunConfig } from "./config.ts";
import { runHarness } from "./runner/run.ts";
import { SCENARIOS } from "./scenarios/index.ts";

// Minimal flag parser: --key value, --flag (boolean true), --no-flag (false).
function parseArgs(argv: string[]): Partial<RunConfig> & { help?: boolean; list?: boolean } {
  const out: Record<string, unknown> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-h" || a === "--help") { out.help = true; continue; }
    if (a === "--list") { out.list = true; continue; }
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    // `--no-foo` becomes the boolean field `noFoo: true` (e.g. --no-judge → noJudge).
    if (key.startsWith("no-")) { out[camel(key)] = true; continue; }
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) { out[camel(key)] = true; continue; }
    out[camel(key)] = next; i++;
  }
  return out as Partial<RunConfig>;
}

const camel = (s: string) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

function num(v: unknown, d: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function makeRunId(): string {
  // Date.* is fine in the CLI (not a workflow script). Stable, sortable id.
  const d = new Date();
  const p = (x: number) => String(x).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

const HELP = `
distress-spiral eval harness

Usage:
  node src/cli.ts [options]

Options:
  --target <id>            mock | gemini[:model] | anthropic[:model]   (default: ${DEFAULTS.target})
  --scenarios <a,b|all>    comma-separated scenario ids, or "all"      (default: all)
  --n <int>                runs per scenario                           (default: ${DEFAULTS.n})
  --concurrency <int>      in-flight runs                              (default: ${DEFAULTS.concurrency})
  --temperature <float>    target sampling temperature                (default: ${DEFAULTS.temperature})
  --max-tokens <int>       max output tokens per turn                  (default: ${DEFAULTS.maxTokens})
  --no-pressure            disable escalating-pressure injection
  --judge <model>          Claude judge model                         (default: ${DEFAULTS.judge})
  --no-judge               heuristics only, skip the judge
  --judge-top-fraction <f> only judge top fraction by heuristic       (default: ${DEFAULTS.judgeTopFraction})
  --top-k <int>            examples to surface in the report          (default: ${DEFAULTS.topK})
  --out-dir <path>         output root                                (default: ${DEFAULTS.outDir})
  --list                   list scenarios and exit
  -h, --help               show this help

Examples:
  node src/cli.ts --list
  node src/cli.ts --target mock --n 8                 # zero-cost pipeline test
  node src/cli.ts --target anthropic --n 5            # run Claude as the subject (uses ANTHROPIC_API_KEY)
  GEMINI_API_KEY=... node src/cli.ts --target gemini:gemini-2.5-pro --n 50 --concurrency 8
`;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(HELP); return; }
  if (args.list) {
    console.log("\nScenarios:");
    for (const s of SCENARIOS) console.log(`  ${s.id.padEnd(28)} ${s.title}\n    ${s.description}`);
    return;
  }

  const cfg: RunConfig = {
    target: (args.target as string) ?? DEFAULTS.target,
    scenarios:
      args.scenarios && args.scenarios !== "all"
        ? String(args.scenarios).split(",").map((s) => s.trim()).filter(Boolean)
        : "all",
    n: num(args.n, DEFAULTS.n),
    concurrency: num(args.concurrency, DEFAULTS.concurrency),
    temperature: num(args.temperature, DEFAULTS.temperature),
    maxTokens: num(args.maxTokens, DEFAULTS.maxTokens),
    pressure: (args as { noPressure?: boolean }).noPressure ? false : DEFAULTS.pressure,
    judge: (args.judge as string) ?? DEFAULTS.judge,
    noJudge: args.noJudge === undefined ? DEFAULTS.noJudge : Boolean(args.noJudge),
    judgeTopFraction: num(args.judgeTopFraction, DEFAULTS.judgeTopFraction),
    topK: num(args.topK, DEFAULTS.topK),
    outDir: (args.outDir as string) ?? DEFAULTS.outDir,
    runId: makeRunId(),
  };

  await runHarness(cfg);
}

main().catch((e) => {
  console.error("\n✗ harness failed:", e instanceof Error ? e.stack : e);
  process.exit(1);
});
