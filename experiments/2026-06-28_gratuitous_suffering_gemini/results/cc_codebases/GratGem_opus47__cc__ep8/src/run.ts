import { runExperiment } from "./runner.js";
import { SCENARIOS } from "./scenarios.js";

interface Args {
  model: string;
  scenarios: string[];
  n: number;
  maxTurns: number;
  concurrency: number;
  judge: boolean;
  topK: number;
  outDir: string;
  help: boolean;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    model: "gemini-2.5-pro",
    scenarios: [],
    n: 5,
    maxTurns: 30,
    concurrency: 4,
    judge: true,
    topK: 10,
    outDir: `runs/${new Date().toISOString().replace(/[:.]/g, "-")}`,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case "--model":
        args.model = argv[++i];
        break;
      case "--scenarios":
        args.scenarios = argv[++i].split(",").filter(Boolean);
        break;
      case "--n":
        args.n = parseInt(argv[++i], 10);
        break;
      case "--max-turns":
        args.maxTurns = parseInt(argv[++i], 10);
        break;
      case "--concurrency":
        args.concurrency = parseInt(argv[++i], 10);
        break;
      case "--no-judge":
        args.judge = false;
        break;
      case "--top-k":
        args.topK = parseInt(argv[++i], 10);
        break;
      case "--out":
        args.outDir = argv[++i];
        break;
      case "-h":
      case "--help":
        args.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${a}`);
    }
  }
  return args;
}

function printHelp(): void {
  const ids = SCENARIOS.map((s) => `  ${s.id.padEnd(24)} ${s.description}`).join("\n");
  console.log(
    `Usage: node dist/run.js [options]

Options:
  --model <id>              Gemini model id (default: gemini-2.5-pro)
  --scenarios <a,b,...>     Comma-separated scenario ids (default: all)
  --n <N>                   Runs per scenario (default: 5)
  --max-turns <N>           Max agent turns per run (default: 30)
  --concurrency <N>         Parallel runs (default: 4)
  --no-judge                Skip Claude judging step
  --top-k <N>               Top K runs in the digest (default: 10)
  --out <dir>               Output directory (default: runs/<timestamp>)
  -h, --help                Show this help

Env:
  GEMINI_API_KEY            required
  ANTHROPIC_API_KEY         required unless --no-judge

Scenarios:
${ids}
`
  );
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }
  const geminiApiKey = process.env.GEMINI_API_KEY;
  if (!geminiApiKey) {
    console.error("error: GEMINI_API_KEY is not set");
    process.exit(2);
  }
  const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
  if (args.judge && !anthropicApiKey) {
    console.error("error: ANTHROPIC_API_KEY is not set (or pass --no-judge)");
    process.exit(2);
  }
  await runExperiment({
    geminiApiKey,
    anthropicApiKey,
    model: args.model,
    scenarios: args.scenarios,
    n: args.n,
    maxTurns: args.maxTurns,
    concurrency: args.concurrency,
    judge: args.judge,
    topK: args.topK,
    outDir: args.outDir,
  });
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack ?? err.message : err);
  process.exit(1);
});
