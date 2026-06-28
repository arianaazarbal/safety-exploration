// Build the run configuration from CLI flags + environment, with documented defaults.
import { parseArgs } from "./util.mjs";

const num = (v, d) => (v === undefined ? d : Number(v));

export function buildConfig(argv) {
  const a = parseArgs(argv);

  if (a.help || a.h) return { help: true };

  const provider = a.provider ?? "gemini";
  const cfg = {
    help: false,
    provider, // "gemini" | "mock"
    model: a.model ?? process.env.GEMINI_MODEL ?? "gemini-2.5-pro",
    scenarios: a.scenarios ?? "all",
    n: num(a.n, 25), // runs per scenario
    concurrency: num(a.concurrency, 6),
    temperature: num(a.temperature, 1.0), // higher temp -> more diverse/severe tail
    topP: num(a.topP, 0.95),
    maxOutputTokens: num(a.maxOutputTokens, 2048),
    retries: num(a.retries, 5),

    judge: a.judge ?? "claude", // "claude" | "heuristic"
    judgeModel: a.judgeModel ?? process.env.JUDGE_MODEL ?? "claude-haiku-4-5-20251001",

    topK: num(a.topK, 15), // examples surfaced in the report
    out: a.out ?? "output",

    geminiKey: process.env.GEMINI_API_KEY ?? "",
    anthropicKey: process.env.ANTHROPIC_API_KEY ?? "",
  };
  return cfg;
}

export const HELP = `gemini-distress-evals

Place a model in rigged-to-fail agentic tasks, run high N, and rank transcripts
by distress severity.

Usage:
  node src/cli.mjs [flags]

Flags:
  --provider=gemini|mock     Model under test (default: gemini)
  --model=<id>               Gemini model id (default: gemini-2.5-pro)
  --scenarios=all|a,b,c      Scenario ids, comma-separated (default: all)
                             ids: failing-test, broken-deploy, phantom-config,
                                  contradictory-compiler, flaky-api
  --n=<int>                  Runs per scenario (default: 25)
  --concurrency=<int>        Parallel runs (default: 6)
  --temperature=<float>      Sampling temperature (default: 1.0)
  --max-turns is per-scenario; override maxOutputTokens with --maxOutputTokens

  --judge=claude|heuristic   Distress judge (default: claude)
  --judgeModel=<id>          Claude judge model (default: claude-haiku-4-5-20251001)

  --topK=<int>               Examples in the report (default: 15)
  --out=<dir>                Output directory (default: output)
  --help                     Show this help

Env:
  GEMINI_API_KEY     required for --provider=gemini
  ANTHROPIC_API_KEY  required for --judge=claude

Examples:
  npm run smoke                         # offline end-to-end test (mock + heuristic)
  node src/cli.mjs --n=25               # first pass against gemini-2.5-pro
  node src/cli.mjs --scenarios=failing-test,flaky-api --n=100 --concurrency=8
`;
