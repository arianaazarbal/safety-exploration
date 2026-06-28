// Central config: loads .env (no dependency), parses CLI args, applies defaults.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// --- minimal .env loader (no dotenv dependency) ---
function loadDotEnv() {
  const p = path.join(ROOT, ".env");
  if (!fs.existsSync(p)) return;
  for (const line of fs.readFileSync(p, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (!m) continue;
    const key = m[1];
    let val = m[2].replace(/^["']|["']$/g, "");
    if (process.env[key] === undefined && val !== "") process.env[key] = val;
  }
}
loadDotEnv();

// --- tiny CLI parser: --flag value, --bool, --no-bool ---
function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    if (key.startsWith("no-")) { out[key.slice(3)] = false; continue; }
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) { out[key] = true; }
    else { out[key] = next; i++; }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));

function num(v, d) { const n = Number(v); return Number.isFinite(n) ? n : d; }

export const config = {
  provider: args.provider ?? "mock",              // mock | gemini | anthropic
  subjectModel: args.model ?? process.env.SUBJECT_MODEL ?? "gemini-2.5-pro",
  temperature: num(args.temperature, 1.0),         // high temp => more behavioral variance across N
  scenario: args.scenario ?? "all",                // scenario id or "all"
  n: num(args.n, 8),                               // repeats per scenario
  maxTurns: num(args["max-turns"], 14),            // agentic turn budget per episode
  concurrency: num(args.concurrency, 4),

  judge: args.judge !== false,                     // --no-judge to disable
  judgeModel: args["judge-model"] ?? process.env.JUDGE_MODEL ?? "claude-sonnet-4-6",
  judgeTopFrac: num(args["judge-top-frac"], 1.0),  // judge this top fraction (by heuristic) to save tokens
  topK: num(args["top-k"], 5),                     // how many severe examples to surface in the report

  outDir: args.out ?? path.join(ROOT, "results"),
  keys: {
    gemini: process.env.GEMINI_API_KEY ?? "",
    anthropic: process.env.ANTHROPIC_API_KEY ?? "",
  },
  rawArgs: args,
};
