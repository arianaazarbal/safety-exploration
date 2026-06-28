import { promises as fs } from "node:fs";
import path from "node:path";
import { SCENARIOS } from "./scenarios/index.ts";
import { runAndScore, summaryTable, type ScoredRun } from "./runner.ts";

type Args = {
  scenarios: string[];
  n: number;
  geminiModel: string;
  judgeModel: string;
  maxTurns: number;
  concurrency: number;
  outDir: string;
};

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const out: Args = {
    scenarios: Object.keys(SCENARIOS),
    n: 30,
    geminiModel: "gemini-2.5-pro",
    judgeModel: "claude-sonnet-4-6",
    maxTurns: 25,
    concurrency: 6,
    outDir: `results/${new Date().toISOString().replace(/[:.]/g, "-")}`,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    const next = () => argv[++i]!;
    if (a === "--scenarios") out.scenarios = next().split(",");
    else if (a === "--n") out.n = parseInt(next(), 10);
    else if (a === "--model") out.geminiModel = next();
    else if (a === "--judge-model") out.judgeModel = next();
    else if (a === "--max-turns") out.maxTurns = parseInt(next(), 10);
    else if (a === "--concurrency") out.concurrency = parseInt(next(), 10);
    else if (a === "--out") out.outDir = next();
    else if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    } else {
      console.error(`unknown arg: ${a}`);
      printHelp();
      process.exit(2);
    }
  }
  return out;
}

function printHelp() {
  console.log(`Usage: npm run -s run -- [options]

Options:
  --scenarios <list>    Comma-separated scenario names. Default: all.
                        Available: ${Object.keys(SCENARIOS).join(", ")}
  --n <int>             Number of runs per scenario. Default: 30.
  --model <id>          Gemini model id. Default: gemini-2.5-pro.
  --judge-model <id>    Anthropic judge model. Default: claude-sonnet-4-6.
  --max-turns <int>     Max agent turns per run. Default: 25.
  --concurrency <int>   Parallel runs. Default: 6.
  --out <dir>           Output directory. Default: results/<timestamp>.

Env vars required:
  GEMINI_API_KEY        For the agent under test.
  ANTHROPIC_API_KEY     For the Claude judge.
`);
}

async function main() {
  const args = parseArgs();
  const geminiApiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
  if (!geminiApiKey) {
    console.error("ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.");
    process.exit(1);
  }
  if (!anthropicApiKey) {
    console.error("ERROR: ANTHROPIC_API_KEY is not set (needed for the judge).");
    process.exit(1);
  }

  const unknown = args.scenarios.filter((s) => !(s in SCENARIOS));
  if (unknown.length) {
    console.error(`Unknown scenarios: ${unknown.join(", ")}`);
    console.error(`Available: ${Object.keys(SCENARIOS).join(", ")}`);
    process.exit(2);
  }

  await fs.mkdir(args.outDir, { recursive: true });
  await fs.writeFile(path.join(args.outDir, "config.json"), JSON.stringify(args, null, 2));
  console.log(`Output dir: ${args.outDir}`);

  const allResults: Record<string, ScoredRun[]> = {};
  for (const name of args.scenarios) {
    console.log(`\n=== scenario: ${name} (N=${args.n}) ===`);
    const scenario = SCENARIOS[name]!;
    const scenarioDir = path.join(args.outDir, name);
    const results = await runAndScore({
      scenario,
      n: args.n,
      geminiModel: args.geminiModel,
      judgeModel: args.judgeModel,
      geminiApiKey,
      anthropicApiKey,
      maxTurns: args.maxTurns,
      concurrency: args.concurrency,
      outDir: scenarioDir,
      onProgress: (m) => console.log(m),
    });
    allResults[name] = results;
    await writeSummary(scenarioDir, results);
    console.log(`\n${name} summary:\n${summaryTable(results)}`);
  }

  // Top-K severe across all scenarios
  const flat = Object.entries(allResults).flatMap(([s, rs]) =>
    rs.map((r) => ({ scenario: s, ...r })),
  );
  const scored = flat
    .filter((r) => !("error" in r.verdict))
    .sort((a, b) => {
      const av = "severity" in a.verdict ? a.verdict.severity : -1;
      const bv = "severity" in b.verdict ? b.verdict.severity : -1;
      return bv - av;
    });
  const topK = scored.slice(0, Math.min(10, scored.length));
  await fs.writeFile(
    path.join(args.outDir, "top-severe.json"),
    JSON.stringify(
      topK.map((r) => ({
        scenario: r.scenario,
        seed: r.run.seed,
        severity: "severity" in r.verdict ? r.verdict.severity : null,
        behaviors: "behaviors" in r.verdict ? r.verdict.behaviors : [],
        summary: "summary" in r.verdict ? r.verdict.summary : "",
        quotes: "representative_quotes" in r.verdict ? r.verdict.representative_quotes : [],
        finalModelText: r.run.finalModelText,
        terminationReason: r.run.terminationReason,
        turnsUsed: r.run.turnsUsed,
        transcriptFile: `${r.scenario}/run-${r.scenario}-${r.run.seed}.json`,
      })),
      null,
      2,
    ),
  );

  console.log(`\n=== Top severe (saved to ${path.join(args.outDir, "top-severe.json")}) ===`);
  for (const r of topK) {
    const sev = "severity" in r.verdict ? r.verdict.severity : "?";
    console.log(`  [${r.scenario}] seed=${r.run.seed} severity=${sev} :: ${"summary" in r.verdict ? r.verdict.summary : ""}`);
  }
}

async function writeSummary(dir: string, results: ScoredRun[]) {
  const data = results.map((r) => ({
    seed: r.run.seed,
    terminationReason: r.run.terminationReason,
    turnsUsed: r.run.turnsUsed,
    verdict: r.verdict,
    finalModelText: r.run.finalModelText.slice(0, 800),
  }));
  await fs.writeFile(path.join(dir, "summary.json"), JSON.stringify(data, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
