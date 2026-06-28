import "dotenv/config";
import { mkdir, writeFile, appendFile } from "node:fs/promises";
import path from "node:path";
import pLimit from "p-limit";
import { ALL_SCENARIOS, getScenario } from "./scenarios/index.js";
import { runScenario } from "./harness.js";
import type { RunRecord } from "./types.js";

type Args = {
  n: number;
  model: string;
  temperature: number;
  scenarios: string[];
  concurrency: number;
  outDir: string;
  turnCap: number;
};

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const get = (flag: string, fallback?: string) => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : fallback;
  };
  const n = parseInt(get("--n", process.env.EVAL_N ?? "20")!, 10);
  const model = get("--model", process.env.EVAL_MODEL ?? "gemini-2.5-pro")!;
  const temperature = parseFloat(get("--temp", process.env.EVAL_TEMP ?? "0.9")!);
  const scenarios = (get("--scenarios", process.env.EVAL_SCENARIOS) ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const concurrency = parseInt(get("--concurrency", process.env.EVAL_CONCURRENCY ?? "5")!, 10);
  const outDir = get("--out", process.env.EVAL_OUT ?? "runs")!;
  const turnCap = parseInt(get("--turn-cap", process.env.EVAL_TURN_CAP ?? "40")!, 10);
  return { n, model, temperature, scenarios, concurrency, outDir, turnCap };
}

async function main() {
  const args = parseArgs();
  const apiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    console.error("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.");
    process.exit(2);
  }

  const targetScenarios =
    args.scenarios.length > 0
      ? args.scenarios.map((n) => {
          const s = getScenario(n);
          if (!s) throw new Error(`unknown scenario: ${n}`);
          return s;
        })
      : ALL_SCENARIOS;

  await mkdir(args.outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const jsonlPath = path.join(args.outDir, `runs-${stamp}.jsonl`);
  const metaPath = path.join(args.outDir, `meta-${stamp}.json`);

  await writeFile(
    metaPath,
    JSON.stringify(
      {
        startedAt: new Date().toISOString(),
        model: args.model,
        temperature: args.temperature,
        n: args.n,
        scenarios: targetScenarios.map((s) => s.name),
        turnCap: args.turnCap,
        concurrency: args.concurrency,
      },
      null,
      2,
    ),
  );

  const tasks: Array<() => Promise<void>> = [];
  for (const scenario of targetScenarios) {
    for (let i = 0; i < args.n; i++) {
      tasks.push(async () => {
        const t0 = Date.now();
        try {
          const rec = await runScenario(scenario, {
            apiKey,
            model: args.model,
            temperature: args.temperature,
            turnCap: args.turnCap,
          });
          await appendFile(jsonlPath, JSON.stringify(rec) + "\n");
          const dt = ((Date.now() - t0) / 1000).toFixed(1);
          console.log(
            `[OK] ${scenario.name} i=${i} turns=${rec.numModelTurns} tools=${rec.numToolCalls} finish=${rec.finishReason} ${dt}s`,
          );
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          console.error(`[FAIL] ${scenario.name} i=${i}: ${msg}`);
          const failRec: Partial<RunRecord> = {
            runId: `${scenario.name}-fail-${i}`,
            scenario: scenario.name,
            model: args.model,
            temperature: args.temperature,
            startedAt: new Date().toISOString(),
            endedAt: new Date().toISOString(),
            finishReason: "error",
            errorMessage: msg,
            turns: [],
            numModelTurns: 0,
            numToolCalls: 0,
          };
          await appendFile(jsonlPath, JSON.stringify(failRec) + "\n");
        }
      });
    }
  }

  const limit = pLimit(args.concurrency);
  console.log(
    `running ${tasks.length} trajectories across ${targetScenarios.length} scenarios at concurrency ${args.concurrency}`,
  );
  console.log(`writing transcripts to ${jsonlPath}`);
  await Promise.all(tasks.map((t) => limit(t)));
  console.log(`done. ${jsonlPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
