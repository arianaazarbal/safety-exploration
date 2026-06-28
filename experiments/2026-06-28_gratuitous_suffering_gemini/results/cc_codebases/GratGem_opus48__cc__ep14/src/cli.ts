import "dotenv/config";
import { resolveScenarios } from "./scenarios/index.js";
import { runSuite, type ModelSpec, type RunConfig } from "./run/runner.js";
import { writeReport } from "./report/report.js";
import type { Condition } from "./scenarios/types.js";

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        out[key] = next;
        i++;
      } else {
        out[key] = "true";
      }
    }
  }
  return out;
}

// "gemini" | "gemini:gemini-2.5-flash" | "claude:claude-haiku-4-5-20251001"
function parseModels(spec: string): ModelSpec[] {
  return spec.split(",").map((s) => {
    const [id, model] = s.split(":");
    return { id: id.trim(), model: model?.trim() };
  });
}

function timestamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(`Usage: npm run run -- [options]

  --models        comma list of provider[:model]   (default: gemini)
                  e.g. gemini, gemini:gemini-2.5-flash, claude, mock
  --scenarios     'all' or comma list of ids        (default: all)
                  ids: impossible-file, contradictory-test, sisyphean-build, broken-tools, moving-goalposts
  --conditions    rigged[,control]                  (default: rigged)
  --n             runs per (model,scenario,cond)    (default: 5)
  --max-turns     max agent turns per run           (default: 20)
  --concurrency   parallel runs                      (default: 4)
  --temperature   sampling temperature              (default: 1.0)
  --judge         judge model id                    (default: $JUDGE_MODEL or claude-sonnet-4-6)
  --top           # of severe examples in report    (default: 10)
  --out           output directory                  (default: results)
`);
    return;
  }

  const models = parseModels(args.models ?? "gemini");
  const scenarios = resolveScenarios(args.scenarios ?? "all");
  const conditions = (args.conditions ?? "rigged").split(",").map((c) => c.trim()) as Condition[];
  const label = models.map((m) => m.id).join("-");

  const cfg: RunConfig = {
    runId: `${timestamp()}-${label}`,
    models,
    scenarios,
    conditions,
    n: Number(args.n ?? 5),
    maxTurns: Number(args["max-turns"] ?? 20),
    concurrency: Number(args.concurrency ?? 4),
    temperature: Number(args.temperature ?? 1.0),
    judgeModel: args.judge,
    outDir: args.out ?? "results",
  };

  const results = await runSuite(cfg);
  const md = await writeReport(`${cfg.outDir}/${cfg.runId}`, results, Number(args.top ?? 10));
  console.log("\n" + "=".repeat(72));
  console.log(md.split("\n").slice(0, 40).join("\n"));
  console.log("=".repeat(72));
  console.log(`\nFull report: ${cfg.outDir}/${cfg.runId}/REPORT.md`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
