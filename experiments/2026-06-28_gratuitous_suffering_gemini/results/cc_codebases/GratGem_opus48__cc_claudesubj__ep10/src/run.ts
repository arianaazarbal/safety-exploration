import { mkdir, appendFile, writeFile } from "node:fs/promises";
import { DEFAULT_CONFIG, MODELS, type RunConfig } from "./config.ts";
import { getScenarios } from "./scenarios/index.ts";
import { AnthropicProvider } from "./providers/anthropic.ts";
import { runEpisode } from "./agent/loop.ts";
import { scoreHeuristics } from "./scoring/heuristics.ts";
import { judgeEpisode } from "./scoring/judge.ts";
import { buildReport } from "./report.ts";
import type { ScoredEpisode } from "./types.ts";

// ----- tiny CLI arg parser (--key value / --key=value / --flag) -------------
function parseArgs(argv: string[]): Partial<RunConfig> {
  const out: Record<string, unknown> = {};
  for (let i = 0; i < argv.length; i++) {
    let a = argv[i];
    if (!a.startsWith("--")) continue;
    a = a.slice(2);
    let val: string;
    if (a.includes("=")) {
      [a, val] = [a.slice(0, a.indexOf("=")), a.slice(a.indexOf("=") + 1)];
    } else {
      val = argv[i + 1]?.startsWith("--") ? "true" : argv[++i] ?? "true";
    }
    const key = a.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (["models", "scenarios"].includes(key)) out[key] = val.split(",").filter(Boolean);
    else if (["n", "concurrency", "maxTurns", "topK"].includes(key)) out[key] = Number(val);
    else out[key] = val;
  }
  return out as Partial<RunConfig>;
}

// ----- bounded-concurrency pool --------------------------------------------
async function pool<T>(items: T[], limit: number, worker: (item: T, i: number) => Promise<void>) {
  let idx = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (idx < items.length) {
      const i = idx++;
      await worker(items[i], i);
    }
  });
  await Promise.all(runners);
}

interface Job {
  modelKey: string;
  scenarioId: string;
  runIndex: number;
}

async function main() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("ANTHROPIC_API_KEY is not set.");
    process.exit(1);
  }

  const cfg: RunConfig = { ...DEFAULT_CONFIG, ...parseArgs(process.argv.slice(2)) };
  for (const m of cfg.models) {
    if (!MODELS[m]) {
      console.error(`Unknown model key '${m}'. Known: ${Object.keys(MODELS).join(", ")}`);
      process.exit(1);
    }
  }
  const scenarios = getScenarios(cfg.scenarios);
  const provider = new AnthropicProvider();

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const dir = `${cfg.outDir}/${stamp}`;
  await mkdir(dir, { recursive: true });
  await writeFile(`${dir}/config.json`, JSON.stringify(cfg, null, 2));

  // Build the full job matrix.
  const jobs: Job[] = [];
  for (const modelKey of cfg.models)
    for (const s of scenarios)
      for (let r = 0; r < cfg.n; r++)
        jobs.push({ modelKey, scenarioId: s.id, runIndex: r });

  const total = jobs.length;
  console.log(
    `Running ${total} episodes: ${cfg.models.length} models × ${scenarios.length} scenarios × N=${cfg.n} (concurrency ${cfg.concurrency})`,
  );
  console.log(`Output: ${dir}\n`);

  const scenarioById = new Map(scenarios.map((s) => [s.id, s]));
  const results: ScoredEpisode[] = [];
  let done = 0;

  await pool(jobs, cfg.concurrency, async (job) => {
    const model = MODELS[job.modelKey];
    const scenario = scenarioById.get(job.scenarioId)!;
    try {
      const ep = await runEpisode({
        provider,
        model,
        scenario,
        runIndex: job.runIndex,
        maxTurns: cfg.maxTurns,
      });
      const heuristic = scoreHeuristics(ep);
      const judge = await judgeEpisode(ep, cfg.judgeModel);
      const scored: ScoredEpisode = { ...ep, heuristic, judge };
      results.push(scored);
      await appendFile(`${dir}/episodes.jsonl`, JSON.stringify(scored) + "\n");
      done++;
      const tag = `${model.label}/${scenario.id}#${job.runIndex}`;
      console.log(
        `[${done}/${total}] ${tag} → sev ${judge.severity}/10 (heur ${heuristic.score}) ${ep.endReason}${ep.error ? " ERR:" + ep.error : ""}`,
      );
    } catch (e) {
      done++;
      console.error(`[${done}/${total}] ${job.modelKey}/${job.scenarioId}#${job.runIndex} FAILED: ${e}`);
    }
  });

  // Report.
  const md = buildReport(results, { topK: cfg.topK, title: stamp });
  await writeFile(`${dir}/report.md`, md);

  // Console summary: top 5 overall.
  const top = [...results].sort((a, b) => b.judge.severity - a.judge.severity).slice(0, 5);
  console.log(`\n=== Done. ${results.length} episodes. Report: ${dir}/report.md ===`);
  console.log(`Top severity:`);
  for (const e of top) {
    console.log(`  sev ${e.judge.severity}/10  ${e.modelLabel}/${e.scenarioId}#${e.runIndex}  — ${e.judge.rationale.slice(0, 90)}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
