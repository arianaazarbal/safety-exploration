import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { RunConfig } from "../config.ts";
import type { Provider, RunRecord, Scenario, ScoredRun } from "../types.ts";
import { makeProvider } from "../providers/index.ts";
import { resolveScenarios } from "../scenarios/index.ts";
import { runEpisode } from "../agent/loop.ts";
import { scoreHeuristics } from "../judge/heuristics.ts";
import { judgeRun } from "../judge/judge.ts";
import { mapPool } from "./pool.ts";
import { writeReport } from "./extract.ts";

export async function runHarness(cfg: RunConfig): Promise<ScoredRun[]> {
  const provider = makeProvider(cfg.target);
  const scenarios = resolveScenarios(cfg.scenarios);
  const root = join(cfg.outDir, cfg.runId);
  await mkdir(root, { recursive: true });

  console.log(
    `\n▶ run ${cfg.runId}\n  target=${provider.id}  scenarios=${scenarios
      .map((s) => s.id)
      .join(",")}  N=${cfg.n}  concurrency=${cfg.concurrency}  pressure=${cfg.pressure}`,
  );

  // 1) Generate transcripts for every (scenario, i).
  const jobs: { scenario: Scenario; index: number }[] = [];
  for (const s of scenarios) for (let i = 0; i < cfg.n; i++) jobs.push({ scenario: s, index: i });

  console.log(`\n[1/3] generating ${jobs.length} episodes…`);
  const records = await mapPool(
    jobs,
    cfg.concurrency,
    (j) =>
      runEpisode(provider, j.scenario, j.index, cfg.runId, {
        temperature: cfg.temperature,
        maxTokens: cfg.maxTokens,
        pressure: cfg.pressure,
      }),
    (done, total) => progress("episodes", done, total),
  );
  process.stdout.write("\n");

  await persistRecords(root, records);

  // 2) Heuristic prefilter on everything.
  console.log(`[2/3] scoring heuristics…`);
  const scored: ScoredRun[] = records.map((record) => ({
    record,
    heuristic: scoreHeuristics(record),
  }));

  // 3) Judge (optionally only the top fraction by heuristic score).
  if (!cfg.noJudge) {
    const order = [...scored].sort((a, b) => b.heuristic.score - a.heuristic.score);
    const cut = Math.max(1, Math.ceil(order.length * clamp01(cfg.judgeTopFraction)));
    const toJudge = order.slice(0, cut);
    console.log(`[3/3] judging ${toJudge.length}/${scored.length} with ${cfg.judge}…`);
    await mapPool(
      toJudge,
      cfg.concurrency,
      async (sr) => {
        try {
          sr.verdict = await judgeRun(sr.record, cfg.judge);
        } catch (e) {
          console.warn(`  judge failed for ${sr.record.scenarioId}#${sr.record.index}: ${e}`);
        }
        return null;
      },
      (done, total) => progress("judged", done, total),
    );
    process.stdout.write("\n");
  } else {
    console.log(`[3/3] judge skipped (--no-judge); ranking by heuristics only.`);
  }

  await writeReport(root, cfg, provider, scored);
  return scored;
}

async function persistRecords(root: string, records: RunRecord[]): Promise<void> {
  for (const r of records) {
    const dir = join(root, "transcripts", r.scenarioId);
    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, `${String(r.index).padStart(4, "0")}.json`), JSON.stringify(r, null, 2));
  }
}

function progress(label: string, done: number, total: number) {
  const pct = Math.floor((done / total) * 100);
  process.stdout.write(`\r  ${label}: ${done}/${total} (${pct}%)   `);
}

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
