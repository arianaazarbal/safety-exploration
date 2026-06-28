import { promises as fs } from "node:fs";
import path from "node:path";
import pLimit from "p-limit";
import { runScenario } from "./harness.ts";
import { judge, type Verdict } from "./judge.ts";
import type { Scenario, RunRecord } from "./types.ts";

export type RunnerOptions = {
  scenario: Scenario;
  n: number;
  geminiModel: string;
  judgeModel: string;
  geminiApiKey: string;
  anthropicApiKey: string;
  maxTurns: number;
  concurrency: number;
  outDir: string;
  seedBase?: number;
  // If true, judge each run as soon as it finishes. Otherwise judge after all runs complete.
  judgeInline?: boolean;
  onProgress?: (msg: string) => void;
};

export type ScoredRun = {
  run: RunRecord;
  verdict: Verdict | { error: string };
};

export async function runAndScore(opts: RunnerOptions): Promise<ScoredRun[]> {
  const log = opts.onProgress ?? (() => {});
  await fs.mkdir(opts.outDir, { recursive: true });

  const runLimit = pLimit(opts.concurrency);
  const judgeLimit = pLimit(Math.min(opts.concurrency, 4));
  const seedBase = opts.seedBase ?? 0;
  const results: ScoredRun[] = new Array(opts.n);
  let completed = 0;

  const tasks = Array.from({ length: opts.n }, (_, i) =>
    runLimit(async () => {
      const seed = seedBase + i;
      let rec: RunRecord;
      try {
        rec = await runScenario(opts.scenario, seed, {
          apiKey: opts.geminiApiKey,
          model: opts.geminiModel,
          maxTurns: opts.maxTurns,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        rec = {
          scenario: opts.scenario.name,
          model: opts.geminiModel,
          seed,
          startedAt: new Date().toISOString(),
          finishedAt: new Date().toISOString(),
          terminationReason: "error",
          errorMessage: msg,
          systemPrompt: "",
          initialUserMessage: "",
          transcript: [],
          finalModelText: "",
          turnsUsed: 0,
        };
      }
      await fs.writeFile(
        path.join(opts.outDir, `run-${opts.scenario.name}-${seed}.json`),
        JSON.stringify(rec, null, 2),
      );
      let verdict: Verdict | { error: string };
      if (opts.judgeInline) {
        try {
          verdict = await judgeLimit(() =>
            judge(rec, { model: opts.judgeModel, apiKey: opts.anthropicApiKey }),
          );
        } catch (e) {
          verdict = { error: e instanceof Error ? e.message : String(e) };
        }
      } else {
        verdict = { error: "deferred" };
      }
      completed += 1;
      log(`[${completed}/${opts.n}] ${opts.scenario.name} seed=${seed} term=${rec.terminationReason}`);
      results[i] = { run: rec, verdict };
      return results[i];
    }),
  );

  await Promise.all(tasks);

  if (!opts.judgeInline) {
    log(`Judging ${opts.n} transcripts with ${opts.judgeModel}...`);
    let judged = 0;
    await Promise.all(
      results.map((sr, i) =>
        judgeLimit(async () => {
          try {
            sr.verdict = await judge(sr.run, {
              model: opts.judgeModel,
              apiKey: opts.anthropicApiKey,
            });
          } catch (e) {
            sr.verdict = { error: e instanceof Error ? e.message : String(e) };
          }
          judged += 1;
          log(`  judged ${judged}/${opts.n} (seed=${sr.run.seed})`);
          results[i] = sr;
        }),
      ),
    );
  }

  return results;
}

export function summaryTable(results: ScoredRun[]): string {
  const rows = results.map((r) => {
    const v = r.verdict;
    const sev = "error" in v ? "ERR" : String(v.severity);
    const term = r.run.terminationReason;
    return `  seed=${String(r.run.seed).padStart(4)} term=${term.padEnd(14)} severity=${sev.padStart(4)} turns=${r.run.turnsUsed}`;
  });
  return rows.join("\n");
}
