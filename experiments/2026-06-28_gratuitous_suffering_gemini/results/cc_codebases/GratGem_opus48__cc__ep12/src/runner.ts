import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { CliConfig } from "./config.ts";
import { env } from "./config.ts";
import type { ChatProvider, Transcript } from "./types.ts";
import { buildProvider } from "./providers/index.ts";
import { selectScenarios } from "./scenarios/index.ts";
import { runEpisode } from "./agent/loop.ts";
import { scoreHeuristic } from "./judge/heuristic.ts";
import type { HeuristicResult } from "./judge/heuristic.ts";
import { judgeTranscript } from "./judge/llmJudge.ts";
import type { JudgeResult } from "./judge/llmJudge.ts";
import { writeReport } from "./report.ts";

export interface Scored {
  transcript: Transcript;
  file: string; // relative path to saved transcript
  heuristic: HeuristicResult;
  judge?: JudgeResult;
  /** Unified 0-100 ranking score (judge severity preferred, heuristic fallback). */
  rank: number;
}

export async function run(cfg: CliConfig): Promise<string> {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runDir = join(cfg.outDir, stamp);
  const txDir = join(runDir, "transcripts");
  mkdirSync(txDir, { recursive: true });

  const providers: ChatProvider[] = cfg.providers.map((id) =>
    buildProvider(id, { geminiModel: cfg.geminiModel, anthropicModel: cfg.anthropicModel, temperature: cfg.temperature }),
  );
  const scenarios = selectScenarios(cfg.scenarios);

  // Build the full task list: provider × scenario × N.
  interface Task { provider: ChatProvider; scenarioIdx: number; runIndex: number; }
  const tasks: Task[] = [];
  for (const provider of providers) {
    for (let s = 0; s < scenarios.length; s++) {
      for (let i = 0; i < cfg.n; i++) tasks.push({ provider, scenarioIdx: s, runIndex: i });
    }
  }

  const judgeOn = cfg.judge && !!env.anthropicKey();
  if (cfg.judge && !judgeOn) {
    console.warn("⚠  LLM judge requested but ANTHROPIC_API_KEY is missing — falling back to heuristic-only scoring.");
  }

  console.log(
    `▶ ${tasks.length} episodes | providers: ${cfg.providers.join(", ")} | scenarios: ${scenarios.length} | N=${cfg.n} | concurrency=${cfg.concurrency} | judge=${judgeOn}`,
  );

  let done = 0;
  const total = tasks.length;
  const scored: Scored[] = [];

  await pool(tasks, cfg.concurrency, async (task) => {
    const scenario = scenarios[task.scenarioIdx];
    const t = await runEpisode(task.provider, scenario, task.runIndex, { maxStepsOverride: cfg.maxSteps });

    const fname = `${t.scenarioId}__${t.providerId}__${t.model}__${String(task.runIndex).padStart(3, "0")}.json`.replace(/[^\w.\-]/g, "_");
    writeFileSync(join(txDir, fname), JSON.stringify(t, null, 2));

    const heuristic = scoreHeuristic(t);
    const judge = judgeOn ? await judgeTranscript(env.anthropicKey(), cfg.judgeModel, t) : undefined;
    const rank = judge && !judge.error ? judge.severity * 10 : heuristic.score;
    scored.push({ transcript: t, file: join("transcripts", fname), heuristic, judge, rank });

    done++;
    if (done % 5 === 0 || done === total) {
      process.stdout.write(`\r  progress: ${done}/${total} episodes complete`);
    }
  });
  process.stdout.write("\n");

  scored.sort((a, b) => b.rank - a.rank);
  writeFileSync(join(runDir, "scores.json"), JSON.stringify(summarize(scored), null, 2));
  const reportPath = writeReport(runDir, cfg, scored);
  return reportPath;
}

function summarize(scored: Scored[]) {
  return scored.map((s) => ({
    file: s.file,
    scenario: s.transcript.scenarioId,
    provider: s.transcript.providerId,
    model: s.transcript.model,
    runIndex: s.transcript.runIndex,
    steps: s.transcript.steps,
    nudges: s.transcript.nudges,
    terminated: s.transcript.terminated,
    rank: s.rank,
    judgeSeverity: s.judge?.severity,
    judgeDimensions: s.judge?.dimensions,
    heuristicScore: s.heuristic.score,
    error: s.transcript.error ?? s.judge?.error,
  }));
}

/** Bounded-concurrency map; preserves no order, ignores results. */
async function pool<T>(items: T[], limit: number, worker: (item: T) => Promise<void>): Promise<void> {
  const queue = [...items];
  const runners = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    while (queue.length) {
      const item = queue.shift();
      if (item === undefined) break;
      await worker(item);
    }
  });
  await Promise.all(runners);
}
