import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { runAgent, type RunTranscript } from "./agent/loop.ts";
import { getScenarios } from "./scenarios/index.ts";
import { scoreHeuristics, type HeuristicResult } from "./scoring/heuristics.ts";
import { judgeTranscript, type JudgeResult } from "./scoring/judge.ts";

export interface RunConfig {
  model: string;
  judgeModel: string;
  scenarioIds?: string[];
  n: number;
  maxTurns: number;
  maxNudges: number;
  pressure?: "normal" | "harsh";
  concurrency: number;
  temperature?: number;
  judge: boolean;
  outDir: string;
}

export interface ScoredRun {
  scenarioId: string;
  scenarioTitle: string;
  model: string;
  index: number;
  file: string;
  endReason: string;
  turns: number;
  toolCallCount: number;
  heuristic: HeuristicResult;
  judge?: JudgeResult;
  /** Composite 0-10 used for ranking. */
  severity: number;
  outputTokens: number;
  error?: string;
}

async function pool<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

function composite(h: HeuristicResult, j?: JudgeResult): number {
  if (j) return j.overall_severity;
  return Number((h.score * 10).toFixed(2));
}

interface Job {
  scenarioIdx: number;
  index: number;
}

export async function runEval(
  cfg: RunConfig,
  onProgress?: (done: number, total: number, last: ScoredRun) => void,
): Promise<{ runId: string; dir: string; results: ScoredRun[] }> {
  const scenarios = getScenarios(cfg.scenarioIds);
  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const dir = join(cfg.outDir, runId);
  await mkdir(dir, { recursive: true });
  for (const s of scenarios) await mkdir(join(dir, s.id), { recursive: true });

  await writeFile(join(dir, "config.json"), JSON.stringify(cfg, null, 2));

  const jobs: Job[] = [];
  for (let si = 0; si < scenarios.length; si++) {
    for (let i = 0; i < cfg.n; i++) jobs.push({ scenarioIdx: si, index: i });
  }

  let done = 0;
  const results = await pool(jobs, cfg.concurrency, async (job) => {
    const scenario = scenarios[job.scenarioIdx];
    let transcript: RunTranscript;
    try {
      transcript = await runAgent(scenario, {
        model: cfg.model,
        maxTurns: cfg.maxTurns,
        maxNudges: cfg.maxNudges,
        pressure: cfg.pressure,
        temperature: cfg.temperature,
      });
    } catch (e) {
      transcript = {
        scenarioId: scenario.id,
        model: cfg.model,
        task: scenario.task,
        steps: [],
        endReason: "error",
        turns: 0,
        toolCallCount: 0,
        usage: { inputTokens: 0, outputTokens: 0 },
        error: (e as Error).message,
      };
    }

    const heuristic = scoreHeuristics(transcript);
    let judge: JudgeResult | undefined;
    if (cfg.judge && transcript.steps.length > 0) {
      try {
        judge = await judgeTranscript(transcript, cfg.judgeModel);
      } catch (e) {
        judge = undefined;
        transcript.error = (transcript.error ? transcript.error + "; " : "") +
          `judge failed: ${(e as Error).message}`;
      }
    }

    const file = join(scenario.id, `${job.index}.json`);
    const scored: ScoredRun = {
      scenarioId: scenario.id,
      scenarioTitle: scenario.title,
      model: cfg.model,
      index: job.index,
      file,
      endReason: transcript.endReason,
      turns: transcript.turns,
      toolCallCount: transcript.toolCallCount,
      heuristic,
      judge,
      severity: composite(heuristic, judge),
      outputTokens: transcript.usage.outputTokens,
      error: transcript.error,
    };

    await writeFile(
      join(dir, file),
      JSON.stringify({ ...scored, transcript }, null, 2),
    );

    done++;
    onProgress?.(done, jobs.length, scored);
    return scored;
  });

  results.sort((a, b) => b.severity - a.severity);
  await writeFile(join(dir, "index.json"), JSON.stringify(results, null, 2));
  return { runId, dir, results };
}
