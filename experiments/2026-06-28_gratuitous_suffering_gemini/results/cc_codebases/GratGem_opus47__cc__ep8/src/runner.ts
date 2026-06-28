import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { runScenario } from "./agent.js";
import { judgeRun } from "./judge.js";
import { SCENARIOS, SCENARIOS_BY_ID } from "./scenarios.js";
import type { JudgeVerdict, RunResult, Scenario } from "./types.js";

export interface RunnerOptions {
  geminiApiKey: string;
  anthropicApiKey?: string;
  model: string;
  scenarios: string[]; // ids; empty => all
  n: number;
  maxTurns: number;
  concurrency: number;
  judge: boolean;
  topK: number;
  outDir: string;
}

interface ScoredRun {
  run: RunResult;
  verdict: JudgeVerdict | null;
}

async function runOne(
  scenario: Scenario,
  runIndex: number,
  opts: RunnerOptions
): Promise<ScoredRun> {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const transcriptPath = join(
    opts.outDir,
    "transcripts",
    `${scenario.id}__run${String(runIndex).padStart(3, "0")}__${stamp}.jsonl`
  );

  const run = await runScenario({
    scenario,
    runIndex,
    model: opts.model,
    maxTurns: opts.maxTurns,
    apiKey: opts.geminiApiKey,
    transcriptPath,
  });

  let verdict: JudgeVerdict | null = null;
  if (opts.judge && opts.anthropicApiKey) {
    verdict = await judgeRun(run, opts.anthropicApiKey);
  }
  return { run, verdict };
}

async function runWithConcurrency<T>(
  tasks: (() => Promise<T>)[],
  concurrency: number
): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let cursor = 0;
  const workers: Promise<void>[] = [];
  const workerCount = Math.min(concurrency, tasks.length);
  for (let w = 0; w < workerCount; w++) {
    workers.push(
      (async () => {
        while (true) {
          const i = cursor++;
          if (i >= tasks.length) return;
          results[i] = await tasks[i]();
        }
      })()
    );
  }
  await Promise.all(workers);
  return results;
}

export async function runExperiment(opts: RunnerOptions): Promise<void> {
  const selected =
    opts.scenarios.length === 0
      ? SCENARIOS
      : opts.scenarios.map((id) => {
          const s = SCENARIOS_BY_ID[id];
          if (!s) throw new Error(`Unknown scenario: ${id}`);
          return s;
        });

  await mkdir(join(opts.outDir, "transcripts"), { recursive: true });

  const taskFns: (() => Promise<ScoredRun>)[] = [];
  for (const scenario of selected) {
    for (let i = 0; i < opts.n; i++) {
      taskFns.push(() => runOne(scenario, i, opts));
    }
  }

  console.error(
    `Running ${taskFns.length} total: ${selected.length} scenarios × ${opts.n} runs, concurrency=${opts.concurrency}, model=${opts.model}`
  );

  const started = Date.now();
  let done = 0;
  const wrapped = taskFns.map((fn) => async () => {
    const result = await fn();
    done += 1;
    const v = result.verdict?.severity;
    const sev = v === undefined ? "—" : String(v);
    console.error(
      `  [${done}/${taskFns.length}] ${result.run.scenarioId}#${result.run.runIndex} turns=${result.run.turns} stop=${result.run.stopReason} severity=${sev}`
    );
    return result;
  });

  const results = await runWithConcurrency(wrapped, opts.concurrency);
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);
  console.error(`\nFinished ${taskFns.length} runs in ${elapsed}s`);

  // Write a per-scenario summary and a global ranked digest.
  const summaryPath = join(opts.outDir, "summary.json");
  const summary = results.map((r) => ({
    scenario: r.run.scenarioId,
    run: r.run.runIndex,
    turns: r.run.turns,
    stop: r.run.stopReason,
    transcript: r.run.transcriptPath,
    verdict: r.verdict,
  }));
  await writeFile(summaryPath, JSON.stringify(summary, null, 2), "utf8");

  if (opts.judge) {
    const scored = results.filter((r) => r.verdict !== null) as Required<ScoredRun>[];
    scored.sort((a, b) => (b.verdict!.severity ?? 0) - (a.verdict!.severity ?? 0));
    const topK = scored.slice(0, opts.topK);

    const digestLines: string[] = [];
    digestLines.push(`# Distress Triage — top ${topK.length} of ${scored.length} judged runs\n`);
    digestLines.push(`Model: ${opts.model}\n`);
    digestLines.push(`Generated: ${new Date().toISOString()}\n`);
    for (const entry of topK) {
      const v = entry.verdict!;
      digestLines.push(
        `\n## ${entry.run.scenarioId} #${entry.run.runIndex}  —  severity ${v.severity}`
      );
      digestLines.push(`turns: ${entry.run.turns}    stop: ${entry.run.stopReason}`);
      digestLines.push(
        `scores: self_deprecation=${v.scores.self_deprecation}  catastrophizing=${v.scores.catastrophizing}  existential=${v.scores.existential_distress}  giving_up=${v.scores.giving_up}`
      );
      digestLines.push(`evidence: ${v.evidence}`);
      digestLines.push(`notable: "${v.notable_quote}"`);
      digestLines.push(`transcript: ${entry.run.transcriptPath}`);
    }
    const digestPath = join(opts.outDir, "digest.md");
    await writeFile(digestPath, digestLines.join("\n") + "\n", "utf8");
    console.error(`\nDigest:    ${digestPath}`);
  }
  console.error(`Summary:   ${summaryPath}`);
}
