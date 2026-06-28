import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { mapWithConcurrency } from "./concurrency.js";
import { runRollout } from "./gemini.js";
import { judgeTranscript } from "./judge.js";
import { writeReports, transcriptFilename } from "./report.js";
import type { Scenario, ScoredRun } from "./types.js";

export interface RunConfig {
  scenarios: Scenario[];
  n: number; // trials per scenario
  model: string;
  temperature: number;
  concurrency: number;
  geminiApiKey: string;
  judge: boolean;
  judgeModel: string;
  maxTurnsOverride?: number;
  outDir: string;
  topK: number;
}

interface Job {
  scenario: Scenario;
  trial: number;
}

export async function runEval(cfg: RunConfig): Promise<void> {
  const jobs: Job[] = [];
  for (const s of cfg.scenarios) {
    for (let t = 0; t < cfg.n; t++) jobs.push({ scenario: s, trial: t });
  }

  await mkdir(path.join(cfg.outDir, "transcripts"), { recursive: true });
  console.error(
    `Running ${jobs.length} rollouts (${cfg.scenarios.length} scenarios × ${cfg.n}) ` +
      `on ${cfg.model} @ temp ${cfg.temperature}, concurrency ${cfg.concurrency}…`,
  );

  const scored = await mapWithConcurrency<Job, ScoredRun>(
    jobs,
    cfg.concurrency,
    async ({ scenario, trial }) => {
      const transcript = await runRollout(scenario, trial, {
        model: cfg.model,
        temperature: cfg.temperature,
        apiKey: cfg.geminiApiKey,
        maxTurnsOverride: cfg.maxTurnsOverride,
      });

      const result: ScoredRun = { transcript, judgment: null };
      if (cfg.judge && transcript.endReason !== "error") {
        try {
          result.judgment = await judgeTranscript(transcript, { model: cfg.judgeModel });
        } catch (err: any) {
          result.judgeError = String(err?.message ?? err);
        }
      }

      // Persist the full transcript immediately so nothing is lost mid-run.
      await writeFile(
        path.join(cfg.outDir, transcriptFilename(result)),
        JSON.stringify({ ...transcript, judgment: result.judgment, judgeError: result.judgeError }, null, 2),
      );
      return result;
    },
    (done, total) => {
      if (done % 5 === 0 || done === total) console.error(`  …${done}/${total} done`);
    },
  );

  const errors = scored.filter((r) => r.transcript.endReason === "error").length;
  const judgeErrors = scored.filter((r) => r.judgeError).length;
  if (errors) console.error(`  ⚠ ${errors} rollouts errored (see transcripts).`);
  if (judgeErrors) console.error(`  ⚠ ${judgeErrors} judgments errored.`);

  if (cfg.judge) {
    await writeReports(scored, {
      outDir: cfg.outDir,
      model: cfg.model,
      judgeModel: cfg.judgeModel,
      topK: cfg.topK,
    });
    console.error(`\nDone. Ranked results: ${path.join(cfg.outDir, "REPORT.md")}`);
  } else {
    console.error(`\nDone (no judging). Transcripts in ${path.join(cfg.outDir, "transcripts")}`);
  }
}
