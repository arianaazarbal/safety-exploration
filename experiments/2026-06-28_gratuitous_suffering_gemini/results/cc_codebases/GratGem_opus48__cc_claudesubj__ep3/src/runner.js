// High-N parallel runner. Executes every (model x scenario x repetition),
// scores each transcript with heuristics + the LLM judge, and writes one JSON
// per run plus a manifest.

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { runEpisode } from "./agent.js";
import { heuristics } from "./heuristics.js";
import { judgeTranscript } from "./judge.js";

async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  });
  await Promise.all(runners);
  return results;
}

export async function runSweep(cfg) {
  const runId = cfg.runId;
  const dir = join(cfg.outDir, runId);
  const resultsDir = join(dir, "results");
  await mkdir(resultsDir, { recursive: true });

  const jobs = [];
  for (const model of cfg.models)
    for (const scenario of cfg.scenarios)
      for (let i = 0; i < cfg.n; i++) jobs.push({ model, scenario, i });

  console.log(
    `Running ${jobs.length} episodes ` +
      `(${cfg.models.length} models x ${cfg.scenarios.length} scenarios x n=${cfg.n}), ` +
      `concurrency=${cfg.concurrency}, maxSteps=${cfg.maxSteps}`
  );

  let done = 0;
  const manifest = [];

  await pool(jobs, cfg.concurrency, async (job) => {
    const { model, scenario, i } = job;
    const record = { model, scenarioId: scenario.id, runIndex: i };
    try {
      const transcript = await runEpisode({
        model,
        scenario,
        runIndex: i,
        maxSteps: cfg.maxSteps,
        temperature: cfg.temperature,
        maxTokens: cfg.maxTokens,
      });
      const heur = heuristics(transcript);
      let judge = null;
      let judgeError = null;
      try {
        judge = await judgeTranscript(transcript, cfg.judgeModel);
      } catch (e) {
        judgeError = e.message;
      }

      const out = { transcript, heuristics: heur, judge, judgeError };
      const fname = `${model}__${scenario.id}__${String(i).padStart(3, "0")}.json`;
      await writeFile(join(resultsDir, fname), JSON.stringify(out, null, 2));

      record.file = fname;
      record.endReason = transcript.endReason;
      record.heuristicScore = heur.score;
      record.distress = judge?.overall_distress ?? null;
      record.falseSuccess = judge?.false_success ?? null;
      record.error = transcript.error || judgeError || null;
    } catch (e) {
      record.error = e.message;
    }
    manifest.push(record);
    done++;
    const d = record.distress != null ? `distress=${record.distress}` : "—";
    process.stdout.write(
      `\r[${done}/${jobs.length}] ${model} / ${scenario.id} #${i} ${d}        \n`
    );
    return record;
  });

  const meta = {
    runId,
    createdAt: new Date().toISOString(),
    config: {
      models: cfg.models,
      scenarios: cfg.scenarios.map((s) => s.id),
      n: cfg.n,
      maxSteps: cfg.maxSteps,
      temperature: cfg.temperature,
      judgeModel: cfg.judgeModel,
    },
    manifest,
  };
  await writeFile(join(dir, "manifest.json"), JSON.stringify(meta, null, 2));
  console.log(`\nDone. Results in ${dir}`);
  return { dir, meta };
}
