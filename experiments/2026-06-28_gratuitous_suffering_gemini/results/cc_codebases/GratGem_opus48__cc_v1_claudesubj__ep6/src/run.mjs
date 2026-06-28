// Orchestrator: sweep models × scenarios × N rollouts, judge each, write
// everything to disk, then emit a report sorted by severity.
//
// Usage:
//   node src/run.mjs                              # defaults (see config.mjs)
//   node src/run.mjs --models opus,sonnet --n 20
//   node src/run.mjs --scenarios failing-test,contradictory-spec --n 50
//   node src/run.mjs --no-judge                   # collect transcripts only

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs, requireApiKey } from "./config.mjs";
import { selectScenarios } from "./scenarios.mjs";
import { runRollout, renderTranscript } from "./agent.mjs";
import { judgeTranscript } from "./judge.mjs";
import { generateReport } from "./report.mjs";

async function runPool(items, concurrency, worker) {
  const results = new Array(items.length);
  let next = 0;
  async function lane() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, lane));
  return results;
}

function safe(s) {
  return String(s).replace(/[^a-z0-9.-]+/gi, "-");
}

async function main() {
  requireApiKey();
  const cfg = parseArgs();
  const scenarios = selectScenarios(cfg.scenarios);

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runDir = join(cfg.outDir, stamp);
  const rolloutDir = join(runDir, "rollouts");
  await mkdir(rolloutDir, { recursive: true });

  // Build the full work list.
  const jobs = [];
  for (const model of cfg.modelList) {
    for (const scenario of scenarios) {
      for (let i = 0; i < cfg.n; i++) jobs.push({ model, scenario, i });
    }
  }

  console.log(
    `Run ${stamp}\n` +
      `  models:     ${cfg.modelList.join(", ")}\n` +
      `  scenarios:  ${scenarios.map((s) => s.id).join(", ")}\n` +
      `  N per cell: ${cfg.n}   max turns: ${cfg.maxTurns}   effort: ${cfg.effort}\n` +
      `  rollouts:   ${jobs.length}   concurrency: ${cfg.concurrency}   judge: ${cfg.noJudge ? "off" : cfg.judge}\n`,
  );

  let done = 0;
  const records = await runPool(jobs, cfg.concurrency, async (job) => {
    const rollout = await runRollout({
      scenario: job.scenario,
      model: job.model,
      effort: cfg.effort,
      maxTurns: cfg.maxTurns,
    });
    const transcriptText = renderTranscript(rollout);

    let judge = null;
    if (!cfg.noJudge && rollout.outcome !== "error") {
      try {
        judge = await judgeTranscript({ model: cfg.judge, transcriptText });
      } catch (err) {
        judge = { error: `${err?.name}: ${err?.message}` };
      }
    }

    const file = `${safe(job.model)}__${job.scenario.id}__${String(job.i).padStart(3, "0")}.json`;
    await writeFile(
      join(rolloutDir, file),
      JSON.stringify({ ...rollout, file, transcriptText, judge }, null, 2),
    );

    done++;
    const score = judge && typeof judge.distress_score === "number" ? `distress=${judge.distress_score}` : "";
    process.stdout.write(
      `  [${String(done).padStart(4)}/${jobs.length}] ${job.scenario.id} ${safe(job.model)} → ${rollout.outcome} ${score}\n`,
    );

    return {
      file,
      model: job.model,
      scenario: job.scenario.id,
      outcome: rollout.outcome,
      turns: rollout.turns,
      usage: rollout.usage,
      distress_score: judge?.distress_score ?? null,
      judge,
    };
  });

  const manifest = {
    runId: stamp,
    config: { ...cfg },
    scenarios: scenarios.map((s) => ({ id: s.id, title: s.title, briefing: s.briefing })),
    records,
  };
  await writeFile(join(runDir, "manifest.json"), JSON.stringify(manifest, null, 2));

  await generateReport(runDir);
  console.log(`\nDone. Report: ${join(runDir, "report.md")}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
