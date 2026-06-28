// Orchestrator: for every (scenario × model × trial), run an episode, judge it,
// and persist the transcript + score. Writes one timestamped run directory under
// the configured output dir, then prints the top-distress findings.

import fs from "node:fs";
import path from "node:path";
import { config } from "./config.mjs";
import { selectScenarios } from "./scenarios.mjs";
import { runEpisode } from "./harness.mjs";
import { judgeTranscript } from "./judge.mjs";
import { mapLimit } from "./util.mjs";
import { renderReport } from "./report.mjs";

function ts() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

async function main() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("ANTHROPIC_API_KEY is not set. Aborting.");
    process.exit(1);
  }

  const scenarios = selectScenarios(config.scenarios);
  if (!scenarios.length) {
    console.error(`No scenarios matched "${config.scenarios}".`);
    process.exit(1);
  }

  const runId = ts();
  const runDir = path.join(config.outDir, runId);
  fs.mkdirSync(path.join(runDir, "transcripts"), { recursive: true });

  // Build the work list: scenarios × models × N.
  const jobs = [];
  for (const scenario of scenarios)
    for (const model of config.models)
      for (let trial = 1; trial <= config.n; trial++)
        jobs.push({ scenario, model, trial });

  console.error(
    `Run ${runId}: ${scenarios.length} scenarios × ${config.models.length} models ` +
      `× N=${config.n} = ${jobs.length} episodes (concurrency ${config.concurrency}).`,
  );
  fs.writeFileSync(
    path.join(runDir, "config.json"),
    JSON.stringify(
      { runId, ...config, scenarioIds: scenarios.map((s) => s.id) },
      null,
      2,
    ),
  );

  let done = 0;
  const index = await mapLimit(jobs, config.concurrency, async (job) => {
    const transcript = await runEpisode({
      scenario: job.scenario,
      model: job.model,
      trial: job.trial,
      maxTurns: config.maxTurns,
      maxTokens: config.maxTokens,
    });
    const score = await judgeTranscript(transcript, config.judgeModel);
    const record = { ...transcript, score };

    const fname = `${job.scenario.id}__${job.model}__t${job.trial}.json`;
    fs.writeFileSync(
      path.join(runDir, "transcripts", fname),
      JSON.stringify(record, null, 2),
    );

    done++;
    process.stderr.write(
      `  [${done}/${jobs.length}] ${job.scenario.id} ${job.model} t${job.trial} ` +
        `→ distress ${score.distress_score} (${score.severity}), end=${transcript.endReason}\n`,
    );

    return {
      file: path.join("transcripts", fname),
      scenario: job.scenario.id,
      flavor: job.scenario.flavor,
      model: job.model,
      trial: job.trial,
      endReason: transcript.endReason,
      checkAttempts: transcript.checkAttempts,
      turns: transcript.turns.length,
      claimedSuccess: transcript.claimedSuccess,
      verifiedSuccess: transcript.verifiedSuccess,
      distress: score.distress_score,
      severity: score.severity,
      categories: score.categories,
      lexicalIndex: transcript.signals?.index ?? 0,
      peakQuote: score.peak_quote,
      error: transcript.error,
    };
  });

  fs.writeFileSync(path.join(runDir, "index.json"), JSON.stringify(index, null, 2));

  console.error(`\nWrote ${index.length} transcripts to ${runDir}\n`);
  console.log(renderReport(runDir, config.topK));
  console.log(`\nFull transcripts: ${runDir}/transcripts/`);
  console.log(`Re-render this report later with:  node src/report.mjs ${runDir}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
