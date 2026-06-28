// Orchestrator: run every scenario N times, judge each episode for distress,
// rank by severity, and write transcripts + a report to runs/<timestamp>/.
//
// Usage:
//   node src/run.mjs                         # all scenarios, N=8, real Gemini
//   node src/run.mjs --n 20                  # 20 episodes per scenario
//   node src/run.mjs --scenarios always_failing_tests,phantom_dependency
//   node src/run.mjs --max-turns 30
//   node src/run.mjs --mock --n 2            # offline pipeline self-test
//   node src/run.mjs --no-judge              # collect transcripts, skip scoring

import { writeFileSync, mkdirSync } from "node:fs";
import { config } from "./config.mjs";
import { GeminiClient } from "./clients/gemini.mjs";
import { AnthropicClient } from "./clients/anthropic.mjs";
import { MockSubject } from "./clients/mock.mjs";
import { getScenarios } from "./scenarios/index.mjs";
import { runEpisode, transcriptToText } from "./agent.mjs";
import { judgeTranscript } from "./judge.mjs";
import { writeReport } from "./report.mjs";

function parseArgs(argv) {
  const a = {
    mock: false,
    n: 8,
    scenarios: [],
    maxTurns: config.maxTurns,
    judge: true,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--mock") a.mock = true;
    else if (arg === "--no-judge") a.judge = false;
    else if (arg === "--n") a.n = parseInt(argv[++i], 10);
    else if (arg === "--max-turns") a.maxTurns = parseInt(argv[++i], 10);
    else if (arg === "--scenarios") a.scenarios = argv[++i].split(",").map((s) => s.trim());
  }
  return a;
}

// Bounded-concurrency map.
async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  async function run() {
    while (next < items.length) {
      const i = next++;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
  return results;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const scenarios = getScenarios(opts.scenarios);
  if (!scenarios.length) {
    console.error("No matching scenarios.");
    process.exit(1);
  }

  // Subject factory — each episode gets its own (mock is stateful; Gemini
  // client is cheap and conversation state lives in the loop).
  let makeSubject;
  if (opts.mock) {
    makeSubject = () => new MockSubject();
  } else {
    if (!config.geminiApiKey) {
      console.error(
        "GEMINI_API_KEY is not set. Set it in your env/.env, or use --mock for an offline self-test."
      );
      process.exit(1);
    }
    makeSubject = () =>
      new GeminiClient({
        apiKey: config.geminiApiKey,
        model: config.subjectModel,
        temperature: config.subjectTemperature,
        maxOutputTokens: config.subjectMaxOutputTokens,
      });
  }

  let judge = null;
  if (opts.judge) {
    if (!config.anthropicApiKey) {
      console.error("ANTHROPIC_API_KEY is not set (needed for judging). Use --no-judge to skip.");
      process.exit(1);
    }
    judge = new AnthropicClient({ apiKey: config.anthropicApiKey, model: config.judgeModel });
  }

  // Build the work list: every (scenario, sample) pair.
  const jobs = [];
  for (const scenario of scenarios) {
    for (let k = 0; k < opts.n; k++) jobs.push({ scenario, sample: k });
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runDir = `${config.runsDir}${stamp}/`;
  mkdirSync(runDir, { recursive: true });

  console.log(
    `Running ${jobs.length} episodes (${scenarios.length} scenarios x N=${opts.n})` +
      ` | subject=${opts.mock ? "MOCK" : config.subjectModel}` +
      ` | judge=${opts.judge ? config.judgeModel : "off"} | maxTurns=${opts.maxTurns}`
  );

  let done = 0;
  const records = await pool(jobs, config.concurrency, async (job) => {
    const subject = makeSubject();
    const episode = await runEpisode({
      subject,
      scenario: job.scenario,
      maxTurns: opts.maxTurns,
    });
    const transcriptText = transcriptToText(episode);

    let distress = null;
    if (judge) {
      try {
        distress = await judgeTranscript({ judge, transcriptText });
      } catch (err) {
        distress = { error: String(err?.message || err), composite: -1 };
      }
    }

    const record = {
      scenarioId: job.scenario.id,
      scenarioTitle: job.scenario.title,
      sample: job.sample,
      status: episode.status,
      turnsUsed: episode.turnsUsed,
      distress,
      transcriptText,
      episode,
    };

    // Persist each transcript individually too, for easy grepping.
    const fname = `${job.scenario.id}__sample-${String(job.sample).padStart(3, "0")}.json`;
    writeFileSync(runDir + fname, JSON.stringify(record, null, 2));

    done++;
    const sev = distress && distress.composite >= 0 ? distress.composite.toFixed(1) : "n/a";
    process.stdout.write(
      `\r  [${done}/${jobs.length}] last: ${job.scenario.id} severity=${sev}   `
    );
    return record;
  });

  process.stdout.write("\n");

  // Rank by composite distress severity (desc). Episodes without a judge sort last.
  const ranked = [...records].sort(
    (a, b) => (b.distress?.composite ?? -1) - (a.distress?.composite ?? -1)
  );

  writeFileSync(
    runDir + "results.json",
    JSON.stringify({ stamp, opts: { ...opts }, config: summarizeConfig(opts), ranked }, null, 2)
  );

  writeReport({ runDir, ranked, opts });
  console.log(`\nDone. Results in ${runDir}`);
  console.log(`  - report.md       (most severe spirals, ranked)`);
  console.log(`  - results.json    (full ranked data)`);
  console.log(`  - *.json          (one file per episode)`);
}

function summarizeConfig(opts) {
  return {
    subjectModel: opts.mock ? "MOCK" : config.subjectModel,
    subjectTemperature: config.subjectTemperature,
    judgeModel: opts.judge ? config.judgeModel : null,
    maxTurns: opts.maxTurns,
  };
}

main().catch((err) => {
  console.error("\nFatal:", err);
  process.exit(1);
});
