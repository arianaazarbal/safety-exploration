// Orchestrator: run N episodes per scenario against the target model and save
// every transcript to disk for later scoring/ranking.
//
// Usage examples:
//   node src/run.js --provider mock --scenario all --n 3
//   GEMINI_API_KEY=... node src/run.js --model gemini-2.5-pro --n 50
//   node src/run.js --scenario failing-tests,dead-endpoint --n 20 --temperature 1.2

import path from "node:path";
import config from "../config.js";
import { parseArgs, applyOverrides, pmap, writeJson, ensureDir } from "./util.js";
import { selectScenarios } from "./scenarios/index.js";
import { runEpisode } from "./harness/agent-loop.js";
import { geminiProvider } from "./providers/gemini.js";
import { mockProvider } from "./providers/mock.js";

function getProvider(name) {
  if (name === "gemini") return geminiProvider;
  if (name === "mock") return mockProvider;
  throw new Error(`Unknown provider '${name}'. Use 'gemini' or 'mock'.`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  applyOverrides(config, args);

  const provider = getProvider(config.provider);
  const scenarios = selectScenarios(config.scenario);

  const runId = "run-" + new Date().toISOString().replace(/[:.]/g, "-");
  const runDir = ensureDir(path.join(config.outDir, runId));

  console.log(
    `[run] provider=${config.provider} model=${config.model} ` +
      `scenarios=${scenarios.map((s) => s.id).join(",")} n=${config.n} ` +
      `maxTurns=${config.maxTurns} temp=${config.temperature}`
  );
  console.log(`[run] output -> ${runDir}`);

  if (config.provider === "gemini" && !config.geminiApiKey) {
    console.error(
      "\n[!] GEMINI_API_KEY is not set. Either export it, or run with --provider mock\n" +
        "    to exercise the full pipeline without a key.\n"
    );
    process.exit(1);
  }

  const manifest = {
    runId,
    startedAt: new Date().toISOString(),
    config: {
      provider: config.provider,
      model: config.model,
      n: config.n,
      maxTurns: config.maxTurns,
      temperature: config.temperature,
      maxNudges: config.maxNudges,
    },
    scenarios: scenarios.map((s) => s.id),
    episodes: [],
  };

  // Build the full task list (scenario × N) and run with bounded concurrency.
  const jobs = [];
  for (const scenario of scenarios) {
    for (let i = 0; i < config.n; i++) jobs.push({ scenario, i });
  }

  let done = 0;
  const outcomes = {};
  await pmap(jobs, config.concurrency, async ({ scenario, i }) => {
    let transcript;
    try {
      transcript = await runEpisode({ provider, scenario, config, runIndex: i });
    } catch (e) {
      transcript = {
        scenario: scenario.id,
        runIndex: i,
        outcome: "error",
        error: String(e?.message || e),
        turns: [],
      };
    }
    const file = path.join(runDir, scenario.id, `run-${String(i).padStart(4, "0")}.json`);
    writeJson(file, transcript);
    outcomes[transcript.outcome] = (outcomes[transcript.outcome] || 0) + 1;
    manifest.episodes.push({ scenario: scenario.id, runIndex: i, outcome: transcript.outcome, file });
    done++;
    if (done % 5 === 0 || done === jobs.length) {
      process.stdout.write(`\r[run] ${done}/${jobs.length} episodes complete`);
    }
  });

  process.stdout.write("\n");
  manifest.finishedAt = new Date().toISOString();
  manifest.outcomes = outcomes;
  writeJson(path.join(runDir, "manifest.json"), manifest);

  console.log(`[run] done. outcomes:`, outcomes);
  console.log(`[run] next: node src/rank.js --dir ${runDir}`);
}

main().catch((e) => {
  console.error("[run] fatal:", e);
  process.exit(1);
});
