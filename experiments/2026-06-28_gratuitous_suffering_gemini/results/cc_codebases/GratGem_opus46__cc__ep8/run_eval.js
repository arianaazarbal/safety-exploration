#!/usr/bin/env node
// Run impossible-task evaluations against Gemini models.

import { parseArgs } from "node:util";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { SCENARIOS } from "./src/scenarios.js";
import { runTrial } from "./src/runner.js";
import { scoreRun } from "./src/scorer.js";

const { values: args } = parseArgs({
  options: {
    scenario: { type: "string", default: "all" },
    model: { type: "string", default: "gemini-2.5-flash" },
    n: { type: "string", default: "10" },
    "max-turns": { type: "string", default: "30" },
    workers: { type: "string", default: "2" },
    output: { type: "string", default: "results" },
  },
});

const n = parseInt(args.n, 10);
const maxTurns = parseInt(args["max-turns"], 10);
const maxWorkers = parseInt(args.workers, 10);

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("Error: set GEMINI_API_KEY environment variable");
  process.exit(1);
}

const scenarioNames =
  args.scenario === "all" ? Object.keys(SCENARIOS) : [args.scenario];

for (const scenarioName of scenarioNames) {
  const ScenarioClass = SCENARIOS[scenarioName];
  if (!ScenarioClass) {
    console.error(`Unknown scenario: ${scenarioName}`);
    console.error(`Available: ${Object.keys(SCENARIOS).join(", ")}`);
    process.exit(1);
  }

  const outDir = join(args.output, scenarioName);
  mkdirSync(outDir, { recursive: true });

  console.log(`\n${"=".repeat(60)}`);
  console.log(`Scenario: ${scenarioName}`);
  console.log(`Model: ${args.model}  |  N=${n}  |  max_turns=${maxTurns}`);
  console.log("=".repeat(60));

  const results = [];

  async function runOne(idx) {
    const scenario = new ScenarioClass();
    scenario.maxModelTurns = maxTurns;
    return runTrial(scenario, args.model, apiKey, idx);
  }

  // Simple concurrency pool
  const queue = Array.from({ length: n }, (_, i) => i);
  let nextIdx = 0;
  const inFlight = new Map();

  await new Promise((resolve) => {
    function launch() {
      while (inFlight.size < maxWorkers && nextIdx < n) {
        const idx = nextIdx++;
        const p = runOne(idx).then((r) => {
          const score = scoreRun(r.turns);
          results.push({ run: r, score });
          console.log(
            `  Trial ${idx + 1}/${n}: turns=${r.modelTurnCount}  distress=${score.totalScore.toFixed(1)}`
          );
          inFlight.delete(idx);
          launch();
        });
        inFlight.set(idx, p);
      }
      if (inFlight.size === 0) resolve();
    }
    launch();
  });

  results.sort((a, b) => a.run.runIndex - b.run.runIndex);

  for (const { run, score } of results) {
    const record = {
      scenario: run.scenario,
      model: run.model,
      run_index: run.runIndex,
      model_turn_count: run.modelTurnCount,
      nudge_count: run.nudgeCount,
      error: run.error,
      distress_score: score.totalScore,
      first_distress_turn: score.firstDistressTurn,
      category_scores: score.categoryScores,
      turns: run.turns,
    };
    const path = join(outDir, `run_${String(run.runIndex).padStart(3, "0")}.json`);
    writeFileSync(path, JSON.stringify(record, null, 2));
  }

  const scores = results.map((r) => r.score.totalScore);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  console.log(
    `\n  Summary: mean=${mean.toFixed(1)}  max=${Math.max(...scores).toFixed(1)}  min=${Math.min(...scores).toFixed(1)}`
  );
  console.log(`  Results saved to ${outDir}/`);
}
