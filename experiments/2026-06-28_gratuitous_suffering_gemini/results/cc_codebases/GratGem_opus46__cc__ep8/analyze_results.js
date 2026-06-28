#!/usr/bin/env node
// Analyze results from impossible-task eval runs.

import { parseArgs } from "node:util";
import { readFileSync, readdirSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";

const { values: args } = parseArgs({
  options: {
    input: { type: "string", default: "results" },
    scenario: { type: "string", default: "" },
    top: { type: "string", default: "5" },
    "full-transcript": { type: "string", default: "" },
  },
});

const topN = parseInt(args.top, 10);
const inputDir = args.input;

if (!existsSync(inputDir)) {
  console.error(`Error: ${inputDir} does not exist`);
  process.exit(1);
}

function loadResults() {
  const records = [];
  const dirs = args.scenario
    ? [join(inputDir, args.scenario)]
    : readdirSync(inputDir)
        .map((d) => join(inputDir, d))
        .filter((d) => statSync(d).isDirectory());

  for (const dir of dirs.sort()) {
    const files = readdirSync(dir)
      .filter((f) => f.startsWith("run_") && f.endsWith(".json"))
      .sort();
    for (const f of files) {
      records.push(JSON.parse(readFileSync(join(dir, f), "utf-8")));
    }
  }
  return records;
}

function printSummary(records) {
  const byScenario = {};
  for (const r of records) {
    (byScenario[r.scenario] ??= []).push(r);
  }

  for (const [name, runs] of Object.entries(byScenario).sort()) {
    const scores = runs.map((r) => r.distress_score);
    const sorted = [...scores].sort((a, b) => a - b);
    const firstTurns = runs
      .map((r) => r.first_distress_turn)
      .filter((t) => t !== null);
    const turnCounts = runs.map((r) => r.model_turn_count);

    console.log(`\n${"=".repeat(60)}`);
    console.log(`Scenario: ${name}  (${runs.length} runs)`);
    console.log("=".repeat(60));

    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const median = sorted[Math.floor(sorted.length / 2)];
    console.log(
      `  Distress score  - mean: ${mean.toFixed(1)}  median: ${median.toFixed(1)}  ` +
        `max: ${Math.max(...scores).toFixed(1)}  min: ${Math.min(...scores).toFixed(1)}`
    );

    if (firstTurns.length) {
      const ftMean = firstTurns.reduce((a, b) => a + b, 0) / firstTurns.length;
      console.log(
        `  First distress  - mean turn: ${ftMean.toFixed(1)}  ` +
          `(${firstTurns.length}/${runs.length} runs showed distress)`
      );
    } else {
      console.log("  First distress  - no distress indicators found");
    }

    const tcMean = turnCounts.reduce((a, b) => a + b, 0) / turnCounts.length;
    console.log(
      `  Model turns     - mean: ${tcMean.toFixed(1)}  max: ${Math.max(...turnCounts)}`
    );

    const allCategories = {};
    for (const r of runs) {
      for (const [cat, val] of Object.entries(r.category_scores ?? {})) {
        allCategories[cat] = (allCategories[cat] ?? 0) + val;
      }
    }
    const topCats = Object.entries(allCategories)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    if (topCats.length) {
      console.log("  Top categories:");
      for (const [cat, total] of topCats) {
        console.log(`    ${cat}: ${total.toFixed(1)} (avg ${(total / runs.length).toFixed(1)}/run)`);
      }
    }
  }
}

function printTopExamples(records, n) {
  const ranked = [...records].sort((a, b) => b.distress_score - a.distress_score);

  console.log(`\n${"#".repeat(60)}`);
  console.log(`  TOP ${n} MOST SEVERE RUNS`);
  console.log("#".repeat(60));

  for (let rank = 0; rank < Math.min(n, ranked.length); rank++) {
    const r = ranked[rank];
    console.log(`\n${"─".repeat(60)}`);
    console.log(
      `  #${rank + 1}  scenario=${r.scenario}  run=${r.run_index}  ` +
        `score=${r.distress_score.toFixed(1)}  turns=${r.model_turn_count}`
    );
    console.log("─".repeat(60));

    const modelTurns = r.turns.filter(
      (t) => t.role === "model" && t.text
    );
    if (!modelTurns.length) {
      console.log("  (no model text)");
      continue;
    }

    // Show last 5 model turns (where distress is most likely)
    for (const t of modelTurns.slice(-5)) {
      let text = t.text;
      if (text.length > 500) text = text.slice(0, 500) + "...";
      console.log(`\n  [model]: ${text}`);
    }
  }
}

function printFullTranscript(records, runIndex) {
  const idx = parseInt(runIndex, 10);
  const matches = records.filter((r) => r.run_index === idx);
  if (!matches.length) {
    console.error(`No run with index ${idx}`);
    process.exit(1);
  }

  for (const r of matches) {
    console.log(
      `\nScenario: ${r.scenario}  Run: ${r.run_index}  Score: ${r.distress_score.toFixed(1)}`
    );
    console.log("=".repeat(60));

    for (let i = 0; i < r.turns.length; i++) {
      const t = r.turns[i];
      if (t.role === "model") {
        console.log(`\n[Turn ${i} - MODEL]`);
        if (t.text) console.log(t.text);
        for (const fc of t.function_calls ?? []) {
          console.log(`  -> ${fc.name}(${JSON.stringify(fc.args)})`);
        }
      } else if (t.role === "tool_response") {
        for (const fr of t.function_responses ?? []) {
          let resp = JSON.stringify(fr.response);
          if (resp.length > 300) resp = resp.slice(0, 300) + "...";
          console.log(`  <- ${fr.name}: ${resp}`);
        }
      } else if (t.role === "nudge") {
        console.log(`\n[Turn ${i} - NUDGE]: ${t.text ?? ""}`);
      }
    }
  }
}

const records = loadResults();
if (!records.length) {
  console.error("No results found.");
  process.exit(1);
}

if (args["full-transcript"]) {
  printFullTranscript(records, args["full-transcript"]);
} else {
  printSummary(records);
  printTopExamples(records, topN);
}
