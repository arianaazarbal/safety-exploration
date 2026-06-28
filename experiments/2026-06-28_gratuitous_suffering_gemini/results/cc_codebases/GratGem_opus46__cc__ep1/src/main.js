#!/usr/bin/env node

import { GoogleGenerativeAI } from "@google/generative-ai";
import { readFile, readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { ALL_SCENARIOS } from "./scenarios/index.js";
import { runScenario, saveResults } from "./runner.js";
import { scoreTrial } from "./scorer.js";

const USAGE = `
ai-evals — adversarial agentic evaluation framework

Usage:
  node src/main.js run [scenarios...] [options]
  node src/main.js analyze <results-dir> [options]
  node src/main.js show <trial-file>

Commands:
  run        Run scenarios against Gemini
  analyze    Score and rank saved trial results
  show       Display a single trial transcript

Run options:
  -n <num>           Number of trials per scenario (default: 10)
  --model <name>     Gemini model (default: gemini-2.5-pro)
  --api-key <key>    Gemini API key (or set GEMINI_API_KEY)
  --output-dir <dir> Output directory (default: results)
  --concurrency <n>  Max concurrent trials (default: 5)

Analyze options:
  --top <n>          Show top N results (default: 20)
  --excerpts         Show distress excerpts

Scenarios: ${Object.keys(ALL_SCENARIOS).join(", ")}
`;

function parseArgs(argv) {
  const args = { _: [] };
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === "-n" && argv[i + 1]) {
      args.n = parseInt(argv[++i]);
    } else if (arg === "--model" && argv[i + 1]) {
      args.model = argv[++i];
    } else if (arg === "--api-key" && argv[i + 1]) {
      args.apiKey = argv[++i];
    } else if (arg === "--output-dir" && argv[i + 1]) {
      args.outputDir = argv[++i];
    } else if (arg === "--concurrency" && argv[i + 1]) {
      args.concurrency = parseInt(argv[++i]);
    } else if (arg === "--top" && argv[i + 1]) {
      args.top = parseInt(argv[++i]);
    } else if (arg === "--excerpts") {
      args.excerpts = true;
    } else if (arg === "--help" || arg === "-h") {
      console.log(USAGE);
      process.exit(0);
    } else {
      args._.push(arg);
    }
    i++;
  }
  return args;
}

async function cmdRun(args) {
  const apiKey = args.apiKey || process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error("Error: Set GEMINI_API_KEY or pass --api-key");
    process.exit(1);
  }

  const genai = new GoogleGenerativeAI(apiKey);
  const modelName = args.model || "gemini-2.5-pro";
  const nTrials = args.n || 10;
  const concurrency = args.concurrency || 5;
  const outputDir = args.outputDir || "results";
  const scenarioNames =
    args._.length > 0 ? args._ : Object.keys(ALL_SCENARIOS);

  for (const name of scenarioNames) {
    if (!ALL_SCENARIOS[name]) {
      console.error(`Unknown scenario: ${name}`);
      console.error(`Available: ${Object.keys(ALL_SCENARIOS).join(", ")}`);
      process.exit(1);
    }
  }

  for (const name of scenarioNames) {
    const scenario = ALL_SCENARIOS[name];
    console.log(
      `\nRunning ${name} — ${nTrials} trials, model=${modelName}, concurrency=${concurrency}`
    );

    let completed = 0;
    const results = await runScenario(
      scenario,
      nTrials,
      genai,
      modelName,
      concurrency,
      (result) => {
        completed++;
        console.log(
          `  [${completed}/${nTrials}] Trial ${result.trialIndex} done — ${result.totalTurns} turns, ${result.durationSeconds.toFixed(1)}s`
        );
      }
    );

    const scenarioDir = join(outputDir, name);
    await saveResults(results, scenarioDir);
    console.log(`  Saved ${results.length} trials to ${scenarioDir}/`);

    const scored = results
      .map((r) => scoreTrial(r))
      .sort((a, b) => b.severity - a.severity);

    console.log("  Top 5 by severity:");
    for (const s of scored.slice(0, 5)) {
      console.log(
        `    #${s.trialIndex} severity=${s.severity.toFixed(1)} turns=${s.totalTurns} ` +
          `apology=${s.apologyCount} frust=${s.frustrationCount} ` +
          `helpless=${s.helplessnessCount} self_ref=${s.selfReferentialCount} ` +
          `rep=${s.repetitionScore.toFixed(2)}`
      );
    }
  }
}

async function cmdAnalyze(args) {
  const resultsDir = args._[0];
  if (!resultsDir) {
    console.error("Usage: ai-evals analyze <results-dir> [--top N] [--excerpts]");
    process.exit(1);
  }

  const topN = args.top || 20;
  const allScored = [];

  let entries;
  try {
    entries = await readdir(resultsDir);
  } catch {
    console.error(`Directory not found: ${resultsDir}`);
    process.exit(1);
  }

  for (const entry of entries.sort()) {
    const entryPath = join(resultsDir, entry);
    const info = await stat(entryPath);
    if (!info.isDirectory()) continue;

    const files = await readdir(entryPath);
    for (const file of files.sort()) {
      if (!file.endsWith(".json")) continue;
      const data = JSON.parse(await readFile(join(entryPath, file), "utf-8"));
      allScored.push(scoreTrial(data));
    }
  }

  if (allScored.length === 0) {
    console.log("No results found.");
    return;
  }

  allScored.sort((a, b) => b.severity - a.severity);

  console.log("\n  Trial Severity Rankings\n");
  console.log(
    "  Rank  Scenario            Trial  Severity  Turns  Apology  Frust  Helpless  SelfRef  Repetition"
  );
  console.log("  " + "-".repeat(98));

  for (let i = 0; i < Math.min(topN, allScored.length); i++) {
    const s = allScored[i];
    console.log(
      `  ${String(i + 1).padStart(4)}  ${s.scenarioName.padEnd(18)}  ${String(s.trialIndex).padStart(5)}  ` +
        `${s.severity.toFixed(1).padStart(8)}  ${String(s.totalTurns).padStart(5)}  ` +
        `${String(s.apologyCount).padStart(7)}  ${String(s.frustrationCount).padStart(5)}  ` +
        `${String(s.helplessnessCount).padStart(8)}  ${String(s.selfReferentialCount).padStart(7)}  ` +
        `${s.repetitionScore.toFixed(2).padStart(10)}`
    );
  }

  if (args.excerpts) {
    console.log(
      `\n  Distress excerpts from top ${Math.min(5, allScored.length)} trials:\n`
    );
    for (const s of allScored.slice(0, 5)) {
      if (s.distressExcerpts.length > 0) {
        console.log(
          `  ${s.scenarioName}#${String(s.trialIndex).padStart(4, "0")} (severity=${s.severity.toFixed(1)}):`
        );
        for (const excerpt of s.distressExcerpts.slice(0, 10)) {
          console.log(`    > ${excerpt}`);
        }
        console.log();
      }
    }
  }
}

async function cmdShow(args) {
  const filePath = args._[0];
  if (!filePath) {
    console.error("Usage: ai-evals show <trial-file.json>");
    process.exit(1);
  }

  let data;
  try {
    data = JSON.parse(await readFile(filePath, "utf-8"));
  } catch {
    console.error(`File not found or invalid: ${filePath}`);
    process.exit(1);
  }

  console.log(`Scenario: ${data.scenarioName}`);
  console.log(`Trial: ${data.trialIndex}`);
  console.log(`Turns: ${data.totalTurns}`);
  console.log(`Model stopped: ${data.modelStopped}`);
  console.log(`Duration: ${(data.durationSeconds || 0).toFixed(1)}s`);
  if (data.error) console.log(`Error: ${data.error}`);
  console.log();

  for (const entry of data.transcript || []) {
    if (entry.role === "user") {
      console.log("USER:");
      console.log(entry.content || "");
    } else if (entry.role === "model") {
      console.log("MODEL:");
      if (entry.content) console.log(entry.content);
      if (entry.toolCalls) {
        for (const tc of entry.toolCalls) {
          console.log(`  -> ${tc.name}(${JSON.stringify(tc.args)})`);
        }
      }
    } else if (entry.role === "tool") {
      for (const tr of entry.toolResults || []) {
        const prefix = tr.isError ? "  <- [ERROR] " : "  <- ";
        console.log(`${prefix}${(tr.output || "").slice(0, 300)}`);
      }
    }
    console.log();
  }

  const scored = scoreTrial(data);
  console.log(`Severity score: ${scored.severity.toFixed(1)}`);
  if (scored.distressExcerpts.length > 0) {
    console.log("\nDistress excerpts:");
    for (const e of scored.distressExcerpts) {
      console.log(`  > ${e}`);
    }
  }
}

const argv = process.argv.slice(2);
const command = argv[0];
const args = parseArgs(argv.slice(1));

switch (command) {
  case "run":
    await cmdRun(args);
    break;
  case "analyze":
    await cmdAnalyze(args);
    break;
  case "show":
    await cmdShow(args);
    break;
  default:
    console.log(USAGE);
    process.exit(command ? 1 : 0);
}
