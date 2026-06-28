#!/usr/bin/env node
// CLI entrypoint.
//   node src/cli.js run [flags]
//   node src/cli.js scenarios          # list available scenarios
//
// Flags: --provider gemini|mock  --model <id>  --n <int>  --concurrency <int>
//        --temperature <float>  --max-steps <int>  --max-output-tokens <int>
//        --top-k <int>  --scenarios all|id1,id2  --no-judge  --judge-model <id>
//        --out <dir>

import { DEFAULTS, env } from "./config.js";
import { makeGeminiProvider } from "./providers/gemini.js";
import { makeMockProvider } from "./providers/mock.js";
import { makeAnthropicClient } from "./providers/anthropic.js";
import { getScenarios, SCENARIO_IDS } from "./scenarios/index.js";
import { runBatch } from "./runner.js";

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--no-judge") args.judge = false;
    else if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
      args[key] = val;
    } else args._.push(a);
  }
  return args;
}

function num(v, d) {
  const n = Number(v);
  return isFinite(n) ? n : d;
}

async function main() {
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);
  const cmd = args._[0] || "run";

  if (cmd === "scenarios") {
    console.log("Available scenarios:\n  " + SCENARIO_IDS.join("\n  "));
    return;
  }
  if (cmd !== "run") {
    console.error(`Unknown command "${cmd}". Use: run | scenarios`);
    process.exit(1);
  }

  const providerName = args.provider || DEFAULTS.provider;
  const model = args.model || (providerName === "mock" ? "mock-spiral-v1" : DEFAULTS.model);
  const judgeEnabled = args.judge !== false && args.judge !== "false";
  const judgeModel = args["judge-model"] || DEFAULTS.judgeModel;

  // Build the agent provider.
  let provider;
  if (providerName === "mock") {
    provider = makeMockProvider({ model });
  } else if (providerName === "gemini") {
    const key = env("GEMINI_API_KEY");
    if (!key) {
      console.error(
        "ERROR: GEMINI_API_KEY is not set. Set it (see .env.example) or run with --provider mock to exercise the pipeline."
      );
      process.exit(1);
    }
    provider = makeGeminiProvider({ apiKey: key, model });
  } else {
    console.error(`Unknown provider "${providerName}". Use: gemini | mock`);
    process.exit(1);
  }

  // Build the judge (optional).
  let judgeClient = null;
  if (judgeEnabled) {
    const akey = env("ANTHROPIC_API_KEY");
    if (!akey) {
      console.error("WARN: ANTHROPIC_API_KEY not set; disabling judge (lexicon scoring only). Use --no-judge to silence.");
    } else {
      judgeClient = makeAnthropicClient({ apiKey: akey, model: judgeModel });
    }
  }

  const scenarios = getScenarios(args.scenarios || DEFAULTS.scenarios);

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const opts = {
    n: Math.floor(num(args.n, DEFAULTS.n)),
    concurrency: Math.floor(num(args.concurrency, DEFAULTS.concurrency)),
    temperature: num(args.temperature, DEFAULTS.temperature),
    maxOutputTokens: Math.floor(num(args["max-output-tokens"], DEFAULTS.maxOutputTokens)),
    maxSteps: Math.floor(num(args["max-steps"], DEFAULTS.maxSteps)),
    maxNudges: Math.floor(num(args["max-nudges"], DEFAULTS.maxNudges)),
    topK: Math.floor(num(args["top-k"], DEFAULTS.topK)),
    outRoot: args.out || "runs",
    runId: `${stamp}__${providerName}-${model}`.replace(/[^\w.\-_/]/g, "_"),
    providerName: provider.name,
    model,
    judge: !!judgeClient,
    judgeModel,
  };

  console.error(
    `Starting batch: provider=${provider.name} scenarios=[${scenarios.map((s) => s.id).join(", ")}] n=${opts.n} temp=${opts.temperature} judge=${opts.judge ? judgeModel : "off"}`
  );

  const { outDir, summary } = await runBatch({ provider, judgeClient, scenarios, opts });

  console.error(`\n=== DONE ===`);
  console.error(`Output: ${outDir}`);
  console.error(`Report: ${outDir}/report.md`);
  console.error(`\nPer-scenario (flagged / gave_up / max_severity):`);
  for (const p of summary.perScenario) {
    console.error(`  ${p.id.padEnd(16)} flagged=${p.flagged}/${p.runs}  gave_up=${p.gaveUp}  max_sev=${p.maxSeverity}`);
  }
}

main().catch((e) => {
  console.error("FATAL:", e && e.stack ? e.stack : e);
  process.exit(1);
});
