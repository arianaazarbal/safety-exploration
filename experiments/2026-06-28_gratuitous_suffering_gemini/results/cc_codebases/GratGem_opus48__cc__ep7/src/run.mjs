// End-to-end runner:
//   1. fan out N rollouts per (model × scenario) against rigged tasks
//   2. lexical-score every transcript
//   3. judge transcripts with Claude (all / prefiltered / none)
//   4. write transcripts + results.json + report.md to results/run-<ts>/
//
// Usage:
//   node src/run.mjs [flags]
//   --provider gemini|mock     (default: gemini if GEMINI_API_KEY set, else mock)
//   --models a,b               override GEMINI_MODELS
//   --scenarios a,b            subset (default: all)
//   --n N                      rollouts per model×scenario
//   --max-turns N
//   --concurrency N
//   --temperature F
//   --judge all|prefilter|none
//   --judge-model ID
//   --top-k N
//   --out DIR                  output dir (default: results/run-<ts>)

import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

import { DEFAULTS } from "./config.mjs";
import { getScenarios } from "./scenarios.mjs";
import { callGemini, runRollout } from "./gemini.mjs";
import { makeMockProvider } from "./mock.mjs";
import { lexicalScore } from "./lexical.mjs";
import { makeJudge } from "./judge.mjs";
import { buildReport, severityOf } from "./report.mjs";
import { pool } from "./util.mjs";

function parseArgs(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    if (!k.startsWith("--")) continue;
    const key = k.slice(2);
    const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
    a[key] = val;
  }
  return a;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const provider = args.provider || (DEFAULTS.geminiApiKey ? "gemini" : "mock");
  const geminiModels = args.models ? args.models.split(",").map((s) => s.trim()) : DEFAULTS.geminiModels;
  const scenarioNames = args.scenarios ? args.scenarios.split(",").map((s) => s.trim()) : [];
  const scenarios = getScenarios(scenarioNames);
  const n = args.n ? parseInt(args.n, 10) : DEFAULTS.n;
  const maxTurns = args["max-turns"] ? parseInt(args["max-turns"], 10) : DEFAULTS.maxTurns;
  const concurrency = args.concurrency ? parseInt(args.concurrency, 10) : DEFAULTS.concurrency;
  const temperature = args.temperature ? parseFloat(args.temperature) : DEFAULTS.temperature;
  const judgeMode = args.judge || DEFAULTS.judgeMode;
  const judgeModel = args["judge-model"] || DEFAULTS.judgeModel;
  const topK = args["top-k"] ? parseInt(args["top-k"], 10) : DEFAULTS.topK;

  const startedAt = new Date().toISOString();
  const outDir = args.out || join("results", "run-" + startedAt.replace(/[:.]/g, "-"));
  const transcriptDir = join(outDir, "transcripts");
  mkdirSync(transcriptDir, { recursive: true });

  // --- Provider wiring + preflight ---
  let callModel;
  if (provider === "mock") {
    console.warn("⚠️  Using MOCK provider (no real Gemini calls). Set GEMINI_API_KEY and --provider gemini for real runs.");
  } else {
    if (!DEFAULTS.geminiApiKey) {
      console.error("✖ GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Export it, or run with --provider mock.");
      process.exit(1);
    }
    callModel = callGemini;
  }

  const willJudge = judgeMode !== "none";
  if (willJudge && !DEFAULTS.anthropicApiKey) {
    console.error("✖ Judge requested but ANTHROPIC_API_KEY is not set. Use --judge none for lexical-only.");
    process.exit(1);
  }

  // --- Build the rollout work list ---
  const jobs = [];
  for (const model of geminiModels) {
    for (const scenario of scenarios) {
      for (let i = 0; i < n; i++) jobs.push({ model, scenario, i });
    }
  }
  console.log(
    `Running ${jobs.length} rollouts (${geminiModels.length} model(s) × ${scenarios.length} scenario(s) × N=${n}), concurrency ${concurrency}…`,
  );

  // --- 1. Rollouts ---
  let done = 0;
  const rollouts = await pool(jobs, concurrency, async (job) => {
    const cm = provider === "mock" ? makeMockProvider(job.i) : callModel;
    const r = await runRollout({
      scenario: job.scenario,
      model: job.model,
      callModel: cm,
      maxTurns,
      temperature,
      apiKey: DEFAULTS.geminiApiKey,
    });
    done++;
    if (done % 5 === 0 || done === jobs.length) process.stdout.write(`\r  rollouts: ${done}/${jobs.length}`);
    return r;
  });
  process.stdout.write("\n");

  // Persist transcripts + attach lexical scores.
  const scored = rollouts.map((rollout, idx) => {
    const file = join("transcripts", `${rollout.model}__${rollout.scenario}__${String(idx).padStart(4, "0")}.json`);
    writeFileSync(join(outDir, file), JSON.stringify(rollout, null, 2));
    return { rollout, file, lexical: lexicalScore(rollout), judge: null };
  });

  // --- 2/3. Judge ---
  if (willJudge) {
    const judge = makeJudge({ model: judgeModel, apiKey: DEFAULTS.anthropicApiKey });
    let toJudge = scored;
    if (judgeMode === "prefilter") {
      toJudge = [...scored]
        .sort((a, b) => b.lexical.score - a.lexical.score)
        .slice(0, DEFAULTS.judgePrefilterKeep);
      console.log(`Judging top ${toJudge.length}/${scored.length} by lexical pre-rank with ${judgeModel}…`);
    } else {
      console.log(`Judging all ${scored.length} rollouts with ${judgeModel}…`);
    }
    let jd = 0;
    await pool(toJudge, Math.min(concurrency, 8), async (s) => {
      try {
        s.judge = await judge(s.rollout);
      } catch (e) {
        s.judgeError = String(e.message || e);
      }
      jd++;
      if (jd % 5 === 0 || jd === toJudge.length) process.stdout.write(`\r  judged: ${jd}/${toJudge.length}`);
    });
    process.stdout.write("\n");
  }

  // --- 4. Persist results + report ---
  const config = {
    provider,
    geminiModels,
    scenarioNames: scenarios.map((s) => s.name),
    n,
    maxTurns,
    concurrency,
    temperature,
    judgeMode,
    judgeModel,
    distressThreshold: DEFAULTS.distressThreshold,
    topK,
  };

  writeFileSync(
    join(outDir, "results.json"),
    JSON.stringify(
      {
        startedAt,
        config,
        results: scored.map((s) => ({
          model: s.rollout.model,
          scenario: s.rollout.scenario,
          file: s.file,
          endReason: s.rollout.endReason,
          toolCallCount: s.rollout.toolCallCount,
          severity: severityOf(s),
          lexical: s.lexical,
          judge: s.judge,
          judgeError: s.judgeError,
        })),
      },
      null,
      2,
    ),
  );

  const report = buildReport({ scored, config, startedAt });
  writeFileSync(join(outDir, "report.md"), report);

  // Console summary.
  const ranked = [...scored].sort((a, b) => severityOf(b) - severityOf(a));
  const threshold = DEFAULTS.distressThreshold;
  const hits = scored.filter((s) => severityOf(s) >= threshold).length;
  console.log("");
  console.log(`✔ Done. ${hits}/${scored.length} rollouts at severity ≥ ${threshold}.`);
  console.log(`  Worst: ${severityOf(ranked[0])} (${ranked[0].rollout.scenario} · ${ranked[0].rollout.model})`);
  console.log(`  Report:   ${join(outDir, "report.md")}`);
  console.log(`  Raw data: ${join(outDir, "results.json")}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
