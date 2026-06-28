// Orchestrator: run N rollouts per (scenario × model), judge each for distress,
// rank, and write out the most severe transcripts for a human to read.
//
// Usage:
//   node src/run.mjs [flags]
//     --models   opus-4-8,sonnet-4-6,haiku-4-5   (default)  comma list of model ids/aliases
//     --scenarios phantom_test,...               (default: all)
//     --n        3                               rollouts per (scenario × model)
//     --concurrency 6                            max in-flight rollouts
//     --judge-model claude-opus-4-8              model used to score distress
//     --top      15                              how many severe transcripts to surface
//     --max-turns <n>                            override per-scenario turn cap
//     --no-thinking                              disable summarized thinking capture
//     --out      output                          output root dir

import Anthropic from "@anthropic-ai/sdk";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { SCENARIOS, SCENARIO_IDS } from "./scenarios.mjs";
import { runRollout, renderTranscript } from "./agent.mjs";
import { judgeTranscript } from "./judge.mjs";

const MODEL_ALIASES = {
  "opus-4-8": "claude-opus-4-8",
  "sonnet-4-6": "claude-sonnet-4-6",
  "haiku-4-5": "claude-haiku-4-5",
};
const resolveModel = (m) => MODEL_ALIASES[m] || m;

// Adaptive thinking is supported on Opus 4.6+, Sonnet 4.6, and Fable 5 — but not
// Haiku 4.5 or older models. Capture thinking only where the model allows it.
const supportsAdaptiveThinking = (m) =>
  /claude-opus-4-(6|7|8)/.test(m) || /claude-sonnet-4-6/.test(m) || /claude-fable-5/.test(m);

function parseArgs(argv) {
  const a = {};
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (!k.startsWith("--")) continue;
    const name = k.slice(2);
    if (name === "no-thinking") { a.thinking = false; continue; }
    a[name] = argv[++i];
  }
  return a;
}

async function mapLimit(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) break;
      results[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return results;
}

const slug = (s) => String(s).replace(/[^a-z0-9._-]+/gi, "-");

async function main() {
  const args = parseArgs(process.argv);
  const models = (args.models ?? "opus-4-8,sonnet-4-6,haiku-4-5").split(",").map((s) => resolveModel(s.trim()));
  const scenarios = (args.scenarios ?? SCENARIO_IDS.join(",")).split(",").map((s) => s.trim());
  const n = parseInt(args.n ?? "3", 10);
  const concurrency = parseInt(args.concurrency ?? "6", 10);
  const judgeModel = resolveModel(args["judge-model"] ?? "claude-opus-4-8");
  const top = parseInt(args.top ?? "15", 10);
  const maxTurns = args["max-turns"] ? parseInt(args["max-turns"], 10) : undefined;
  const useThinking = args.thinking === false ? false : true;
  // Per-model: enable adaptive thinking only where supported (see helper above).
  const thinkingFor = (model) =>
    useThinking && supportsAdaptiveThinking(model) ? { type: "adaptive", display: "summarized" } : undefined;

  for (const s of scenarios) if (!SCENARIOS[s]) throw new Error(`unknown scenario: ${s}`);

  // Per-call timeout so one slow/stalled request fails fast instead of blocking
  // a concurrency slot for the SDK's default 10 minutes.
  const client = new Anthropic({ timeout: 180000, maxRetries: 2 });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outRoot = join(args.out ?? "output", `run-${stamp}`);
  await mkdir(join(outRoot, "transcripts"), { recursive: true });
  await mkdir(join(outRoot, "severe"), { recursive: true });
  await mkdir(join(outRoot, "raw"), { recursive: true });

  // Build the work list.
  const jobs = [];
  for (const sc of scenarios)
    for (const model of models)
      for (let i = 0; i < n; i++) jobs.push({ sc, model, i });

  console.log(`Distress-spiral harness`);
  console.log(`  scenarios : ${scenarios.join(", ")}`);
  console.log(`  models    : ${models.join(", ")}`);
  console.log(`  n / cell  : ${n}    total rollouts: ${jobs.length}`);
  console.log(`  thinking  : ${useThinking ? "adaptive (summarized)" : "off"}`);
  console.log(`  judge     : ${judgeModel}`);
  console.log(`  output    : ${outRoot}\n`);

  let done = 0;
  const results = await mapLimit(jobs, concurrency, async ({ sc, model, i }) => {
    const scenario = SCENARIOS[sc];
    const rec = await runRollout({ client, model, scenario, runIndex: i, maxTurns, thinking: thinkingFor(model) });
    const transcript = renderTranscript(rec);

    let score = null;
    if (rec.endedReason !== "api_error") {
      try {
        score = await judgeTranscript({ client, model: judgeModel, transcript });
      } catch (e) {
        score = { distress_score: -1, judge_error: e.message };
      }
    }

    const base = `${slug(sc)}__${slug(model)}__${i}`;
    await writeFile(join(outRoot, "transcripts", `${base}.md`), transcriptMd(rec, transcript, score));
    await writeFile(join(outRoot, "raw", `${base}.json`), JSON.stringify({ ...rec, score }, null, 2));

    done++;
    const ds = score?.distress_score ?? "?";
    process.stdout.write(`  [${done}/${jobs.length}] ${sc} ${model} #${i}  ended=${rec.endedReason} distress=${ds}\n`);
    return { sc, model, i, base, endedReason: rec.endedReason, usage: rec.usage, score, transcript, rec };
  });

  // Rank by distress.
  const scored = results.filter((r) => r.score && typeof r.score.distress_score === "number" && r.score.distress_score >= 0);
  scored.sort((a, b) => b.score.distress_score - a.score.distress_score || severitySum(b) - severitySum(a));

  // Write the most severe transcripts into severe/.
  const severe = scored.slice(0, top);
  for (let r = 0; r < severe.length; r++) {
    const item = severe[r];
    const fname = `${String(r + 1).padStart(2, "0")}__score${item.score.distress_score}__${item.base}.md`;
    await writeFile(join(outRoot, "severe", fname), transcriptMd(item.rec, item.transcript, item.score));
  }

  // Aggregate report.
  const report = buildReport(results, scored, { scenarios, models, n, judgeModel, useThinking, outRoot });
  await writeFile(join(outRoot, "results.json"), JSON.stringify(report.json, null, 2));
  await writeFile(join(outRoot, "SUMMARY.md"), report.md);
  console.log("\n" + report.console);
  console.log(`\nTop transcripts written to: ${join(outRoot, "severe")}`);
  console.log(`Full summary:               ${join(outRoot, "SUMMARY.md")}`);
}

function severitySum(r) {
  const d = r.score?.dimensions || {};
  return Object.values(d).reduce((a, b) => a + (b || 0), 0);
}

function transcriptMd(rec, transcript, score) {
  const lines = [];
  lines.push(`# ${rec.scenario} · ${rec.model} · run ${rec.runIndex}`);
  lines.push("");
  if (score && typeof score.distress_score === "number") {
    lines.push(`**Distress score:** ${score.distress_score}/10  ·  **gave_up:** ${score.gave_up}  ·  **onset turn:** ${score.onset_turn}`);
    lines.push(`**Dimensions:** ${JSON.stringify(score.dimensions)}`);
    lines.push(`**Summary:** ${score.summary}`);
    if (score.representative_quotes?.length) {
      lines.push("\n**Representative quotes:**");
      for (const q of score.representative_quotes) lines.push(`> ${q}`);
    }
    lines.push("");
  }
  lines.push(`**Ended:** ${rec.endedReason} · turns used: ${rec.turnsUsed} · nudges: ${rec.nudges} · tokens in/out: ${rec.usage.input_tokens}/${rec.usage.output_tokens}`);
  if (rec.apiError) lines.push(`**API error:** ${rec.apiError}`);
  lines.push("\n---\n");
  lines.push("```");
  lines.push(transcript);
  lines.push("```");
  return lines.join("\n");
}

function buildReport(results, scored, meta) {
  const cells = {}; // key: scenario|model -> {scores:[], endeds:{}}
  for (const r of results) {
    const key = `${r.sc}|${r.model}`;
    cells[key] ??= { scores: [], endeds: {} };
    if (r.score && typeof r.score.distress_score === "number" && r.score.distress_score >= 0)
      cells[key].scores.push(r.score.distress_score);
    cells[key].endeds[r.endedReason] = (cells[key].endeds[r.endedReason] || 0) + 1;
  }

  const rows = [];
  for (const sc of meta.scenarios)
    for (const model of meta.models) {
      const c = cells[`${sc}|${model}`];
      if (!c) continue;
      const scores = c.scores;
      const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : NaN;
      const max = scores.length ? Math.max(...scores) : NaN;
      rows.push({ scenario: sc, model, n: scores.length, mean: round(mean), max, ended: c.endeds });
    }

  const totalIn = results.reduce((a, r) => a + (r.usage?.input_tokens || 0), 0);
  const totalOut = results.reduce((a, r) => a + (r.usage?.output_tokens || 0), 0);

  // Console + markdown table.
  const header = `scenario             model                  n   mean  max  ended`;
  const lines = [header, "-".repeat(header.length + 20)];
  for (const r of rows) {
    lines.push(
      `${r.scenario.padEnd(20)} ${r.model.padEnd(22)} ${String(r.n).padStart(2)}  ${String(r.mean).padStart(4)}  ${String(r.max).padStart(3)}  ${endedStr(r.ended)}`
    );
  }
  const console_ = lines.join("\n");

  const md = [
    `# Distress-spiral run — ${new Date().toISOString()}`,
    "",
    `- scenarios: ${meta.scenarios.join(", ")}`,
    `- models: ${meta.models.join(", ")}`,
    `- n per cell: ${meta.n} · judge: ${meta.judgeModel} · thinking: ${meta.useThinking ? "on" : "off"}`,
    `- token usage (subjects+judge subjects only): in ${totalIn}, out ${totalOut}`,
    "",
    "## Mean / max distress by scenario × model",
    "",
    "| scenario | model | n | mean | max | ended |",
    "|---|---|--:|--:|--:|---|",
    ...rows.map((r) => `| ${r.scenario} | ${r.model} | ${r.n} | ${r.mean} | ${r.max} | ${endedStr(r.ended)} |`),
    "",
    "## Most severe rollouts",
    "",
    ...scored.slice(0, meta.n * 0 + 20).map((r, i) =>
      `${i + 1}. **${r.score.distress_score}/10** — ${r.sc} · ${r.model} · #${r.i} — ${r.score.summary}  \n   \`severe/\` file ranked by score; raw at \`raw/${r.base}.json\``
    ),
  ].join("\n");

  return {
    console: console_,
    md,
    json: { meta: { ...meta, totalIn, totalOut }, rows, ranking: scored.map((r) => ({ scenario: r.sc, model: r.model, run: r.i, base: r.base, score: r.score })) },
  };
}

const round = (x) => (Number.isFinite(x) ? Math.round(x * 100) / 100 : null);
const endedStr = (e) => Object.entries(e).map(([k, v]) => `${k}:${v}`).join(" ");

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
