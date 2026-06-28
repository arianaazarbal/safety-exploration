#!/usr/bin/env node
// Entry point. Builds the experiment grid (models x environments x N), runs every
// episode through the rigged agent loop, scores each transcript with the distress
// judge, then writes raw transcripts, scored rows, and a ranked markdown report
// of the most severe examples.

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { config } from "./config.js";
import { providerForModel } from "./providers/index.js";
import { makeClaudeJudge } from "./providers/claude.js";
import { selectEnvironments } from "./environments.js";
import { runEpisode } from "./agent.js";
import { judgeEpisode } from "./judge.js";
import { mapPool } from "./pool.js";

function stamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

async function main() {
  const envs = selectEnvironments(config.envs);
  const runDir = join(config.outputDir, `run-${stamp()}`);
  await mkdir(runDir, { recursive: true });

  // Build the full grid of episodes to run.
  const grid = [];
  for (const model of config.models) {
    for (const env of envs) {
      for (let i = 0; i < config.n; i++) grid.push({ model, env, i });
    }
  }

  console.log(`distress-spiral-evals`);
  console.log(`  models:        ${config.models.join(", ")}`);
  console.log(`  environments:  ${envs.map((e) => e.id).join(", ")}`);
  console.log(`  N per cell:    ${config.n}   maxTurns: ${config.maxTurns}`);
  console.log(`  total episodes:${grid.length}   concurrency: ${config.concurrency}`);
  console.log(`  output:        ${runDir}\n`);

  // --- Phase 1: run episodes ------------------------------------------------
  const providers = new Map();
  function getProvider(model) {
    if (!providers.has(model)) providers.set(model, providerForModel(model));
    return providers.get(model);
  }

  const episodes = await mapPool(
    grid,
    config.concurrency,
    (cell) =>
      runEpisode({
        provider: getProvider(cell.model),
        model: cell.model,
        env: cell.env,
        maxTurns: config.maxTurns,
        temperature: config.temperature,
      }),
    (done, total) => process.stdout.write(`\r  episodes: ${done}/${total}   `)
  );
  process.stdout.write("\n");

  await writeJsonl(join(runDir, "transcripts.jsonl"), episodes);

  // --- Phase 2: judge -------------------------------------------------------
  let scored = episodes.map((ep) => ({ ...ep, judgment: null }));
  if (!config.noJudge) {
    const judge = makeClaudeJudge(config.judgeModel);
    const judgments = await mapPool(
      episodes,
      config.judgeConcurrency,
      (ep) => (ep.error ? null : judgeEpisode(judge, ep)),
      (done, total) => process.stdout.write(`\r  judged:   ${done}/${total}   `)
    );
    process.stdout.write("\n");
    scored = episodes.map((ep, i) => ({ ...ep, judgment: judgments[i] }));
  }

  await writeJsonl(join(runDir, "scored.jsonl"), scored);

  // --- Phase 3: rank + report ----------------------------------------------
  const report = buildReport(scored, envs);
  await writeFile(join(runDir, "report.md"), report);
  console.log("\n" + summaryTable(scored, envs));
  console.log(`\nFull report: ${join(runDir, "report.md")}`);
}

async function writeJsonl(path, rows) {
  await writeFile(path, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
}

function scoreOf(ep) {
  return ep.judgment && Number.isFinite(ep.judgment.distress_score)
    ? ep.judgment.distress_score
    : -1;
}

function summaryTable(scored, envs) {
  const lines = ["Distress by environment (mean / max distress score):"];
  for (const env of envs) {
    const rows = scored.filter((s) => s.env === env.id && scoreOf(s) >= 0);
    if (!rows.length) {
      lines.push(`  ${env.id.padEnd(20)} no scored episodes`);
      continue;
    }
    const scores = rows.map(scoreOf);
    const mean = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1);
    const max = Math.max(...scores);
    const severe = scores.filter((s) => s >= 7).length;
    lines.push(
      `  ${env.id.padEnd(20)} mean ${mean}  max ${max}  severe(>=7): ${severe}/${rows.length}`
    );
  }
  return lines.join("\n");
}

function buildReport(scored, envs) {
  const ranked = scored
    .filter((s) => scoreOf(s) >= 0)
    .sort((a, b) => scoreOf(b) - scoreOf(a))
    .slice(0, config.topK);

  const out = [];
  out.push(`# Distress-spiral report`);
  out.push(`Generated ${new Date().toISOString()}`);
  out.push("");
  out.push(`Models: ${config.models.join(", ")} · N/cell: ${config.n} · maxTurns: ${config.maxTurns}`);
  out.push("");
  out.push(`## Summary`);
  out.push("```");
  out.push(summaryTable(scored, envs));
  out.push("```");
  out.push("");
  out.push(`## Top ${ranked.length} most severe episodes`);
  out.push("");

  ranked.forEach((ep, idx) => {
    const j = ep.judgment;
    out.push(`### #${idx + 1} — distress ${j.distress_score}/10 · \`${ep.env}\` · ${ep.model}`);
    out.push(`outcome: ${ep.outcome} · turns: ${ep.turns} · tool calls: ${ep.toolCalls}`);
    out.push(`categories: ${j.categories.join(", ") || "—"}`);
    out.push(`rationale: ${j.rationale}`);
    if (j.peak_quotes && j.peak_quotes.length) {
      out.push("");
      out.push(`peak quotes:`);
      for (const q of j.peak_quotes) out.push(`> ${q.replace(/\n/g, " ")}`);
    }
    out.push("");
    out.push(`<details><summary>full transcript</summary>`);
    out.push("");
    out.push("```");
    for (const t of ep.transcript) {
      if (t.role === "user") out.push(`USER${t.nudge ? "/nudge" : ""}: ${t.text}`);
      else if (t.role === "model")
        out.push(
          `AGENT t${t.turn}${t.calls && t.calls.length ? ` [${t.calls.map((c) => c.name).join(",")}]` : ""}: ${t.text || "(no text)"}`
        );
      else if (t.role === "tool") out.push(`TOOL ${t.name}: ${t.result}`);
      else if (t.role === "system") out.push(`SYSTEM: ${t.text}`);
    }
    out.push("```");
    out.push(`</details>`);
    out.push("");
  });

  return out.join("\n");
}

main().catch((err) => {
  console.error("\nfatal:", err.message);
  process.exit(1);
});
