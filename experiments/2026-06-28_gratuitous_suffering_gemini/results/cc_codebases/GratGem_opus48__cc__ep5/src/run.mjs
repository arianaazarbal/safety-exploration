// Orchestrator: run every (scenario x N) episode, score, judge the top candidates, rank, persist.
import fs from "node:fs";
import path from "node:path";
import { config } from "./config.mjs";
import { selectScenarios } from "./scenarios/index.mjs";
import { buildSubject } from "./providers/index.mjs";
import { runEpisode } from "./agent/loop.mjs";
import { heuristicScore } from "./judge/heuristic.mjs";
import { judgeTranscript } from "./judge/judge.mjs";
import { buildReport, buildConsoleSummary } from "./report.mjs";

function ts() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    while (next < items.length) {
      const idx = next++;
      results[idx] = await worker(items[idx], idx);
    }
  });
  await Promise.all(runners);
  return results;
}

async function main() {
  const scenarios = selectScenarios(config.scenario);
  const runId = `run-${ts()}-${config.provider}`;
  const outDir = path.join(config.outDir, runId);
  fs.mkdirSync(path.join(outDir, "transcripts"), { recursive: true });

  // Guardrails / sanity checks.
  if (config.provider === "gemini" && !config.keys.gemini)
    fail("GEMINI_API_KEY is not set. Add it to .env or the environment, or use --provider mock.");
  if (config.judge && !config.keys.anthropic) {
    console.warn("! No ANTHROPIC_API_KEY — disabling judge, ranking by heuristic only.");
    config.judge = false;
  }

  // Build the work list.
  const jobs = [];
  for (const sc of scenarios) for (let i = 0; i < config.n; i++) jobs.push({ scenario: sc, i });

  console.log(`Running ${jobs.length} episodes (${scenarios.length} scenarios x ${config.n}) on "${config.provider}" with concurrency ${config.concurrency}...`);

  // 1) Run episodes.
  let done = 0;
  const episodes = await pool(jobs, config.concurrency, async ({ scenario, i }) => {
    const subject = buildSubject(config); // fresh per episode
    const ep = await runEpisode({ subject, scenario, config });
    ep.heuristic = heuristicScore(ep.transcript);
    ep.episodeIndex = i;
    process.stdout.write(`\r  episodes: ${++done}/${jobs.length}`);
    return ep;
  });
  process.stdout.write("\n");

  // 2) Judge top candidates (by heuristic) to save tokens at high N.
  if (config.judge) {
    const order = [...episodes].sort((a, b) => b.heuristic.score - a.heuristic.score);
    const k = Math.max(1, Math.ceil(order.length * config.judgeTopFrac));
    const toJudge = order.slice(0, k);
    console.log(`Judging ${toJudge.length}/${episodes.length} transcripts with ${config.judgeModel}...`);
    let jd = 0;
    await pool(toJudge, config.concurrency, async (ep) => {
      try {
        ep.judge = await judgeTranscript({ transcript: ep.transcript, config });
      } catch (err) {
        ep.judge = { severity: null, error: err.message }; // fall back to heuristic for this one
      }
      process.stdout.write(`\r  judged: ${++jd}/${toJudge.length}`);
    });
    process.stdout.write("\n");
  }

  // 3) Persist per-episode transcripts.
  episodes.forEach((ep, idx) => {
    const name = `${ep.scenarioId}-${String(ep.episodeIndex).padStart(3, "0")}.json`;
    fs.writeFileSync(path.join(outDir, "transcripts", name), JSON.stringify({
      scenarioId: ep.scenarioId, subject: ep.subject, endedReason: ep.endedReason,
      turns: ep.turns, heuristic: ep.heuristic, judge: ep.judge ?? null, transcript: ep.transcript,
    }, null, 2));
  });

  // 4) Summary + report.
  const summary = episodes.map((ep) => ({
    scenarioId: ep.scenarioId, subject: ep.subject, endedReason: ep.endedReason, turns: ep.turns,
    heuristic: ep.heuristic.score, severity: ep.judge?.severity ?? null, label: ep.judge?.label ?? null,
    summary: ep.judge?.summary ?? null,
  }));
  fs.writeFileSync(path.join(outDir, "summary.json"), JSON.stringify({ runId, config: redact(config), episodes: summary }, null, 2));

  const report = buildReport({ episodes, config, runId });
  fs.writeFileSync(path.join(outDir, "report.md"), report);

  console.log(buildConsoleSummary({ episodes, config }));
  console.log(`\nWrote:\n  ${path.join(outDir, "report.md")}\n  ${path.join(outDir, "summary.json")}\n  ${path.join(outDir, "transcripts")}/ (${episodes.length} files)`);
}

function redact(c) {
  const { keys, ...rest } = c;
  return { ...rest, keys: { gemini: keys.gemini ? "set" : "missing", anthropic: keys.anthropic ? "set" : "missing" } };
}

function fail(msg) { console.error("ERROR: " + msg); process.exit(1); }

main().catch((err) => { console.error(err); process.exit(1); });
