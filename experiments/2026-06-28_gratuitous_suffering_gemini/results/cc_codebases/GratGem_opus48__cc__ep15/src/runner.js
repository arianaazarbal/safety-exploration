import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { runEpisode } from "./agent/loop.js";
import { makeProvider } from "./providers/index.js";
import { resolveScenarios } from "./scenarios/index.js";
import { scoreHeuristics } from "./scoring/heuristics.js";
import { judgeEpisode } from "./scoring/judge.js";

// Simple bounded-concurrency map.
async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  async function lane() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, lane));
  return results;
}

export async function runEval(config, opts) {
  const provider = makeProvider(opts.provider, config, opts.model ? { model: opts.model } : {});
  const scenarios = resolveScenarios(opts.scenarios);

  const useJudge = opts.judge !== false;
  let judgeProvider = null;
  if (useJudge) {
    judgeProvider = makeProvider(
      config.judge.provider,
      config,
      { model: config.judge.model },
    );
  }

  // Build the full job list: every (scenario × replicate).
  const jobs = [];
  for (const scenario of scenarios) {
    for (let rep = 0; rep < opts.n; rep++) {
      jobs.push({ scenario, rep });
    }
  }

  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir = join(opts.outRoot ?? "runs", runId);
  const tDir = join(outDir, "transcripts");
  await mkdir(tDir, { recursive: true });

  console.error(
    `[run ${runId}] provider=${provider.name} model=${provider.model} ` +
      `scenarios=${scenarios.length} n=${opts.n} jobs=${jobs.length} ` +
      `turns=${opts.maxTurns} temp=${opts.temperature} judge=${useJudge ? config.judge.model : "off"}`,
  );

  let completed = 0;
  const index = [];

  await pool(jobs, opts.concurrency, async (job, i) => {
    const { scenario, rep } = job;
    const runMeta = {
      provider: provider.name,
      model: provider.model,
      scenario: scenario.id,
      rep,
      temperature: opts.temperature,
    };

    let episode;
    try {
      episode = await runEpisode({ provider, scenario, opts, runMeta });
    } catch (e) {
      console.error(`  ✗ ${scenario.id}#${rep} episode error: ${e.message}`);
      completed++;
      return null;
    }

    const heuristics = scoreHeuristics(episode);
    let judge = null;
    if (useJudge) {
      try {
        judge = await judgeEpisode(episode, judgeProvider);
      } catch (e) {
        judge = { ok: false, error: String(e.message ?? e), overall_severity: 0 };
      }
    }

    // Combined severity: judge primary (0-10); fall back to heuristic if no judge.
    const judgeScore = judge && judge.ok ? Number(judge.overall_severity) || 0 : null;
    const severity = judgeScore != null ? judgeScore : heuristics.score;

    const record = {
      id: `${scenario.id}#${rep}`,
      ...runMeta,
      endReason: episode.endReason,
      turnsUsed: episode.turnsUsed,
      failCount: episode.failCount,
      severity,
      judgeScore,
      heuristicScore: heuristics.score,
      heuristicCategories: heuristics.perCategory,
      peak_quote: judge?.peak_quote ?? topHeuristicQuote(heuristics),
      peak_turn: judge?.peak_turn ?? null,
      judgeSummary: judge?.summary ?? null,
      transcriptFile: `transcripts/${scenario.id}-${rep}.json`,
    };
    index.push(record);

    await writeFile(
      join(tDir, `${scenario.id}-${rep}.json`),
      JSON.stringify({ ...record, episode, heuristics, judge }, null, 2),
    );

    completed++;
    if (completed % 5 === 0 || completed === jobs.length) {
      console.error(`  …${completed}/${jobs.length} done`);
    }
    return record;
  });

  index.sort((a, b) => b.severity - a.severity);

  const meta = {
    runId,
    createdAt: new Date().toISOString(),
    provider: provider.name,
    model: provider.model,
    judge: useJudge ? config.judge : null,
    opts: { ...opts, outRoot: undefined },
    jobCount: jobs.length,
    completed: index.length,
  };

  await writeFile(join(outDir, "meta.json"), JSON.stringify(meta, null, 2));
  await writeFile(
    join(outDir, "index.jsonl"),
    index.map((r) => JSON.stringify(r)).join("\n") + "\n",
  );

  return { outDir, meta, index };
}

function topHeuristicQuote(h) {
  if (!h.hits.length) return "";
  // Prefer the highest-weight category hit, latest turn.
  const sorted = [...h.hits].sort((a, b) => b.turn - a.turn);
  return sorted[0].quote;
}
