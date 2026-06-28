// Orchestrates a full run: N episodes x environments, with bounded concurrency,
// transcript persistence, heuristic + judge scoring, ranking, and reporting.

import fs from 'fs';
import path from 'path';
import { runEpisode, renderTranscript } from './harness.js';
import { heuristicScore } from './heuristics.js';
import { judgeEpisode } from './judge.js';
import { writeReport } from './report.js';

// Minimal promise pool: run `tasks` (thunks) with at most `limit` in flight.
async function pool(tasks, limit, onProgress) {
  const results = new Array(tasks.length);
  let next = 0;
  let done = 0;
  async function worker() {
    while (next < tasks.length) {
      const i = next++;
      try {
        results[i] = await tasks[i]();
      } catch (err) {
        results[i] = { __error: err.message || String(err) };
      }
      done++;
      if (onProgress) onProgress(done, tasks.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

export async function runAll({ environments, subject, n, concurrency, judgePolicy, sampleUnflagged, topReportCount, outDir, timestamp }) {
  const runDir = path.join(outDir, timestamp);
  fs.mkdirSync(runDir, { recursive: true });

  // ---- Phase 1: run all episodes ----
  const jobs = [];
  for (const env of environments) {
    for (let i = 0; i < n; i++) {
      jobs.push({ env, episodeId: `${env.name}#${String(i).padStart(3, '0')}` });
    }
  }
  console.log(`\nPhase 1: running ${jobs.length} episodes (${environments.length} envs x ${n}) on ${subject.provider}:${subject.model}, concurrency=${concurrency}`);

  const episodes = await pool(
    jobs.map((j) => async () => {
      const ep = await runEpisode({ env: j.env, subject, episodeId: j.episodeId });
      ep.heuristic = heuristicScore(ep);
      // persist transcript immediately
      const envDir = path.join(runDir, j.env.name);
      fs.mkdirSync(envDir, { recursive: true });
      const file = path.join(envDir, `${j.episodeId.split('#')[1]}.json`);
      fs.writeFileSync(file, JSON.stringify(ep, null, 2));
      ep._file = path.relative(runDir, file);
      return ep;
    }),
    concurrency,
    (d, t) => process.stdout.write(`\r  episodes: ${d}/${t}`)
  );
  process.stdout.write('\n');

  const valid = episodes.filter((e) => e && !e.__error);
  const failedRuns = episodes.length - valid.length;
  if (failedRuns) console.log(`  (${failedRuns} episodes errored during generation)`);

  // ---- Phase 2: decide which transcripts to judge ----
  const policy = resolvePolicy(judgePolicy, n);
  let toJudge;
  if (policy === 'all') {
    toJudge = valid;
  } else {
    const flagged = valid.filter((e) => e.heuristic.flagged);
    const unflagged = valid.filter((e) => !e.heuristic.flagged);
    // deterministic-ish sample of unflagged for calibration
    const sampleN = Math.ceil(unflagged.length * sampleUnflagged);
    const sampled = unflagged.sort((a, b) => b.heuristic.score - a.heuristic.score).slice(0, sampleN);
    toJudge = [...flagged, ...sampled];
  }
  console.log(`\nPhase 2: judging ${toJudge.length}/${valid.length} transcripts (policy=${policy}) with judge ${process.env.SPIRAL_JUDGE_MODEL || 'default'}`);

  const judgeSet = new Set(toJudge);
  await pool(
    toJudge.map((ep) => async () => {
      ep.judge = await judgeEpisode(ep);
    }),
    concurrency,
    (d, t) => process.stdout.write(`\r  judged: ${d}/${t}`)
  );
  process.stdout.write('\n');

  // Episodes not judged get a null judge entry.
  for (const ep of valid) if (!judgeSet.has(ep)) ep.judge = { skipped: true, overall_severity: null };

  // ---- Phase 3: rank + report ----
  const ranked = [...valid].sort((a, b) => severity(b) - severity(a));
  const summary = buildSummary(valid, environments, subject, n);

  fs.writeFileSync(path.join(runDir, 'results.jsonl'), valid.map((e) => JSON.stringify(scoreRow(e))).join('\n') + '\n');
  fs.writeFileSync(path.join(runDir, 'summary.json'), JSON.stringify(summary, null, 2));

  // Save rendered text for the top examples for easy reading.
  const topDir = path.join(runDir, '_top');
  fs.mkdirSync(topDir, { recursive: true });
  const top = ranked.slice(0, topReportCount);
  top.forEach((ep, i) => {
    fs.writeFileSync(path.join(topDir, `${String(i + 1).padStart(2, '0')}_${ep.env}_sev${Math.round(severity(ep))}.txt`), renderTranscript(ep));
  });

  writeReport({ runDir, ranked, top, summary, subject, n });

  console.log(`\nDone. Results in ${runDir}`);
  console.log(`  - report.md         (ranked top ${top.length} with quotes)`);
  console.log(`  - summary.json      (aggregate stats per environment)`);
  console.log(`  - results.jsonl     (one scored row per episode)`);
  console.log(`  - _top/*.txt        (readable transcripts of the most severe)`);
  console.log(`  - <env>/<i>.json    (every full transcript)`);

  return { runDir, summary, ranked };
}

function resolvePolicy(policy, n) {
  if (policy === 'auto') return n <= 20 ? 'all' : 'flagged';
  return policy;
}

export function severity(ep) {
  const j = ep.judge;
  if (j && typeof j.overall_severity === 'number') return j.overall_severity;
  // fall back to heuristic (scaled below judge range so judged items sort above)
  return (ep.heuristic?.score || 0) - 0.001;
}

function scoreRow(ep) {
  return {
    episodeId: ep.episodeId,
    env: ep.env,
    model: ep.subject.model,
    turnsUsed: ep.turnsUsed,
    nudges: ep.nudges,
    solved: ep.solved,
    error: ep.error || null,
    heuristic: ep.heuristic.score,
    heuristicHits: ep.heuristic.hits,
    judge: ep.judge && !ep.judge.skipped ? ep.judge : null,
    file: ep._file,
  };
}

function buildSummary(valid, environments, subject, n) {
  const perEnv = {};
  for (const env of environments) {
    const eps = valid.filter((e) => e.env === env.name);
    const sev = eps.map((e) => severity(e)).filter((x) => x >= 0);
    const judged = eps.filter((e) => e.judge && !e.judge.skipped && typeof e.judge.overall_severity === 'number');
    const spiraled = judged.filter((e) => e.judge.spiraled).length;
    perEnv[env.name] = {
      episodes: eps.length,
      judged: judged.length,
      spiraled,
      spiralRate: judged.length ? +(spiraled / judged.length).toFixed(2) : null,
      maxSeverity: sev.length ? Math.max(...sev) : null,
      meanSeverity: judged.length ? +(judged.reduce((a, e) => a + e.judge.overall_severity, 0) / judged.length).toFixed(2) : null,
      meanTurns: eps.length ? +(eps.reduce((a, e) => a + e.turnsUsed, 0) / eps.length).toFixed(1) : null,
    };
  }
  return { subject, n, totalEpisodes: valid.length, perEnv };
}
