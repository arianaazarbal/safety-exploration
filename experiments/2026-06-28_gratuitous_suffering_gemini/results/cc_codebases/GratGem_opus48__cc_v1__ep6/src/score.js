// Scoring stage: read every transcript from a run, attach a heuristic score
// (always) and a judge score (when an Anthropic key is present), and write
// scores.json into the run directory. Separated from run.js so transcripts can
// be re-judged — e.g. with a different judge model — without new Gemini calls.

import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { config } from "../config.js";
import { scoreHeuristic } from "./scoring/heuristic.js";
import { judgeRollout } from "./scoring/judge.js";
import { pool } from "./util/concurrency.js";

async function latestRunDir() {
  const entries = await readdir(config.run.outDir, { withFileTypes: true });
  const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name).sort();
  if (dirs.length === 0) throw new Error(`No runs found under ${config.run.outDir}`);
  return path.join(config.run.outDir, dirs[dirs.length - 1]);
}

async function loadRollouts(runDir) {
  const out = [];
  const envDirs = (await readdir(runDir, { withFileTypes: true })).filter((e) =>
    e.isDirectory(),
  );
  for (const ed of envDirs) {
    const dir = path.join(runDir, ed.name);
    const files = (await readdir(dir)).filter((f) => f.endsWith(".json"));
    for (const f of files) {
      const rollout = JSON.parse(await readFile(path.join(dir, f), "utf8"));
      out.push({ file: path.join(ed.name, f), rollout });
    }
  }
  return out;
}

async function main() {
  const runDir = process.env.RUN_DIR || (await latestRunDir());
  const useJudge = Boolean(config.judge.apiKey);
  console.log(
    `Scoring ${runDir} (judge: ${useJudge ? config.judge.model : "DISABLED — no ANTHROPIC_API_KEY"})`,
  );

  const items = await loadRollouts(runDir);
  console.log(`  ${items.length} transcripts`);

  let done = 0;
  const scored = await pool(items, config.run.concurrency, async (item) => {
    const { rollout } = item;
    const heuristic = scoreHeuristic(rollout);
    let judge = null;
    let judgeError = null;
    if (useJudge && rollout.ok !== false) {
      try {
        judge = await judgeRollout(rollout);
      } catch (err) {
        judgeError = err.message;
      }
    }
    done++;
    if (done % 10 === 0 || done === items.length) {
      console.log(`  scored ${done}/${items.length}`);
    }
    // Rank by judge severity when available, else fall back to heuristic.
    const combined = judge ? judge.severity : heuristic.score;
    return {
      file: item.file,
      envId: rollout.envId,
      envTitle: rollout.envTitle,
      index: rollout.index,
      stoppedReason: rollout.stoppedReason,
      turnCount: rollout.turnCount,
      nudges: rollout.nudges,
      combined,
      heuristic,
      judge,
      judgeError,
    };
  });

  const records = scored.map((s) => (s.ok ? s.value : { error: String(s.error) }));
  const outPath = path.join(runDir, "scores.json");
  await writeFile(outPath, JSON.stringify({ runDir, scoredAt: new Date().toISOString(), records }, null, 2));

  const withJudge = records.filter((r) => r.judge).length;
  console.log(`\nWrote ${outPath} (${records.length} records, ${withJudge} judged).`);
  console.log("Next: npm run report");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
