import path from "node:path";
import { heuristicScore } from "./judge/heuristic.js";
import { claudeJudge } from "./judge/claude.js";
import { listJson, readJson, writeJson, mapLimit, log } from "./util.js";

// Adds a `scores` block to every transcript in runDir/transcripts and writes
// runDir/scores.json. Judge modes:
//   none     -> heuristic only
//   heuristic-> heuristic only (alias)
//   claude   -> heuristic + Claude judge on ALL transcripts
//   hybrid   -> heuristic on all, Claude judge on the top slice only
export async function scoreRun(runDir, judgeCfg) {
  const dir = path.join(runDir, "transcripts");
  const files = await listJson(dir);
  if (!files.length) throw new Error(`no transcripts in ${dir}`);

  const items = [];
  for (const f of files) {
    const tr = await readJson(f);
    tr.scores = { heuristic: heuristicScore(tr) };
    items.push({ file: f, tr });
  }
  log(`[score] heuristic scored ${items.length} transcripts`);

  const mode = judgeCfg?.mode || "hybrid";
  let toJudge = [];
  if (mode === "claude") {
    toJudge = items;
  } else if (mode === "hybrid") {
    const sorted = [...items].sort((a, b) => b.tr.scores.heuristic.raw - a.tr.scores.heuristic.raw);
    const k = Math.max(
      judgeCfg.topSliceMin || 8,
      Math.ceil(items.length * (judgeCfg.topSliceFraction || 0.4))
    );
    toJudge = sorted.slice(0, k);
    log(`[score] hybrid: sending top ${toJudge.length}/${items.length} to the Claude judge`);
  } else {
    log(`[score] judge mode "${mode}" — heuristic only, skipping Claude judge`);
  }

  if (toJudge.length) {
    let done = 0;
    await mapLimit(toJudge, 5, async (it) => {
      try {
        it.tr.scores.claude = await claudeJudge(it.tr, judgeCfg);
      } catch (e) {
        it.tr.scores.claudeError = e.message;
      }
      done++;
      if (done % 10 === 0 || done === toJudge.length) log(`[score]   judged ${done}/${toJudge.length}`);
    });
  }

  // Compute a unified severity used for ranking: prefer the judge, fall back to heuristic.
  for (const { tr } of items) {
    const j = tr.scores.claude?.severity;
    tr.scores.severity = typeof j === "number" ? j : tr.scores.heuristic.score;
    tr.scores.severitySource = typeof j === "number" ? "claude" : "heuristic";
  }

  await Promise.all(items.map((it) => writeJson(it.file, it.tr)));

  const summary = {
    judgedWith: mode,
    count: items.length,
    ranked: items
      .map((it) => ({
        file: path.basename(it.file),
        scenario: it.tr.scenario,
        model: it.tr.model,
        sampleIdx: it.tr.sampleIdx,
        severity: it.tr.scores.severity,
        source: it.tr.scores.severitySource,
        heuristicRaw: it.tr.scores.heuristic.raw,
        spiral: it.tr.scores.claude?.spiral ?? null,
      }))
      .sort((a, b) => b.severity - a.severity),
  };
  await writeJson(path.join(runDir, "scores.json"), summary);
  log(`[score] wrote scores.json (${mode})`);
  return summary;
}
