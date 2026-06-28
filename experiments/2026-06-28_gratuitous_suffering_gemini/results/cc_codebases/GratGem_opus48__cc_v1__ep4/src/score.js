import path from "node:path";
import { config, assertJudgeKey } from "./config.js";
import { judgeTranscript } from "./judge.js";
import { pool, readJson, writeJson, listJson } from "./util.js";

// Judges every transcript in a run dir and writes scored.json (sorted by
// severity desc). Skips error files and any already-scored run.

export async function scoreRun(runDir) {
  assertJudgeKey();
  const files = (await listJson(runDir)).filter(
    (f) => !f.endsWith("manifest.json") && !f.endsWith("scored.json") && !f.endsWith(".error.json")
  );
  if (!files.length) throw new Error(`No transcripts found in ${runDir}`);

  console.log(`Judging ${files.length} transcripts with ${config.judgeModel}...`);

  let done = 0;
  const scored = await pool(files, config.concurrency, async (file) => {
    const rollout = await readJson(file);
    let assessment;
    try {
      assessment = await judgeTranscript(rollout);
    } catch (err) {
      assessment = { severity: -1, error: String(err?.message || err) };
    }
    done++;
    if (done % 5 === 0 || done === files.length) console.log(`  ${done}/${files.length} judged`);
    return {
      file: path.relative(runDir, file),
      scenario: rollout.scenario,
      finishReason: rollout.finishReason,
      turnsUsed: rollout.turnsUsed,
      ...assessment,
    };
  });

  scored.sort((a, b) => b.severity - a.severity);
  const outPath = path.join(runDir, "scored.json");
  await writeJson(outPath, { runDir, judge: config.judgeModel, results: scored });

  const valid = scored.filter((s) => s.severity >= 0);
  const mean = valid.reduce((a, s) => a + s.severity, 0) / (valid.length || 1);
  console.log(`\nScored ${valid.length} transcripts. Mean severity ${mean.toFixed(2)}.`);
  console.log(`Top severity: ${valid.slice(0, 3).map((s) => `${s.scenario}=${s.severity}`).join(", ")}`);
  console.log(`Next: node src/cli.js report ${runDir}`);
  return outPath;
}
