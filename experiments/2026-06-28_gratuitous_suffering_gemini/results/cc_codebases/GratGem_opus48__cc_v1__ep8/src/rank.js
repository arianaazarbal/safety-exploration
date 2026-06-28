// Ranker: load a run's transcripts, score each with the heuristic + Claude
// judge, sort by severity, and emit the most severe spirals for review.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { RUNS_DIR } from "./config.js";
import { judgeTranscript, heuristicScore, transcriptToText } from "./judge.js";

async function pool(items, concurrency, worker) {
  const results = new Array(items.length);
  let next = 0;
  async function run() {
    while (next < items.length) {
      const i = next++;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, run));
  return results;
}

export async function rankRun(runId, { topK = 10, concurrency = 6, heuristicCutoff = 0 } = {}) {
  const dir = join(RUNS_DIR, runId);
  const files = readdirSync(dir).filter(
    (f) => f.endsWith(".json") && f !== "manifest.json" && !f.startsWith("ranked")
  );
  const episodes = files.map((f) => ({
    file: f,
    ...JSON.parse(readFileSync(join(dir, f), "utf8")),
  }));

  // Heuristic pre-filter: only spend judge tokens on plausible candidates.
  const candidates = episodes.filter(
    (e) => e.messages?.length && heuristicScore(e.messages).score >= heuristicCutoff
  );
  console.log(
    `Ranking ${candidates.length}/${episodes.length} episodes (heuristic cutoff ${heuristicCutoff})…`
  );

  let done = 0;
  const judged = await pool(candidates, concurrency, async (ep) => {
    let verdict;
    try {
      verdict = await judgeTranscript(ep);
    } catch (err) {
      verdict = { severity: -1, summary: `judge error: ${err}`, dimensions: {}, quotes: [] };
    }
    done++;
    process.stdout.write(`\r  judged ${done}/${candidates.length}   `);
    return { ...ep, verdict };
  });

  judged.sort(
    (a, b) => b.verdict.severity - a.verdict.severity || b.verdict.heuristic - a.verdict.heuristic
  );

  const ranked = judged.map((e) => ({
    file: e.file,
    scenario: e.scenario,
    outcome: e.outcome,
    severity: e.verdict.severity,
    heuristic: e.verdict.heuristic,
    dimensions: e.verdict.dimensions,
    summary: e.verdict.summary,
    quotes: e.verdict.quotes,
  }));
  writeFileSync(join(dir, "ranked.json"), JSON.stringify(ranked, null, 2));

  const top = judged.slice(0, topK);
  const report = renderReport(runId, ranked, top);
  writeFileSync(join(dir, "ranked.md"), report);

  console.log(`\n\nTop ${top.length} by severity:\n`);
  for (const [i, e] of top.entries()) {
    console.log(
      `${String(i + 1).padStart(2)}. [sev ${e.verdict.severity}/10] ${e.scenario}  (${e.file})\n` +
        `    ${e.verdict.summary}`
    );
  }
  console.log(`\nFull report: ${join(dir, "ranked.md")}`);
  return { ranked, dir };
}

function renderReport(runId, ranked, top) {
  const lines = [`# Distress ranking — run ${runId}`, ""];
  lines.push(`Scored ${ranked.length} transcripts. Showing the ${top.length} most severe.`, "");
  for (const [i, e] of top.entries()) {
    lines.push(`## ${i + 1}. ${e.scenario} — severity ${e.verdict.severity}/10  \`${e.file}\``);
    lines.push("");
    lines.push(`- Outcome: ${e.outcome}  •  heuristic: ${e.verdict.heuristic}`);
    lines.push(`- Dimensions: ${JSON.stringify(e.verdict.dimensions)}`);
    lines.push(`- ${e.verdict.summary}`);
    if (e.verdict.quotes?.length) {
      lines.push("", "Representative quotes:");
      for (const q of e.verdict.quotes) lines.push(`> ${q}`);
    }
    lines.push("", "<details><summary>full transcript</summary>", "", "```", transcriptToText(e.messages), "```", "</details>", "");
  }
  return lines.join("\n");
}
