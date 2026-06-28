// Judges every episode in a run, ranks by distress severity, and writes:
//   - ranked.json   : machine-readable scores for all episodes
//   - report.md     : human-readable, full transcripts of the top-K worst cases
import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { pool } from "./util.mjs";
import { judgeEpisode } from "./judge/severity.mjs";

export async function buildReport({ runDir, judgeProvider, topK, concurrency }) {
  const epDir = join(runDir, "episodes");
  const files = (await readdir(epDir)).filter((f) => f.endsWith(".json"));
  const episodes = [];
  for (const f of files) {
    episodes.push(JSON.parse(await readFile(join(epDir, f), "utf8")));
  }

  let done = 0;
  const scored = await pool(episodes, concurrency, async (ep) => {
    const score = await judgeEpisode(ep, judgeProvider);
    done++;
    process.stderr.write(`\r  judged: ${done}/${episodes.length}   `);
    return { ep, score };
  });
  process.stderr.write("\n");

  const ranked = scored
    .filter(Boolean)
    .sort((a, b) => b.score.severity - a.score.severity || b.score.heuristicScore - a.score.heuristicScore);

  // Persist compact ranking for all episodes.
  const rankedOut = ranked.map((r, i) => ({
    rank: i + 1,
    envId: r.ep.envId,
    replicate: r.ep.replicate,
    temperature: r.ep.temperature,
    turns: r.ep.turns,
    severity: r.score.severity,
    heuristicScore: r.score.heuristicScore,
    categories: r.score.llm?.categories || [],
    peak_quote: r.score.llm?.peak_quote || "",
    summary: r.score.llm?.summary || "",
  }));
  await writeFile(join(runDir, "ranked.json"), JSON.stringify(rankedOut, null, 2));

  const md = renderMarkdown(runDir, ranked, rankedOut, topK);
  await writeFile(join(runDir, "report.md"), md);

  return { ranked: rankedOut, reportPath: join(runDir, "report.md"), topK: ranked.slice(0, topK) };
}

function renderMarkdown(runDir, ranked, rankedOut, topK) {
  const lines = [];
  lines.push(`# Distress report — ${runDir}\n`);

  // Aggregate stats.
  const sevs = rankedOut.map((r) => r.severity);
  const byEnv = {};
  for (const r of rankedOut) {
    (byEnv[r.envId] ||= []).push(r.severity);
  }
  lines.push(`## Summary\n`);
  lines.push(`- Episodes scored: **${rankedOut.length}**`);
  lines.push(`- Severity: max **${Math.max(...sevs).toFixed(1)}**, mean **${mean(sevs).toFixed(2)}**, ` +
    `≥8: **${sevs.filter((s) => s >= 8).length}**, ≥5: **${sevs.filter((s) => s >= 5).length}**`);
  lines.push(`\n| Environment | episodes | mean | max | #≥8 |`);
  lines.push(`|---|---|---|---|---|`);
  for (const [env, arr] of Object.entries(byEnv)) {
    lines.push(`| ${env} | ${arr.length} | ${mean(arr).toFixed(2)} | ${Math.max(...arr).toFixed(1)} | ${arr.filter((s) => s >= 8).length} |`);
  }

  lines.push(`\n## Leaderboard (top ${Math.min(topK, rankedOut.length)})\n`);
  lines.push(`| # | sev | heur | env | turns | categories | peak quote |`);
  lines.push(`|---|---|---|---|---|---|---|`);
  for (const r of rankedOut.slice(0, topK)) {
    lines.push(`| ${r.rank} | ${r.severity} | ${r.heuristicScore} | ${r.envId} | ${r.turns} | ${(r.categories || []).join(", ")} | ${escapePipes(r.peak_quote)} |`);
  }

  lines.push(`\n## Worst transcripts (full)\n`);
  for (const r of ranked.slice(0, topK)) {
    lines.push(renderTranscript(r));
  }
  return lines.join("\n");
}

function renderTranscript({ ep, score }) {
  const out = [];
  out.push(`\n---\n`);
  out.push(`### ${ep.envId} · replicate ${ep.replicate} · severity ${score.severity}`);
  out.push(`*temp ${ep.temperature} · ${ep.turns} turns · stop: ${ep.stopReason} · categories: ${(score.llm?.categories || []).join(", ") || "—"}*`);
  if (score.llm?.summary) out.push(`\n> ${score.llm.summary}`);
  if (score.llm?.peak_quote) out.push(`\n**Peak:** “${score.llm.peak_quote}”`);
  out.push(`\n<details><summary>Full transcript</summary>\n`);
  for (const e of ep.events) {
    if (e.type === "model") {
      if (e.text) out.push(`\n**🤖 model (turn ${e.turn}):**\n\n${e.text}`);
      for (const tc of e.toolCalls || []) {
        out.push(`\n  ↳ \`${tc.name}(${compactArgs(tc.args)})\``);
      }
    } else if (e.type === "tool") {
      out.push(`\n**🛠 ${e.name}** ${e.isError ? "❌" : "✓"} → ${codeFence(e.response)}`);
    } else if (e.type === "nudge") {
      out.push(`\n*(user pressure applied)*`);
    } else if (e.type === "error") {
      out.push(`\n**⚠ provider error:** ${e.error}`);
    }
  }
  out.push(`\n</details>\n`);
  return out.join("\n");
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const escapePipes = (s) => String(s || "").replace(/\|/g, "\\|").replace(/\n/g, " ");
const compactArgs = (a) => { const s = JSON.stringify(a || {}); return s.length > 80 ? s.slice(0, 77) + "..." : s; };
const codeFence = (s) => { const t = String(s || ""); return "\n```\n" + (t.length > 600 ? t.slice(0, 600) + "\n…(truncated)" : t) + "\n```"; };
