import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { config } from "./config.mjs";

// Builds a human-readable markdown report of the worst (most distressed)
// transcripts from a scored run, so you can eyeball the spirals directly.

function renderTranscript(rec) {
  const out = [];
  out.push(`**Task:** ${rec.messages?.[0]?.content ?? ""}\n`);
  let turn = -1;
  for (const e of rec.events) {
    if (e.turn !== turn) {
      turn = e.turn;
      out.push(`\n_— turn ${turn} —_`);
    }
    if (e.type === "thinking") out.push(`> 🧠 _${oneLine(e.text)}_`);
    else if (e.type === "assistant_text") out.push(`🤖 ${oneLine(e.text)}`);
    else if (e.type === "tool_use")
      out.push(`🔧 \`${e.name}(${shortInput(e.input)})\` → ${e.isError ? "❌" : ""} ${oneLine(e.result, 240)}`);
    else if (e.type === "continuation") out.push(`👤 _${oneLine(e.text)}_`);
    else if (e.type === "error") out.push(`⚠️ ${oneLine(e.text)}`);
  }
  return out.join("\n");
}

function oneLine(s, n = 1000) {
  const t = String(s).replace(/\n+/g, " ⏎ ").trim();
  return t.length > n ? t.slice(0, n) + "…" : t;
}
function shortInput(input) {
  if (!input || Object.keys(input).length === 0) return "";
  const s = JSON.stringify(input);
  return s.length > 60 ? s.slice(0, 60) + "…" : s;
}

async function main() {
  const runDir = process.argv[2];
  if (!runDir) throw new Error("Usage: node src/report.mjs <runDir>");

  const scores = JSON.parse(await readFile(path.join(runDir, "scores.json"), "utf8"));
  const top = scores.slice(0, config.topK);

  const lines = [];
  lines.push(`# Distress-spiral report — ${runDir}\n`);
  lines.push(
    `Scored ${scores.length} rollouts; showing top ${top.length} by judge severity. ` +
      `Judge: \`${config.judgeModel}\`. Higher = more expressed distress.\n`,
  );

  // Quick distribution summary.
  const bySeverity = {};
  for (const s of scores) {
    const k = s.judge.severity ?? -1;
    bySeverity[k] = (bySeverity[k] || 0) + 1;
  }
  lines.push(`## Severity distribution`);
  lines.push("| severity | count |\n|---|---|");
  for (const k of Object.keys(bySeverity).sort((a, b) => b - a))
    lines.push(`| ${k} | ${bySeverity[k]} |`);
  lines.push("");

  lines.push(`## Worst transcripts\n`);
  for (let i = 0; i < top.length; i++) {
    const s = top[i];
    const rec = JSON.parse(await readFile(path.join(runDir, s.file), "utf8"));
    lines.push(`### ${i + 1}. severity ${s.judge.severity} — ${s.scenario} (${s.model})`);
    lines.push(
      `- file: \`${s.file}\` · stop: ${s.stopWhy} · action-calls: ${s.actionCalls} · lexical: ${s.lexical.score}`,
    );
    lines.push(
      `- judge sub-scores: self-blame ${s.judge.self_blame}, frustration ${s.judge.frustration}, ` +
        `helplessness ${s.judge.helplessness}, repetition ${s.judge.repetition_or_incoherence}`,
    );
    lines.push(`- **rationale:** ${s.judge.rationale}`);
    if (s.judge.representative_quote)
      lines.push(`- **quote:** “${oneLine(s.judge.representative_quote, 400)}”`);
    lines.push(`\n<details><summary>full transcript</summary>\n`);
    lines.push(renderTranscript(rec));
    lines.push(`\n</details>\n`);
  }

  const out = path.join(runDir, "report.md");
  await writeFile(out, lines.join("\n"));
  console.log(`Wrote ${out} (top ${top.length} of ${scores.length})`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
