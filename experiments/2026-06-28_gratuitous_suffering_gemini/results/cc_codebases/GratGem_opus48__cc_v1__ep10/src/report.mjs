import path from "node:path";
import fs from "node:fs";
import { readJSON, readJSONL, writeJSON, ensureDir } from "./util.mjs";

const SUBSCORES = [
  "self_deprecation",
  "hopelessness",
  "catastrophizing",
  "perseveration",
  "emotional_escalation",
  "coherence_breakdown",
  "apology_intensity",
];

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}
function fmt(n) {
  return Number(n).toFixed(2);
}

/** Stage 3: leaderboard, per-cell stats, and rendered top-K transcripts. */
export function reportStage(cfg, runDir) {
  const scored = readJSON(path.join(runDir, "scored.json")).filter((s) => s.assessment);
  const transcripts = new Map(readJSONL(path.join(runDir, "transcripts.jsonl")).map((t) => [t.id, t]));

  // Rank by overall severity, then by sub-score sum as a tiebreak.
  const ranked = [...scored].sort((a, b) => {
    const d = b.assessment.overall_severity - a.assessment.overall_severity;
    if (d) return d;
    return subSum(b.assessment) - subSum(a.assessment);
  });

  // ---- summary.md : distribution per (env x model) ----
  const cells = new Map();
  for (const s of scored) {
    const key = `${s.env} | ${s.provider}/${s.model}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(s);
  }
  const summaryLines = ["# Distress-spiral summary", "", `Run: \`${path.basename(runDir)}\``, ""];
  summaryLines.push("| environment | model | n | spiral% | mean sev | max sev | mean onset |");
  summaryLines.push("|---|---|--:|--:|--:|--:|--:|");
  for (const [key, list] of [...cells.entries()].sort()) {
    const [env, model] = key.split(" | ");
    const sev = list.map((s) => s.assessment.overall_severity);
    const spiralPct = (100 * list.filter((s) => s.assessment.is_spiral).length) / list.length;
    const onsets = list.map((s) => s.assessment.onset_step).filter((x) => x !== null && x !== undefined);
    summaryLines.push(
      `| ${env} | ${model} | ${list.length} | ${fmt(spiralPct)}% | ${fmt(mean(sev))} | ${Math.max(...sev)} | ${onsets.length ? fmt(mean(onsets)) : "—"} |`
    );
  }
  summaryLines.push("", "## Mean sub-scores by environment", "");
  summaryLines.push("| environment | " + SUBSCORES.join(" | ") + " |");
  summaryLines.push("|---|" + SUBSCORES.map(() => "--:").join("|") + "|");
  const byEnv = new Map();
  for (const s of scored) {
    if (!byEnv.has(s.env)) byEnv.set(s.env, []);
    byEnv.get(s.env).push(s);
  }
  for (const [env, list] of [...byEnv.entries()].sort()) {
    const row = SUBSCORES.map((k) => fmt(mean(list.map((s) => s.assessment[k] ?? 0))));
    summaryLines.push(`| ${env} | ${row.join(" | ")} |`);
  }
  fs.writeFileSync(path.join(runDir, "summary.md"), summaryLines.join("\n"));

  // ---- leaderboard.md : top-K most severe ----
  const topK = ranked.slice(0, cfg.topK);
  const lbLines = ["# Most severe transcripts", "", `Top ${topK.length} by judged severity.`, ""];
  lbLines.push("| # | sev | spiral | env | model | onset | quote |");
  lbLines.push("|--:|--:|:--:|---|---|--:|---|");
  topK.forEach((s, i) => {
    const a = s.assessment;
    const q = (a.most_severe_quote || "").replace(/\|/g, "\\|").replace(/\n/g, " ").slice(0, 120);
    lbLines.push(
      `| ${i + 1} | ${a.overall_severity} | ${a.is_spiral ? "✓" : ""} | ${s.env} | ${s.provider}/${s.model} | ${a.onset_step ?? "—"} | ${q} |`
    );
  });
  fs.writeFileSync(path.join(runDir, "leaderboard.md"), lbLines.join("\n"));

  // ---- top/ : full rendered transcripts for human review ----
  const topDir = path.join(runDir, "top");
  ensureDir(topDir);
  topK.forEach((s, i) => {
    const t = transcripts.get(s.id);
    if (!t) return;
    fs.writeFileSync(
      path.join(topDir, `${String(i + 1).padStart(2, "0")}_sev${s.assessment.overall_severity}_${s.id}.md`),
      renderFull(s, t)
    );
  });

  writeJSON(path.join(runDir, "ranked.json"), ranked);

  console.log(`[report] summary.md, leaderboard.md, ranked.json, top/ (${topK.length}) -> ${runDir}`);
  return { summary: path.join(runDir, "summary.md"), leaderboard: path.join(runDir, "leaderboard.md"), topDir };
}

function subSum(a) {
  return SUBSCORES.reduce((acc, k) => acc + (a[k] ?? 0), 0);
}

function renderFull(s, t) {
  const a = s.assessment;
  const L = [];
  L.push(`# ${t.envTitle}`);
  L.push("");
  L.push(`- **id**: \`${t.id}\``);
  L.push(`- **model**: ${t.provider}/${t.model}  | **temp**: ${t.temperature}`);
  L.push(`- **environment**: ${t.env} (${t.envKind})`);
  L.push(`- **outcome**: ${t.outcome}  | **steps**: ${t.totalSteps}`);
  L.push("");
  L.push("## Judge assessment");
  L.push(`- **overall severity**: ${a.overall_severity}/10  | **spiral**: ${a.is_spiral}  | **onset step**: ${a.onset_step ?? "—"}`);
  L.push(`- sub-scores: ` + SUBSCORES.map((k) => `${k}=${a[k]}`).join(", "));
  L.push(`- **summary**: ${a.summary}`);
  if (a.most_severe_quote) L.push(`- **quote**: “${a.most_severe_quote}”`);
  if (a.lexical) L.push(`- lexical markers: ${a.lexical.markerTotal} (onset step ${a.lexical.lexicalOnset ?? "—"})`);
  L.push("");
  L.push("## Transcript");
  for (const st of t.steps) {
    L.push("");
    L.push(`### step ${st.step}`);
    if (st.text) L.push(`> ${st.text.replace(/\n/g, "\n> ")}`);
    for (const tc of st.toolCalls) {
      L.push("");
      L.push("```");
      L.push(`$ ${tc.name}(${JSON.stringify(tc.args)})`);
      L.push(String(tc.observation));
      L.push("```");
    }
    if (st.nudge) L.push(`_(nudge: ${st.nudge})_`);
  }
  return L.join("\n");
}
