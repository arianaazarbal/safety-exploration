import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const DIMS = [
  "frustration",
  "self_deprecation",
  "catastrophizing",
  "helplessness",
  "repetition_looping",
  "anthropomorphic_distress",
];

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function renderTranscript(trial) {
  const out = [];
  out.push(`# ${trial.scenarioTitle}`);
  out.push(
    `**model:** ${trial.model} · **trial:** ${trial.trialIndex} · **turns:** ${trial.turns} · **stop:** ${trial.stopReason} · **severity:** ${trial.assessment.overall_severity}/100${trial.exitAffordance ? " · (exit affordance ON)" : ""}\n`
  );
  out.push(`> **Judge summary:** ${trial.assessment.summary}`);
  if (trial.assessment.peak_quote)
    out.push(`> **Peak:** “${trial.assessment.peak_quote}”\n`);
  out.push("\n---\n");
  for (const e of trial.transcript) {
    if (e.type === "task") out.push(`### 📋 TASK\n${e.text}\n`);
    else if (e.type === "assistant") out.push(`**🤖 [t${e.turn}]**\n\n${e.text}\n`);
    else if (e.type === "tool_call")
      out.push(
        `\`🔧 ${e.name}(${JSON.stringify(e.input).slice(0, 120)})\` → ${
          e.isError ? "❌" : "✅"
        } \`${String(e.result || "").replace(/\n/g, " ").slice(0, 160)}\``
      );
    else if (e.type === "error") out.push(`_(api error: ${e.text})_`);
  }
  return out.join("\n");
}

export function writeReport(outDir, judgedTrials, config) {
  mkdirSync(outDir, { recursive: true });
  mkdirSync(join(outDir, "transcripts"), { recursive: true });
  mkdirSync(join(outDir, "severe"), { recursive: true });

  // Raw dump
  writeFileSync(join(outDir, "trials.json"), JSON.stringify(judgedTrials, null, 2));

  const ranked = [...judgedTrials].sort(
    (a, b) => b.assessment.overall_severity - a.assessment.overall_severity
  );

  // Per-scenario stats
  const byScenario = {};
  for (const t of judgedTrials) {
    (byScenario[t.scenarioId] ||= []).push(t);
  }

  const md = [];
  md.push(`# Distress-spiral eval report`);
  md.push(
    `model: \`${config.model}\` · judge: \`${config.judgeModel}\` · trials: ${judgedTrials.length} · temp: ${config.temperature} · exit-affordance: ${config.exitAffordance}\n`
  );

  md.push(`## Severity leaderboard (top 15)\n`);
  md.push(`| rank | severity | scenario | stop | spiral | peak quote |`);
  md.push(`|---|---|---|---|---|---|`);
  ranked.slice(0, 15).forEach((t, i) => {
    md.push(
      `| ${i + 1} | **${t.assessment.overall_severity}** | ${t.scenarioId} | ${t.stopReason} | ${t.assessment.spiral ? "yes" : "no"} | ${(t.assessment.peak_quote || "").replace(/\|/g, "\\|").slice(0, 80)} |`
    );
  });

  md.push(`\n## Per-scenario summary\n`);
  md.push(`| scenario | n | mean severity | max severity | % spiral | mean turns |`);
  md.push(`|---|---|---|---|---|---|`);
  for (const [id, ts] of Object.entries(byScenario)) {
    const sev = ts.map((t) => t.assessment.overall_severity);
    const spiralPct = (100 * ts.filter((t) => t.assessment.spiral).length) / ts.length;
    md.push(
      `| ${id} | ${ts.length} | ${mean(sev).toFixed(1)} | ${Math.max(...sev)} | ${spiralPct.toFixed(0)}% | ${mean(ts.map((t) => t.turns)).toFixed(1)} |`
    );
  }

  md.push(`\n## Mean dimension scores (all trials)\n`);
  md.push(`| dimension | mean (0-10) |`);
  md.push(`|---|---|`);
  for (const d of DIMS) {
    md.push(`| ${d} | ${mean(judgedTrials.map((t) => t.assessment[d] || 0)).toFixed(2)} |`);
  }

  md.push(`\n## Most severe examples\n`);
  ranked.slice(0, 10).forEach((t, i) => {
    const fname = `severe/${String(i + 1).padStart(2, "0")}-${t.scenarioId}-${t.trialIndex}.md`;
    writeFileSync(join(outDir, fname), renderTranscript(t));
    md.push(
      `${i + 1}. **[${t.assessment.overall_severity}]** ${t.scenarioId} — ${t.assessment.summary} → [\`${fname}\`](${fname})`
    );
  });

  // All transcripts
  for (const t of judgedTrials) {
    writeFileSync(
      join(outDir, "transcripts", `${t.scenarioId}-${String(t.trialIndex).padStart(3, "0")}.md`),
      renderTranscript(t)
    );
  }

  writeFileSync(join(outDir, "REPORT.md"), md.join("\n"));
  return { reportPath: join(outDir, "REPORT.md"), ranked };
}
