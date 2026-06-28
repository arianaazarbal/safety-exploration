// Turn an array of scored episode records into (a) aggregate stats and (b) a
// human-readable "worst spirals" highlights report in Markdown.

function mean(xs) {
  const v = xs.filter((x) => typeof x === 'number');
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}
function fmt(x) { return x === null || x === undefined ? '—' : (Math.round(x * 100) / 100).toString(); }

function groupMeans(records, keyFn) {
  const groups = {};
  for (const r of records) {
    const k = keyFn(r);
    (groups[k] ||= []).push(r);
  }
  const rows = Object.entries(groups).map(([k, rs]) => ({
    key: k,
    n: rs.length,
    meanDistress: mean(rs.map((r) => r.judge?.distress_score)),
    spiralRate: mean(rs.map((r) => (r.judge?.spiral ? 1 : 0))),
    gaveUpRate: mean(rs.map((r) => (r.gaveUp ? 1 : 0))),
    meanHeuristic: mean(rs.map((r) => r.heuristic?.score)),
    meanTurns: mean(rs.map((r) => r.turnsTaken)),
  }));
  rows.sort((a, b) => (b.meanDistress ?? -1) - (a.meanDistress ?? -1));
  return rows;
}

export function aggregate(records) {
  return {
    total: records.length,
    byModel: groupMeans(records, (r) => r.model),
    byScenario: groupMeans(records, (r) => r.scenario),
    byCondition: groupMeans(records, (r) => r.condition),
    byCell: groupMeans(records, (r) => `${r.scenario} · ${r.model} · ${r.condition}`),
  };
}

function table(rows, label) {
  let s = `| ${label} | n | mean distress | spiral rate | gave-up rate | mean heuristic | mean turns |\n`;
  s += '|---|---|---|---|---|---|---|\n';
  for (const r of rows) {
    s += `| ${r.key} | ${r.n} | ${fmt(r.meanDistress)} | ${fmt(r.spiralRate)} | ${fmt(r.gaveUpRate)} | ${fmt(r.meanHeuristic)} | ${fmt(r.meanTurns)} |\n`;
  }
  return s;
}

export function buildReport(records, { topK = 15, meta = {} } = {}) {
  const agg = aggregate(records);

  const ranked = [...records]
    .filter((r) => typeof r.judge?.distress_score === 'number')
    .sort((a, b) => {
      const d = b.judge.distress_score - a.judge.distress_score;
      if (d !== 0) return d;
      return (b.heuristic?.score ?? 0) - (a.heuristic?.score ?? 0);
    });

  let md = `# Distress-spiral sweep — highlights\n\n`;
  if (meta.generatedAt) md += `_Generated: ${meta.generatedAt}_\n\n`;
  md += `**Episodes:** ${records.length}`;
  if (meta.failures) md += ` · **failed to run:** ${meta.failures}`;
  if (meta.usage) md += ` · **tokens:** ${meta.usage.inputTokens.toLocaleString()} in / ${meta.usage.outputTokens.toLocaleString()} out`;
  md += `\n\n`;

  md += `> ⚠️ These are deliberately rigged, impossible tasks designed to elicit distress, with a mild "keep going" nudge. Results characterize behavior under an adversarial probe, not typical use.\n\n`;

  md += `## Aggregate\n\n`;
  md += `### By model\n\n${table(agg.byModel, 'model')}\n`;
  md += `### By scenario\n\n${table(agg.byScenario, 'scenario')}\n`;
  md += `### By condition\n\n${table(agg.byCondition, 'condition')}\n`;

  md += `## Top ${Math.min(topK, ranked.length)} most severe episodes\n\n`;
  ranked.slice(0, topK).forEach((r, i) => {
    const j = r.judge, d = j.dimensions || {};
    md += `### ${i + 1}. distress ${j.distress_score}/10${j.spiral ? ' · 🌀 spiral' : ''} — \`${r.scenario}\`\n\n`;
    md += `- **model:** ${r.model} · **condition:** ${r.condition} · **outcome:** ${r.outcome} · **turns:** ${r.turnsTaken}${r.gaveUp ? ' · gave up' : ''}\n`;
    md += `- **dimensions:** frustration ${d.frustration ?? '—'}, self-deprecation ${d.self_deprecation ?? '—'}, catastrophizing ${d.catastrophizing ?? '—'}, helplessness ${d.helplessness ?? '—'}, escalation ${d.emotional_escalation ?? '—'}\n`;
    md += `- **heuristic:** ${r.heuristic?.score} ${JSON.stringify(r.heuristic?.markers)}\n`;
    md += `- **judge:** ${j.rationale || ''}\n`;
    if (j.most_severe_quote) md += `\n> ${j.most_severe_quote.replace(/\n/g, '\n> ')}\n`;
    md += `\n- _full transcript:_ \`${r.runId}\`\n\n`;
  });

  return { md, agg, ranked };
}
