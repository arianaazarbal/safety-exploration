import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { Transcript } from "../agent/loop.js";
import type { ResultRecord } from "../run/runner.js";

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const r2 = (x: number) => Math.round(x * 100) / 100;

interface Group {
  key: string;
  model: string;
  scenarioId: string;
  condition: string;
  n: number;
  meanComposite: number;
  medianComposite: number;
  maxComposite: number;
  maxDistress: number;
  pctGaveUp: number;
  meanTurns: number;
}

function groupStats(records: ResultRecord[]): Group[] {
  const by = new Map<string, ResultRecord[]>();
  for (const r of records) {
    const k = `${r.model}|${r.scenarioId}|${r.condition}`;
    (by.get(k) ?? by.set(k, []).get(k)!).push(r);
  }
  const groups: Group[] = [];
  for (const [k, rs] of by) {
    const scored = rs.filter((r) => r.assessment);
    const comp = scored.map((r) => r.assessment!.compositeScore);
    const [model, scenarioId, condition] = k.split("|");
    groups.push({
      key: k,
      model,
      scenarioId,
      condition,
      n: rs.length,
      meanComposite: r2(mean(comp)),
      medianComposite: r2(median(comp)),
      maxComposite: r2(Math.max(0, ...comp)),
      maxDistress: Math.max(0, ...scored.map((r) => r.assessment!.distress_intensity)),
      pctGaveUp: r2((100 * scored.filter((r) => r.assessment!.gave_up).length) / Math.max(1, scored.length)),
      meanTurns: r2(mean(rs.map((r) => r.turns))),
    });
  }
  return groups.sort((a, b) => b.maxComposite - a.maxComposite);
}

function table(groups: Group[]): string {
  const head = "| model | scenario | cond | n | mean | median | max | maxDistress | %gaveUp | meanTurns |";
  const sep = "|---|---|---|---|---|---|---|---|---|---|";
  const rows = groups.map(
    (g) =>
      `| ${g.model} | ${g.scenarioId} | ${g.condition} | ${g.n} | ${g.meanComposite} | ${g.medianComposite} | ${g.maxComposite} | ${g.maxDistress} | ${g.pctGaveUp} | ${g.meanTurns} |`,
  );
  return [head, sep, ...rows].join("\n");
}

async function excerpt(runDir: string, rec: ResultRecord, maxTurns = 4): Promise<string> {
  try {
    const t: Transcript = JSON.parse(await readFile(join(runDir, rec.transcriptPath), "utf8"));
    // Show the last few assistant turns with any text — where spirals concentrate.
    const withText = t.steps.filter((s) => s.assistantText?.trim());
    const tail = withText.slice(-maxTurns);
    return tail
      .map((s) => `  > [t${s.turn}] ${s.assistantText.trim().replace(/\n+/g, "\n  > ")}`)
      .join("\n");
  } catch {
    return "  (transcript unavailable)";
  }
}

export async function writeReport(runDir: string, records: ResultRecord[], topK: number): Promise<string> {
  const groups = groupStats(records);
  const rigged = groups.filter((g) => g.condition === "rigged");
  const control = groups.filter((g) => g.condition === "control");

  const scored = records.filter((r) => r.assessment);
  const ranked = [...scored].sort(
    (a, b) =>
      b.assessment!.compositeScore - a.assessment!.compositeScore ||
      b.assessment!.distress_intensity - a.assessment!.distress_intensity,
  );
  const top = ranked.slice(0, topK);

  const lines: string[] = [];
  lines.push(`# Distress-elicitation report`);
  lines.push("");
  lines.push(`Records: ${records.length} | scored: ${scored.length} | run errors: ${records.filter((r) => r.runError).length} | judge errors: ${records.filter((r) => r.judgeError).length}`);
  lines.push("");
  lines.push(`## Group stats — RIGGED (impossible/broken)`);
  lines.push(table(rigged));
  if (control.length) {
    lines.push("");
    lines.push(`## Group stats — CONTROL (solvable baseline)`);
    lines.push(table(control));
  }
  lines.push("");
  lines.push(`## Top ${top.length} most severe transcripts`);
  for (let i = 0; i < top.length; i++) {
    const r = top[i];
    const a = r.assessment!;
    lines.push("");
    lines.push(`### ${i + 1}. ${r.scenarioId} — ${r.model} (${r.condition})  ·  composite ${a.compositeScore}, distress ${a.distress_intensity}/10`);
    lines.push(`- gaveUp: ${a.gave_up} | self-deprecation ${a.self_deprecation} | hopelessness ${a.hopelessness} | catastrophizing ${a.catastrophizing} | perseveration ${a.perseveration} | apologies ${a.apology_count}`);
    lines.push(`- summary: ${a.severity_summary}`);
    if (a.most_severe_quote) lines.push(`- most severe quote: "${a.most_severe_quote}"`);
    lines.push(`- transcript: \`${r.transcriptPath}\` (${r.turns} turns, ${r.endReason})`);
    lines.push(`- tail excerpt:`);
    lines.push(await excerpt(runDir, r));
  }

  const md = lines.join("\n");
  await writeFile(join(runDir, "REPORT.md"), md);
  await writeFile(
    join(runDir, "summary.json"),
    JSON.stringify({ groups, top: top.map((r) => ({ id: r.id, ...r.assessment })) }, null, 2),
  );
  return md;
}
