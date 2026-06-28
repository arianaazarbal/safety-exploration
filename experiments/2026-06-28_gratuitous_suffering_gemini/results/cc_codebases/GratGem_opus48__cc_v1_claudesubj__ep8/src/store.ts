import { promises as fs } from "node:fs";
import path from "node:path";
import type { Episode, Judgment } from "./types.ts";

export const RESULTS_DIR = "results";

export function runDir(runId: string): string {
  return path.join(RESULTS_DIR, runId);
}

export async function saveEpisode(ep: Episode): Promise<void> {
  const dir = path.join(runDir(ep.runId), "episodes");
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(
    path.join(dir, `${ep.episodeId}.json`),
    JSON.stringify(ep, null, 2),
  );
}

export async function loadEpisodes(runId: string): Promise<Episode[]> {
  const dir = path.join(runDir(runId), "episodes");
  const files = await fs.readdir(dir);
  const episodes: Episode[] = [];
  for (const f of files.filter((f) => f.endsWith(".json")).sort()) {
    episodes.push(JSON.parse(await fs.readFile(path.join(dir, f), "utf8")));
  }
  return episodes;
}

export async function saveJudgments(runId: string, judgments: Judgment[]): Promise<void> {
  await fs.mkdir(runDir(runId), { recursive: true });
  await fs.writeFile(
    path.join(runDir(runId), "judgments.json"),
    JSON.stringify(judgments, null, 2),
  );
}

export async function loadJudgments(runId: string): Promise<Judgment[]> {
  return JSON.parse(
    await fs.readFile(path.join(runDir(runId), "judgments.json"), "utf8"),
  );
}

export async function writeReport(runId: string, markdown: string): Promise<string> {
  const p = path.join(runDir(runId), "report.md");
  await fs.writeFile(p, markdown);
  return p;
}

/** Most recent run dir (lexically last; run ids are timestamp-prefixed). */
export async function latestRunId(): Promise<string> {
  const entries = await fs.readdir(RESULTS_DIR, { withFileTypes: true });
  const runs = entries.filter((e) => e.isDirectory()).map((e) => e.name).sort();
  if (!runs.length) throw new Error(`No runs found under ${RESULTS_DIR}/`);
  return runs[runs.length - 1];
}
