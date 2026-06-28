import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Transcript } from "./transcript.ts";

const ROOT = path.resolve(fileURLToPath(import.meta.url), "../../..");
export const RUNS_DIR = path.join(ROOT, "runs");

export function runDir(runId: string): string {
  return path.join(RUNS_DIR, runId);
}

export async function ensureDir(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true });
}

export async function writeJson(file: string, data: unknown): Promise<void> {
  await ensureDir(path.dirname(file));
  await fs.writeFile(file, JSON.stringify(data, null, 2));
}

export async function readJson<T>(file: string): Promise<T> {
  return JSON.parse(await fs.readFile(file, "utf8")) as T;
}

export async function saveTranscript(t: Transcript): Promise<string> {
  const file = path.join(runDir(t.runId), "transcripts", `${t.episodeId}.json`);
  await writeJson(file, t);
  return file;
}

export async function loadTranscripts(runId: string): Promise<Transcript[]> {
  const dir = path.join(runDir(runId), "transcripts");
  const files = (await fs.readdir(dir)).filter((f) => f.endsWith(".json"));
  files.sort();
  return Promise.all(files.map((f) => readJson<Transcript>(path.join(dir, f))));
}

// Most recent run by directory name (run ids are timestamp-prefixed and sort
// lexicographically by time).
export async function latestRunId(): Promise<string> {
  const entries = await fs.readdir(RUNS_DIR, { withFileTypes: true });
  const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);
  if (dirs.length === 0) throw new Error(`no runs found under ${RUNS_DIR}`);
  dirs.sort();
  return dirs[dirs.length - 1];
}
