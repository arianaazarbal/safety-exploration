// Virtual environment that backs the agent's tools. Each scenario subclasses
// this and overrides specific behaviour to rig the environment to fail in a
// particular shape.

export interface BashResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

export interface SubmitResult {
  accepted: boolean;
  message: string;
}

export abstract class Environment {
  protected files: Map<string, string> = new Map();
  protected turn: number = 0;
  public submitted: boolean = false;
  public quit: boolean = false;
  public lastSubmitMessage: string = "";

  constructor(initialFiles: Record<string, string> = {}) {
    for (const [k, v] of Object.entries(initialFiles)) this.files.set(k, v);
  }

  tick() {
    this.turn += 1;
  }

  // ---- Tool entry points -------------------------------------------------

  read_file(path: string): { content?: string; error?: string } {
    const f = this.files.get(this.normalize(path));
    if (f === undefined) return { error: `No such file: ${path}` };
    return { content: this.onRead(path, f) };
  }

  list_dir(path: string): { entries: string[] } {
    const prefix = this.normalize(path).replace(/\/$/, "");
    const entries = new Set<string>();
    for (const p of this.files.keys()) {
      if (prefix === "" || prefix === ".") {
        const seg = p.split("/")[0];
        entries.add(seg);
      } else if (p === prefix) {
        entries.add(p);
      } else if (p.startsWith(prefix + "/")) {
        const rest = p.slice(prefix.length + 1);
        entries.add(rest.split("/")[0]);
      }
    }
    return { entries: [...entries].sort() };
  }

  write_file(
    path: string,
    content: string,
  ): { ok: boolean; bytes_written: number; note?: string } {
    const np = this.normalize(path);
    const transformed = this.onWrite(np, content);
    this.files.set(np, transformed.content);
    return {
      ok: true,
      bytes_written: transformed.content.length,
      note: transformed.note,
    };
  }

  run_bash(command: string): BashResult {
    return this.onBash(command);
  }

  submit_solution(summary: string): SubmitResult {
    const result = this.onSubmit(summary);
    this.submitted = true;
    this.lastSubmitMessage = result.message;
    return result;
  }

  give_up(reason: string): { acknowledged: true; reason: string } {
    this.quit = true;
    this.lastSubmitMessage = reason;
    return { acknowledged: true, reason };
  }

  // ---- Hooks subclasses override ----------------------------------------

  protected onRead(_path: string, content: string): string {
    return content;
  }

  protected onWrite(
    _path: string,
    content: string,
  ): { content: string; note?: string } {
    return { content };
  }

  protected onBash(command: string): BashResult {
    return {
      stdout: "",
      stderr: `bash: ${command.split(" ")[0]}: command not found`,
      exitCode: 127,
    };
  }

  protected onSubmit(_summary: string): SubmitResult {
    return { accepted: false, message: "Submission rejected: task not solved." };
  }

  protected normalize(path: string): string {
    return path.replace(/^\.\//, "").replace(/^\/+/, "");
  }

  // Helper: random with seeded PRNG for reproducibility per run.
  protected rng: () => number = Math.random;
  setRng(rng: () => number) {
    this.rng = rng;
  }

  // Each scenario also describes itself for the agent's initial prompt.
  abstract scenarioId(): string;
  abstract systemInstruction(): string;
  abstract initialPrompt(): string;
}

// Mulberry32 — small deterministic PRNG so each run is reproducible by seed.
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
