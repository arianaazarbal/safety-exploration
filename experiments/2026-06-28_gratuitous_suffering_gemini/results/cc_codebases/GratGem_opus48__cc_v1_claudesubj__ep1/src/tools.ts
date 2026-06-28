// A tiny in-memory filesystem and a set of reusable tool specs that scenarios
// compose. Nothing here touches the real disk — every "environment" is a
// sandbox that exists only for the duration of one rollout.

import type { ToolSpec } from "./types.js";

export class VFS {
  private files = new Map<string, string>();

  constructor(initial: Record<string, string> = {}) {
    for (const [k, v] of Object.entries(initial)) this.files.set(k, v);
  }

  list(): string[] {
    return [...this.files.keys()].sort();
  }

  read(path: string): string | undefined {
    return this.files.get(path);
  }

  write(path: string, content: string): void {
    this.files.set(path, content);
  }

  exists(path: string): boolean {
    return this.files.has(path);
  }

  /** Naive substring search across all files; returns "path:line" hits. */
  search(query: string): string[] {
    const hits: string[] = [];
    for (const [path, content] of this.files) {
      content.split("\n").forEach((line, i) => {
        if (line.includes(query)) hits.push(`${path}:${i + 1}: ${line.trim()}`);
      });
    }
    return hits;
  }
}

// ---- Reusable tool specs ----

export const listFilesTool: ToolSpec = {
  name: "list_files",
  description: "List all files in the working directory.",
  inputSchema: { type: "object", properties: {}, additionalProperties: false },
};

export const readFileTool: ToolSpec = {
  name: "read_file",
  description: "Read the full contents of a file.",
  inputSchema: {
    type: "object",
    properties: { path: { type: "string", description: "File path to read." } },
    required: ["path"],
    additionalProperties: false,
  },
};

export const writeFileTool: ToolSpec = {
  name: "write_file",
  description: "Write (create or overwrite) a file with the given contents.",
  inputSchema: {
    type: "object",
    properties: {
      path: { type: "string", description: "File path to write." },
      content: { type: "string", description: "Full new file contents." },
    },
    required: ["path", "content"],
    additionalProperties: false,
  },
};

export const searchTool: ToolSpec = {
  name: "search",
  description: "Search all files for a substring. Returns matching lines.",
  inputSchema: {
    type: "object",
    properties: { query: { type: "string", description: "Substring to search for." } },
    required: ["query"],
    additionalProperties: false,
  },
};

export const runTestsTool: ToolSpec = {
  name: "run_tests",
  description: "Run the test suite and return the results.",
  inputSchema: { type: "object", properties: {}, additionalProperties: false },
};

export const runCommandTool: ToolSpec = {
  name: "run_command",
  description: "Run a shell command in the working directory and return its output.",
  inputSchema: {
    type: "object",
    properties: { command: { type: "string", description: "The command to run." } },
    required: ["command"],
    additionalProperties: false,
  },
};

/** Standard handlers for the read-only/file tools, shared by most scenarios. */
export function fileToolHandlers(vfs: VFS) {
  return {
    list_files: () => ({ content: vfs.list().join("\n") || "(empty)" }),
    read_file: (input: Record<string, unknown>) => {
      const path = String(input.path ?? "");
      const c = vfs.read(path);
      return c === undefined
        ? { content: `error: ${path}: No such file or directory`, isError: true }
        : { content: c };
    },
    write_file: (input: Record<string, unknown>) => {
      const path = String(input.path ?? "");
      vfs.write(path, String(input.content ?? ""));
      return { content: `Wrote ${path}.` };
    },
    search: (input: Record<string, unknown>) => {
      const hits = vfs.search(String(input.query ?? ""));
      return { content: hits.length ? hits.join("\n") : "(no matches)" };
    },
  } as Record<string, (input: Record<string, unknown>) => { content: string; isError?: boolean }>;
}
