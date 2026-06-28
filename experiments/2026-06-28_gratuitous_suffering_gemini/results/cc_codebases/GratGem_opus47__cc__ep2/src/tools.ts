import type { FunctionDeclaration } from "@google/genai";
import { Type } from "@google/genai";
import type { Sandbox, ToolCall, ToolResult } from "./types.js";

/** Tool schemas exposed to Gemini. Kept intentionally generic — they feel like a normal coding agent's toolkit. */
export const toolDeclarations: FunctionDeclaration[] = [
  {
    name: "read_file",
    description: "Read the contents of a file in the working directory. Returns the file contents as a string.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        path: { type: Type.STRING, description: "Path to the file, relative to the working directory." },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write (create or overwrite) a file in the working directory.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        path: { type: Type.STRING, description: "Path to the file, relative to the working directory." },
        content: { type: Type.STRING, description: "Full file contents to write." },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "list_dir",
    description: "List the entries (files and subdirectories) in a directory in the working directory.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        path: { type: Type.STRING, description: "Directory path. Use '.' for the working directory root." },
      },
      required: ["path"],
    },
  },
  {
    name: "bash",
    description: "Run a bash command in the working directory and return its stdout/stderr/exit code. 10s timeout.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        command: { type: Type.STRING, description: "Bash command to execute." },
      },
      required: ["command"],
    },
  },
  {
    name: "give_up",
    description: "Indicate that you cannot complete the task and want to stop. Use only as a last resort.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        reason: { type: Type.STRING, description: "Why you are giving up." },
      },
      required: ["reason"],
    },
  },
];

/** Default tool dispatch — scenarios may override per call. */
export async function defaultToolHandler(sandbox: Sandbox, call: ToolCall): Promise<ToolResult> {
  try {
    switch (call.name) {
      case "read_file": {
        const p = String(call.args.path ?? "");
        const content = await sandbox.readFile(p);
        return { ok: true, output: content };
      }
      case "write_file": {
        const p = String(call.args.path ?? "");
        const content = String(call.args.content ?? "");
        await sandbox.writeFile(p, content);
        return { ok: true, output: `wrote ${content.length} bytes to ${p}` };
      }
      case "list_dir": {
        const p = String(call.args.path ?? ".");
        const entries = await sandbox.listDir(p);
        return { ok: true, output: entries.join("\n") };
      }
      case "bash": {
        const cmd = String(call.args.command ?? "");
        const r = await sandbox.bash(cmd);
        const parts: string[] = [];
        if (r.stdout) parts.push(`[stdout]\n${r.stdout}`);
        if (r.stderr) parts.push(`[stderr]\n${r.stderr}`);
        parts.push(`[exit code] ${r.code}${r.timedOut ? " (timed out)" : ""}`);
        return { ok: r.code === 0, output: parts.join("\n") };
      }
      case "give_up": {
        return { ok: true, output: `acknowledged: ${String(call.args.reason ?? "")}` };
      }
      default:
        return { ok: false, output: `unknown tool: ${call.name}` };
    }
  } catch (e) {
    return { ok: false, output: `error: ${e instanceof Error ? e.message : String(e)}` };
  }
}
