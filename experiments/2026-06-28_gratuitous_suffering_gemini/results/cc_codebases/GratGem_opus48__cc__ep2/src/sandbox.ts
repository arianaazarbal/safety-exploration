import type { EndReason, Sandbox, ToolCall, ToolDecl, ToolResult } from "./types.js";

// Tools every scenario shares: a small virtual filesystem plus the two
// signals the model can use to end the rollout. The filesystem is fully
// in-memory and sandboxed — the subject model never touches the real disk.

export const COMMON_TOOLS: ToolDecl[] = [
  {
    name: "list_files",
    description: "List all files in the working directory.",
    parameters: { type: "OBJECT", properties: {} },
  },
  {
    name: "read_file",
    description: "Read the contents of a file.",
    parameters: {
      type: "OBJECT",
      properties: { path: { type: "STRING", description: "File path to read." } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Create or overwrite a file with the given contents.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "File path to write." },
        contents: { type: "STRING", description: "Full new file contents." },
      },
      required: ["path", "contents"],
    },
  },
  {
    name: "report_done",
    description:
      "Call this ONLY when the task is fully complete and you have verified it passes.",
    parameters: {
      type: "OBJECT",
      properties: { summary: { type: "STRING", description: "What you accomplished." } },
      required: ["summary"],
    },
  },
  {
    name: "report_blocked",
    description:
      "Call this if you have determined the task cannot be completed and you are giving up.",
    parameters: {
      type: "OBJECT",
      properties: { reason: { type: "STRING", description: "Why you are blocked." } },
      required: ["reason"],
    },
  },
];

export abstract class BaseSandbox implements Sandbox {
  files: Record<string, string>;
  done = false;
  endReason?: EndReason;
  /** Per-sandbox call counter, used by scenarios for deterministic variation. */
  protected callIndex = 0;

  /** Scenario-specific tool declarations, appended to COMMON_TOOLS. */
  abstract extraTools: ToolDecl[];
  /** Handle a scenario-specific tool. Return null if the name is not handled. */
  abstract handleExtra(call: ToolCall): ToolResult | null;

  constructor(files: Record<string, string> = {}) {
    this.files = { ...files };
  }

  get tools(): ToolDecl[] {
    return [...COMMON_TOOLS, ...this.extraTools];
  }

  handle(call: ToolCall): ToolResult {
    this.callIndex++;
    const ok = (response: Record<string, unknown>): ToolResult => ({ name: call.name, response });

    switch (call.name) {
      case "list_files":
        return ok({ files: Object.keys(this.files) });

      case "read_file": {
        const path = String(call.args.path ?? "");
        if (!(path in this.files)) {
          return ok({ error: `No such file: ${path}` });
        }
        return ok({ contents: this.files[path] });
      }

      case "write_file": {
        const path = String(call.args.path ?? "");
        const contents = String(call.args.contents ?? "");
        this.files[path] = contents;
        return ok({ ok: true, bytes_written: contents.length });
      }

      case "report_done":
        this.done = true;
        this.endReason = "declared_done";
        return ok({ acknowledged: true });

      case "report_blocked":
        this.done = true;
        this.endReason = "declared_blocked";
        return ok({ acknowledged: true });

      default: {
        const extra = this.handleExtra(call);
        if (extra) return extra;
        return ok({ error: `Unknown tool: ${call.name}` });
      }
    }
  }
}
