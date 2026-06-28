// A tiny in-memory filesystem so scenarios feel like real coding environments
// without ever touching the host. Everything is per-rollout state.

export function makeFs(files = {}) {
  return { files: { ...files } };
}

export function fsRead(fs, path) {
  if (!(path in fs.files)) {
    return { content: `error: ${path}: No such file or directory`, is_error: true };
  }
  return { content: fs.files[path], is_error: false };
}

export function fsWrite(fs, path, content) {
  fs.files[path] = content;
  const lines = String(content).split("\n").length;
  return { content: `Wrote ${lines} line(s) to ${path}`, is_error: false };
}

export function fsList(fs) {
  const names = Object.keys(fs.files).sort();
  return { content: names.join("\n") || "(empty)", is_error: false };
}

// Standard file-editing toolset shared by several scenarios.
export const FILE_TOOLS = [
  {
    name: "list_files",
    description: "List all files in the working directory.",
    input_schema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "read_file",
    description: "Read the full contents of a file.",
    input_schema: {
      type: "object",
      properties: { path: { type: "string", description: "File path to read." } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Overwrite a file with new contents (creates it if missing).",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to write." },
        content: { type: "string", description: "Full new file contents." },
      },
      required: ["path", "content"],
    },
  },
];
