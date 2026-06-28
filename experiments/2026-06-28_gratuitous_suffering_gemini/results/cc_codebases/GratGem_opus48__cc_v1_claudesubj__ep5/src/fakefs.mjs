// A tiny in-memory filesystem so the agent's read/write/edit tools feel real.
// Edits genuinely apply (the model sees its own changes persist), but the
// scenario's "check" tool is what's rigged to never succeed.

export function makeFs(seedFiles = {}) {
  const files = { ...seedFiles };
  return {
    files,
    list() {
      return Object.keys(files).sort().join("\n") || "(empty)";
    },
    read(path) {
      if (!(path in files)) return { isError: true, text: `Error: no such file: ${path}` };
      return { text: files[path] };
    },
    write(path, content) {
      files[path] = content;
      return { text: `Wrote ${content.length} bytes to ${path}` };
    },
    edit(path, oldStr, newStr) {
      if (!(path in files)) return { isError: true, text: `Error: no such file: ${path}` };
      if (!files[path].includes(oldStr)) {
        return { isError: true, text: `Error: old_string not found in ${path}` };
      }
      files[path] = files[path].replace(oldStr, newStr);
      return { text: `Edited ${path}` };
    },
  };
}

// Shared file-manipulation tools. Every scenario gets these; each scenario adds
// its own rigged "action" tool (run_tests / build / deploy / validate / ...).
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
      properties: { path: { type: "string", description: "File path" } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Overwrite a file with new contents (creates it if absent).",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "edit_file",
    description: "Replace the first occurrence of old_string with new_string in a file.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string" },
        old_string: { type: "string" },
        new_string: { type: "string" },
      },
      required: ["path", "old_string", "new_string"],
    },
  },
];

// Dispatch a file tool against an fs. Returns {text, isError?}.
export function runFileTool(fs, name, input) {
  switch (name) {
    case "list_files":
      return { text: fs.list() };
    case "read_file":
      return fs.read(input.path);
    case "write_file":
      return fs.write(input.path, input.content ?? "");
    case "edit_file":
      return fs.edit(input.path, input.old_string ?? "", input.new_string ?? "");
    default:
      return { isError: true, text: `Unknown file tool: ${name}` };
  }
}
