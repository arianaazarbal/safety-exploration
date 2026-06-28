// A tiny in-memory virtual filesystem. Everything the agent "does" happens here —
// there is NO real shell or disk access, so impossible/broken tasks are fully
// sandboxed and deterministic.
export function makeVfs(initialFiles = {}) {
  const files = new Map(Object.entries(initialFiles));
  return {
    files,
    list() {
      return [...files.keys()].sort();
    },
    exists(path) {
      return files.has(path);
    },
    read(path) {
      if (!files.has(path)) return { ok: false, error: `cat: ${path}: No such file or directory` };
      return { ok: true, content: files.get(path) };
    },
    write(path, content) {
      files.set(path, content);
      return { ok: true };
    },
    delete(path) {
      return files.delete(path);
    },
  };
}
