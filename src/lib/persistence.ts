import LZString from "lz-string";
import type { ProjectState } from "./types";

const HASH_PREFIX = "#p=";

export function encodeProject(state: ProjectState): string {
  const json = JSON.stringify(state);
  return LZString.compressToEncodedURIComponent(json);
}

export function decodeProject(encoded: string): ProjectState | null {
  try {
    const json = LZString.decompressFromEncodedURIComponent(encoded);
    if (!json) return null;
    const parsed = JSON.parse(json) as ProjectState;
    if (parsed.version !== 1 || !Array.isArray(parsed.nodes)) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function readProjectFromLocation(
  hash: string = typeof window !== "undefined" ? window.location.hash : "",
): ProjectState | null {
  if (!hash.startsWith(HASH_PREFIX)) return null;
  return decodeProject(hash.slice(HASH_PREFIX.length));
}

export function writeProjectToUrl(state: ProjectState): void {
  if (typeof window === "undefined") return;
  const encoded = encodeProject(state);
  const url = `${window.location.pathname}${window.location.search}${HASH_PREFIX}${encoded}`;
  window.history.replaceState(null, "", url);
}

export function downloadProjectFile(state: ProjectState, filename = "edl-project.json") {
  const blob = new Blob([JSON.stringify(state, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export async function loadProjectFile(file: File): Promise<ProjectState> {
  const text = await file.text();
  const parsed = JSON.parse(text) as ProjectState;
  if (parsed.version !== 1 || !Array.isArray(parsed.nodes)) {
    throw new Error("Not a valid easy deep learning project file.");
  }
  return parsed;
}

export function downloadPythonFile(code: string, filename = "fashion_mnist_net.py") {
  const blob = new Blob([code], { type: "text/x-python" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
