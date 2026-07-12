/**
 * Export helpers: Colab notebook, clipboard, legacy single-cell notebook.
 */

import type { ProjectState } from "./types";
import { buildColabNotebookFromProject } from "./colab";
import { downloadBlob } from "./zip";

function toSources(text: string): string[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  return lines.map((line, i) => (i < lines.length - 1 ? `${line}\n` : line));
}

/** Legacy: one markdown + one code cell (used by Code panel .ipynb). */
export function buildNotebook(pythonCode: string): object {
  const cells = [
    {
      cell_type: "markdown",
      metadata: {},
      source: toSources(
        `# Fashion-MNIST model — easy deep learning

This notebook was generated from the visual builder. PyTorch is pre-installed on Colab.

1. Runtime → Run all
2. Or run the model cell, then the training sketch
`,
      ),
    },
    {
      cell_type: "code",
      metadata: {},
      execution_count: null,
      outputs: [],
      source: toSources(pythonCode),
    },
  ];

  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
      language_info: {
        name: "python",
        version: "3.10.0",
      },
      colab: { provenance: [] },
    },
    cells,
  };
}

export function downloadNotebook(
  pythonCode: string,
  filename = "edl_fashion_mnist.ipynb",
) {
  const nb = buildNotebook(pythonCode);
  downloadBlob(
    new Blob([JSON.stringify(nb, null, 2)], { type: "application/json" }),
    filename,
  );
}

/**
 * Entire project as one Google Colab notebook (.ipynb):
 * intro, imports, one cell pair per layer block, assemble, smoke test, train sketch.
 */
export function downloadColabProject(
  state: ProjectState,
  filename = "edl_fashion_mnist_colab.ipynb",
): { ok: true } | { ok: false; error: string } {
  const { notebook, error } = buildColabNotebookFromProject(state);
  if (error || !notebook || Object.keys(notebook).length === 0) {
    return { ok: false, error: error ?? "Could not build Colab notebook." };
  }
  downloadBlob(
    new Blob([JSON.stringify(notebook, null, 2)], { type: "application/json" }),
    filename,
  );
  return { ok: true };
}

/**
 * Open Google Colab. Downloads a block-structured .ipynb then opens Colab.
 */
export function openInColab(pythonCode: string): { downloaded: boolean } {
  downloadNotebook(pythonCode);
  window.open(
    "https://colab.research.google.com/#create=true",
    "_blank",
    "noopener,noreferrer",
  );
  return { downloaded: true };
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}
