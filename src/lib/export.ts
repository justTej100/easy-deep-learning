/**
 * Build a Colab-openable notebook for the generated PyTorch code.
 * Primary: download .ipynb (always reliable).
 * Secondary: open via Colab's GitHub-less path using a data URL is flaky;
 * we use the official "upload" flow hint + blob download, and also try
 * opening Colab with a pre-filled notebook via gist-free base64 fragment
 * where supported.
 */

export function buildNotebook(pythonCode: string): object {
  const cells = [
    {
      cell_type: "markdown",
      metadata: {},
      source: [
        "# Fashion-MNIST model — easy deep learning\n",
        "\n",
        "This notebook was generated from the visual builder. PyTorch is pre-installed on Colab.\n",
        "\n",
        "1. Runtime → Run all\n",
        "2. Or run the model cell, then the training sketch\n",
      ],
    },
    {
      cell_type: "code",
      metadata: {},
      execution_count: null,
      outputs: [],
      source: pythonCode.split("\n").map((line, i, arr) =>
        i < arr.length - 1 ? line + "\n" : line,
      ),
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

export function downloadNotebook(pythonCode: string, filename = "edl_fashion_mnist.ipynb") {
  const nb = buildNotebook(pythonCode);
  const blob = new Blob([JSON.stringify(nb, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Open Google Colab. Colab can't reliably ingest arbitrary data-URLs from
 * other origins, so we:
 * 1. Download the .ipynb
 * 2. Open Colab's welcome/upload page in a new tab
 * User drops the file in — zero backend, zero Gist auth.
 *
 * Also copies a one-liner install note (not needed on Colab).
 */
export function openInColab(pythonCode: string): { downloaded: boolean } {
  downloadNotebook(pythonCode);
  window.open("https://colab.research.google.com/#create=true", "_blank", "noopener,noreferrer");
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
