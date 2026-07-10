"use client";

import { useMemo, useState } from "react";
import { generatePyTorchCode } from "@/lib/codegen";
import { copyToClipboard, downloadNotebook, openInColab } from "@/lib/export";
import { downloadPythonFile } from "@/lib/persistence";
import type { LayerNodeData } from "@/lib/types";

type Props = {
  nodes: Array<{ id: string; data: LayerNodeData }>;
  edges: Array<{ source: string; target: string }>;
};

export function CodePanel({ nodes, edges }: Props) {
  const [copied, setCopied] = useState(false);
  const [colabHint, setColabHint] = useState(false);

  const { code, error } = useMemo(
    () => generatePyTorchCode(nodes, edges),
    [nodes, edges],
  );

  async function handleCopy() {
    if (!code) return;
    const ok = await copyToClipboard(code);
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }

  function handleColab() {
    if (!code) return;
    openInColab(code);
    setColabHint(true);
    setTimeout(() => setColabHint(false), 8000);
  }

  return (
    <div className="flex h-full flex-col bg-stone-950 text-stone-100">
      <div className="flex flex-wrap items-center gap-2 border-b border-stone-800 px-3 py-2">
        <h2 className="mr-auto text-xs font-semibold uppercase tracking-wider text-stone-400">
          Annotated PyTorch
        </h2>
        <button
          type="button"
          disabled={!code}
          onClick={handleCopy}
          className="rounded bg-stone-800 px-2.5 py-1 text-[11px] font-medium text-stone-100 hover:bg-stone-700 disabled:opacity-40"
        >
          {copied ? "Copied" : "Copy code"}
        </button>
        <button
          type="button"
          disabled={!code}
          onClick={() => code && downloadPythonFile(code)}
          className="rounded bg-stone-800 px-2.5 py-1 text-[11px] font-medium text-stone-100 hover:bg-stone-700 disabled:opacity-40"
        >
          Download .py
        </button>
        <button
          type="button"
          disabled={!code}
          onClick={() => code && downloadNotebook(code)}
          className="rounded bg-stone-800 px-2.5 py-1 text-[11px] font-medium text-stone-100 hover:bg-stone-700 disabled:opacity-40"
        >
          Download .ipynb
        </button>
        <button
          type="button"
          disabled={!code}
          onClick={handleColab}
          className="rounded bg-amber-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-amber-500 disabled:opacity-40"
        >
          Open in Colab
        </button>
      </div>

      {colabHint && (
        <div className="border-b border-amber-700/50 bg-amber-950/80 px-3 py-2 text-[11px] text-amber-100">
          Downloaded <code className="font-mono">edl_fashion_mnist.ipynb</code> and opened
          Colab. Upload that file via File → Upload notebook (no account on this site —
          Colab may ask you to sign in with Google).
        </div>
      )}

      {error ? (
        <div className="px-4 py-3 text-sm text-rose-300">{error}</div>
      ) : (
        <pre className="flex-1 overflow-auto p-4 font-mono text-[11px] leading-relaxed text-emerald-100/90">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}
