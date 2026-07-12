"use client";

import { useMemo, useState } from "react";
import { generateCode, type Framework } from "@/lib/codegen";
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
  const [framework, setFramework] = useState<Framework>("pytorch");

  const { code, error } = useMemo(
    () => generateCode(nodes, edges, { framework }),
    [nodes, edges, framework],
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

  const filename =
    framework === "keras" ? "visual_net_keras.py" : "visual_net.py";

  return (
    <div className="flex h-full flex-col bg-stone-950 text-stone-100">
      <div className="flex flex-wrap items-center gap-2 border-b border-stone-800 px-3 py-2">
        <h2 className="mr-auto text-xs font-semibold uppercase tracking-wider text-stone-400">
          Export code
        </h2>
        <div className="flex rounded border border-stone-700 p-0.5 text-[11px]">
          <button
            type="button"
            onClick={() => setFramework("pytorch")}
            className={`rounded px-2 py-0.5 ${
              framework === "pytorch" ? "bg-zinc-100 text-zinc-950" : "text-stone-400"
            }`}
          >
            PyTorch
          </button>
          <button
            type="button"
            onClick={() => setFramework("keras")}
            className={`rounded px-2 py-0.5 ${
              framework === "keras" ? "bg-zinc-100 text-zinc-950" : "text-stone-400"
            }`}
          >
            Keras
          </button>
        </div>
        <button
          type="button"
          disabled={!code}
          onClick={handleCopy}
          className="rounded bg-stone-800 px-2.5 py-1 text-[11px] font-medium text-stone-100 hover:bg-stone-700 disabled:opacity-40"
        >
          {copied ? "Copied" : "Copy"}
        </button>
        <button
          type="button"
          disabled={!code}
          onClick={() => code && downloadPythonFile(code, filename)}
          className="rounded bg-stone-800 px-2.5 py-1 text-[11px] font-medium text-stone-100 hover:bg-stone-700 disabled:opacity-40"
        >
          Download .py
        </button>
        {framework === "pytorch" && (
          <>
            <button
              type="button"
              disabled={!code}
              onClick={() => code && downloadNotebook(code)}
              className="rounded bg-stone-800 px-2.5 py-1 text-[11px] font-medium text-stone-100 hover:bg-stone-700 disabled:opacity-40"
            >
              .ipynb
            </button>
            <button
              type="button"
              disabled={!code}
              onClick={handleColab}
              className="rounded bg-zinc-100 px-2.5 py-1 text-[11px] font-medium text-zinc-950 hover:bg-white disabled:opacity-40"
            >
              Colab
            </button>
          </>
        )}
      </div>

      {colabHint && (
        <div className="border-b border-amber-700/50 bg-amber-950/80 px-3 py-2 text-[11px] text-amber-100">
          Downloaded notebook and opened Colab — upload the .ipynb via File → Upload notebook.
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
