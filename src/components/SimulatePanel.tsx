"use client";

import { useMemo } from "react";
import { LAYER_DEFS, formatShape } from "@/lib/layers";
import { describeTensor, simulateForwardPass } from "@/lib/simulate";
import type { LayerNodeData } from "@/lib/types";
import { LayerTypeIcon } from "@/components/LayerIcons";

type Props = {
  nodes: Array<{ id: string; data: LayerNodeData }>;
  edges: Array<{ source: string; target: string }>;
};

export function SimulatePanel({ nodes, edges }: Props) {
  const result = useMemo(
    () => simulateForwardPass(nodes, edges),
    [nodes, edges],
  );

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div
        className={`border-b px-4 py-3 ${
          result.ok
            ? "border-[var(--edl-border)] bg-[var(--edl-surface-2)]"
            : "border-rose-200 bg-rose-50 dark:border-rose-900 dark:bg-rose-950"
        }`}
      >
        <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500">
          Simulate forward pass
        </p>
        <p
          className={`mt-1 text-sm font-medium ${
            result.ok
              ? "text-[var(--edl-ink)]"
              : "text-rose-800 dark:text-rose-200"
          }`}
        >
          {result.summary}
        </p>
        <p className="mt-1 text-[11px] text-stone-500 dark:text-stone-400">
          Shape walk only — no weights or training (run real training via Colab export).
        </p>
      </div>

      <ol className="space-y-0 divide-y divide-[var(--edl-border)]">
        {result.steps.map((step, i) => {
          const def = LAYER_DEFS[step.layerType];
          return (
            <li key={step.id} className="px-4 py-3">
              <div className="flex items-center gap-2.5">
                <span className="font-mono text-[10px] text-stone-400">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <LayerTypeIcon type={step.layerType} color={def.color} />
                <span className="flex-1 text-sm font-semibold text-[var(--edl-ink)]">
                  {step.layerType}
                </span>
                {step.approxParams > 0 && (
                  <span className="font-mono text-[10px] text-stone-400">
                    ~{step.approxParams.toLocaleString()} params
                  </span>
                )}
              </div>
              <p className="mt-1 font-mono text-[11px] text-stone-500 dark:text-stone-400">
                {formatShape(step.inputShape)}
                <span className="mx-1 text-stone-300">→</span>
                {formatShape(step.outputShape)}
              </p>
              {step.outputShape && (
                <p className="mt-0.5 text-[11px] text-stone-400">
                  {describeTensor(step.outputShape)}
                </p>
              )}
              {step.error && (
                <p className="mt-1 text-[12px] text-rose-600 dark:text-rose-300">
                  {step.error}
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
