"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { LayerNodeData } from "@/lib/types";
import { LAYER_DEFS, formatShape } from "@/lib/layers";
import { LayerTypeIcon } from "@/components/LayerIcons";

export type LayerFlowNode = Node<LayerNodeData, "layer">;

function LayerNodeComponent({ data, selected }: NodeProps<LayerFlowNode>) {
  const def = LAYER_DEFS[data.layerType];
  const hasError = Boolean(data.error);

  return (
    <div
      className={`min-w-[180px] rounded-lg border-2 bg-[var(--edl-surface)] shadow-sm transition-shadow ${
        selected
          ? "ring-2 ring-offset-2 ring-zinc-400 shadow-md ring-offset-[var(--edl-canvas)]"
          : ""
      } ${hasError ? "border-rose-500" : "border-[var(--edl-border)]"}`}
      style={{ borderTopColor: def.color, borderTopWidth: 4 }}
    >
      {data.layerType !== "Input" && (
        <Handle
          type="target"
          position={Position.Top}
          className="!h-2.5 !w-2.5 !border-2 !border-[var(--edl-surface)] !bg-stone-500"
        />
      )}

      <div className="px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <span
            className="text-[10px] font-semibold uppercase tracking-wider"
            style={{ color: def.color }}
          >
            {def.category === "research" ? "research" : def.category}
          </span>
          {hasError && (
            <span className="rounded bg-rose-100 px-1.5 py-0.5 text-[10px] font-medium text-rose-700 dark:bg-rose-950 dark:text-rose-300">
              fix me
            </span>
          )}
        </div>
        <div className="mt-1.5 flex items-center gap-2">
          <LayerTypeIcon type={data.layerType} color={def.color} />
          <div className="text-sm font-semibold text-[var(--edl-ink)]">{def.label}</div>
        </div>
        <div className="mt-1 font-mono text-[11px] text-stone-500 dark:text-stone-400">
          {data.layerType === "Input" ? (
            <span>out {formatShape(data.outputShape)}</span>
          ) : (
            <span>
              {formatShape(data.inputShape)}
              <span className="mx-1 text-stone-300 dark:text-stone-600">→</span>
              {formatShape(data.outputShape)}
            </span>
          )}
        </div>
        {data.layerType === "LoopBlock" && (
          <div className="mt-1 text-[11px] text-[var(--edl-muted)]">
            ×{data.params.repeats ?? 2} shared passes
          </div>
        )}
      </div>

      {data.layerType !== "Output" && (
        <Handle
          type="source"
          position={Position.Bottom}
          className="!h-2.5 !w-2.5 !border-2 !border-[var(--edl-surface)] !bg-stone-500"
        />
      )}
    </div>
  );
}

export const LayerNode = memo(LayerNodeComponent);
