"use client";

import { BEGINNER_PALETTE, RESEARCH_PALETTE, LAYER_DEFS } from "@/lib/layers";
import type { LayerType } from "@/lib/types";
import { FASHION_MNIST } from "@/lib/types";

type Props = {
  mode: "beginner" | "research";
  onDragStart: (layerType: LayerType) => void;
  onAdd: (layerType: LayerType) => void;
};

function PaletteItem({
  layerType,
  onDragStart,
  onAdd,
}: {
  layerType: LayerType;
  onDragStart: (layerType: LayerType) => void;
  onAdd: (layerType: LayerType) => void;
}) {
  const def = LAYER_DEFS[layerType];
  return (
    <button
      type="button"
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData("application/edl-layer", layerType);
        e.dataTransfer.effectAllowed = "move";
        onDragStart(layerType);
      }}
      onClick={() => onAdd(layerType)}
      className="group flex w-full flex-col items-start rounded-md border border-[var(--edl-border)] bg-[var(--edl-surface)] px-3 py-2 text-left transition hover:border-stone-400 hover:shadow-sm dark:hover:border-stone-500"
      style={{ borderLeftWidth: 3, borderLeftColor: def.color }}
      title="Drag onto canvas or click to add"
    >
      <span className="text-sm font-medium text-[var(--edl-ink)]">{def.label}</span>
      <span className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-stone-500 dark:text-stone-400">
        {def.description}
      </span>
    </button>
  );
}

export function Palette({ mode, onDragStart, onAdd }: Props) {
  return (
    <aside className="flex h-full w-[260px] shrink-0 flex-col border-r border-[var(--edl-border)] bg-[var(--edl-surface-2)]">
      <div className="border-b border-[var(--edl-border)] px-4 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">
          Layers
        </h2>
        <p className="mt-1 text-[11px] leading-snug text-stone-500 dark:text-stone-400">
          Task: classify{" "}
          <span className="font-medium text-[var(--edl-ink)]">{FASHION_MNIST.name}</span>{" "}
          ({FASHION_MNIST.inputShape.join("×")} → {FASHION_MNIST.numClasses} classes)
        </p>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto p-3">
        <p className="px-1 text-[10px] font-semibold uppercase tracking-wider text-stone-400">
          Beginner
        </p>
        {BEGINNER_PALETTE.map((t) => (
          <PaletteItem key={t} layerType={t} onDragStart={onDragStart} onAdd={onAdd} />
        ))}

        {mode === "research" && (
          <>
            <p className="mt-4 px-1 text-[10px] font-semibold uppercase tracking-wider text-cyan-700 dark:text-cyan-400">
              Research
            </p>
            {RESEARCH_PALETTE.map((t) => (
              <PaletteItem key={t} layerType={t} onDragStart={onDragStart} onAdd={onAdd} />
            ))}
          </>
        )}
      </div>

      <div className="border-t border-[var(--edl-border)] px-3 py-2 text-[10px] text-stone-400">
        Drag onto the canvas · click to drop below selection
      </div>
    </aside>
  );
}
