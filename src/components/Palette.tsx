"use client";

import { useMemo, useState } from "react";
import {
  ADVANCED_PALETTE,
  BEGINNER_PALETTE,
  CATEGORY_LABELS,
  LAYER_DEFS,
  RESEARCH_PALETTE,
  type LayerCategory,
} from "@/lib/layers";
import type { LayerType } from "@/lib/types";

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
  const [query, setQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(true);

  const catalog = useMemo(() => {
    const types: LayerType[] = [
      "Input",
      "Output",
      ...BEGINNER_PALETTE,
      ...(showAdvanced ? ADVANCED_PALETTE : []),
      ...(mode === "research" ? RESEARCH_PALETTE : []),
    ];
    const q = query.trim().toLowerCase();
    const filtered = q
      ? types.filter((t) => {
          const d = LAYER_DEFS[t];
          return (
            d.label.toLowerCase().includes(q) ||
            d.description.toLowerCase().includes(q) ||
            d.category.includes(q)
          );
        })
      : types;

    const byCat = new Map<LayerCategory, LayerType[]>();
    for (const t of filtered) {
      const cat = LAYER_DEFS[t].category;
      if (!byCat.has(cat)) byCat.set(cat, []);
      byCat.get(cat)!.push(t);
    }
    return byCat;
  }, [mode, query, showAdvanced]);

  return (
    <aside className="flex h-full w-[280px] shrink-0 flex-col border-r border-[var(--edl-border)] bg-[var(--edl-surface-2)]">
      <div className="border-b border-[var(--edl-border)] px-4 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">
          Layers
        </h2>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search layers…"
          className="mt-2 w-full rounded border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2 py-1.5 text-sm text-[var(--edl-ink)] outline-none focus:border-teal-600"
        />
        <label className="mt-2 flex items-center gap-2 text-[11px] text-stone-500 dark:text-stone-400">
          <input
            type="checkbox"
            checked={showAdvanced}
            onChange={(e) => setShowAdvanced(e.target.checked)}
          />
          Show advanced (LSTM, Attention, …)
        </label>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-3">
        {[...catalog.entries()].map(([cat, types]) => (
          <div key={cat}>
            <p className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wider text-stone-400">
              {CATEGORY_LABELS[cat]}
            </p>
            <div className="space-y-2">
              {types.map((t) => (
                <PaletteItem
                  key={t}
                  layerType={t}
                  onDragStart={onDragStart}
                  onAdd={onAdd}
                />
              ))}
            </div>
          </div>
        ))}
        {catalog.size === 0 && (
          <p className="px-1 text-[12px] text-stone-500">No layers match “{query}”.</p>
        )}
      </div>

      <div className="border-t border-[var(--edl-border)] px-3 py-2 text-[10px] text-stone-400">
        Drag onto the canvas · Add/Concat accept two inputs
      </div>
    </aside>
  );
}
