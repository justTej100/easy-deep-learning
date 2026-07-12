"use client";

import { useMemo, useState } from "react";
import {
  ADVANCED_LAYERS,
  BASIC_LAYERS,
  EXTRA_LAYERS,
  LAYER_DEFS,
  RESEARCH_PALETTE,
} from "@/lib/layers";
import {
  SIDEBAR_TEMPLATES,
  TEMPLATES,
  countTemplateNodes,
  type TemplateId,
} from "@/lib/templates";
import type { LayerType } from "@/lib/types";
import { LayerTypeIcon, TemplateTypeIcon } from "@/components/LayerIcons";

type Props = {
  mode: "beginner" | "research";
  onDragStart: (layerType: LayerType) => void;
  onAdd: (layerType: LayerType) => void;
  onLoadTemplate: (id: TemplateId) => void;
};

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      className={`shrink-0 text-stone-400 transition ${open ? "rotate-0" : "-rotate-90"}`}
      aria-hidden
    >
      <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

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
      className="flex w-full items-start gap-2.5 rounded-lg border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2.5 py-2 text-left transition hover:bg-[var(--edl-surface-2)]"
      title="Drag onto canvas or click to add"
    >
      <LayerTypeIcon type={layerType} color={def.color} />
      <span className="min-w-0">
        <span className="block text-[13px] font-semibold text-[var(--edl-ink)]">{def.label}</span>
        <span className="mt-0.5 block line-clamp-2 text-[11px] leading-snug text-stone-500 dark:text-stone-400">
          {def.description}
        </span>
      </span>
    </button>
  );
}

function Section({
  title,
  open,
  onToggle,
  hint,
  children,
}: {
  title: string;
  open: boolean;
  onToggle: () => void;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border-b border-[var(--edl-border)]">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between px-3 py-2.5 text-left"
      >
        <span>
          <span className="block text-[13px] font-semibold text-[var(--edl-ink)]">{title}</span>
          {hint && (
            <span className="mt-0.5 block text-[10px] text-stone-500 dark:text-stone-400">
              {hint}
            </span>
          )}
        </span>
        <Chevron open={open} />
      </button>
      {open && <div className="space-y-1.5 px-3 pb-3">{children}</div>}
    </div>
  );
}

export function Palette({ mode, onDragStart, onAdd, onLoadTemplate }: Props) {
  const [query, setQuery] = useState("");
  const [openBasic, setOpenBasic] = useState(true);
  const [openAdvanced, setOpenAdvanced] = useState(true);
  const [openExtras, setOpenExtras] = useState(false);
  const [openTemplates, setOpenTemplates] = useState(true);
  const [openResearch, setOpenResearch] = useState(mode === "research");

  const q = query.trim().toLowerCase();

  const matches = (t: LayerType) => {
    if (!q) return true;
    const d = LAYER_DEFS[t];
    return (
      d.label.toLowerCase().includes(q) || d.description.toLowerCase().includes(q)
    );
  };

  const basic = useMemo(() => BASIC_LAYERS.filter(matches), [q]);
  const advanced = useMemo(() => ADVANCED_LAYERS.filter(matches), [q]);
  const extras = useMemo(() => EXTRA_LAYERS.filter(matches), [q]);
  const research = useMemo(() => RESEARCH_PALETTE.filter(matches), [q]);

  const templates = useMemo(() => {
    const list = TEMPLATES.filter((t) => SIDEBAR_TEMPLATES.includes(t.id));
    if (!q) return list;
    return list.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.description.toLowerCase().includes(q),
    );
  }, [q]);

  return (
    <aside className="flex h-full w-[300px] shrink-0 flex-col border-r border-[var(--edl-border)] bg-[var(--edl-bg)]">
      <div className="border-b border-[var(--edl-border)] px-3 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-stone-500">
          Layers & templates
        </h2>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search…"
          className="mt-2 w-full rounded-md border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2.5 py-1.5 text-sm text-[var(--edl-ink)] outline-none focus:border-zinc-500"
        />
      </div>

      <div className="flex-1 overflow-y-auto">
        <Section
          title="Basic Layers"
          open={openBasic}
          onToggle={() => setOpenBasic((v) => !v)}
          hint="Drag to canvas or click to add"
        >
          {basic.map((t) => (
            <PaletteItem
              key={t}
              layerType={t}
              onDragStart={onDragStart}
              onAdd={onAdd}
            />
          ))}
          {basic.length === 0 && (
            <p className="text-[11px] text-stone-500">No matches</p>
          )}
        </Section>

        <Section
          title="Advanced Layers"
          open={openAdvanced}
          onToggle={() => setOpenAdvanced((v) => !v)}
        >
          {advanced.map((t) => (
            <PaletteItem
              key={t}
              layerType={t}
              onDragStart={onDragStart}
              onAdd={onAdd}
            />
          ))}
          {advanced.length === 0 && (
            <p className="text-[11px] text-stone-500">No matches</p>
          )}
        </Section>

        <Section
          title="Templates"
          open={openTemplates}
          onToggle={() => setOpenTemplates((v) => !v)}
          hint="Drag to canvas or click to add"
        >
          {templates.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => onLoadTemplate(t.id)}
              className="flex w-full items-center gap-2.5 rounded-lg border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2.5 py-2 text-left transition hover:bg-[var(--edl-surface-2)]"
            >
              <TemplateTypeIcon icon={t.icon} color={t.color} />
              <span className="min-w-0 flex-1">
                <span className="block text-[13px] font-semibold text-[var(--edl-ink)]">
                  {t.name}
                </span>
                <span className="mt-0.5 block line-clamp-2 text-[11px] text-stone-500 dark:text-stone-400">
                  {t.description}
                </span>
              </span>
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--edl-surface-2)] text-[10px] font-medium text-stone-500">
                {countTemplateNodes(t.id)}
              </span>
            </button>
          ))}
          {templates.length === 0 && (
            <p className="text-[11px] text-stone-500">No matches</p>
          )}
        </Section>

        <Section
          title="More layers"
          open={openExtras}
          onToggle={() => setOpenExtras((v) => !v)}
          hint="Activations & extras"
        >
          {extras.map((t) => (
            <PaletteItem
              key={t}
              layerType={t}
              onDragStart={onDragStart}
              onAdd={onAdd}
            />
          ))}
        </Section>

        {mode === "research" && (
          <Section
            title="Research"
            open={openResearch}
            onToggle={() => setOpenResearch((v) => !v)}
          >
            {research.map((t) => (
              <PaletteItem
                key={t}
                layerType={t}
                onDragStart={onDragStart}
                onAdd={onAdd}
              />
            ))}
          </Section>
        )}
      </div>
    </aside>
  );
}
