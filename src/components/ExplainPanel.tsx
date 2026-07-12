"use client";

import { LAYER_DEFS, formatShape } from "@/lib/layers";
import type { LayerNodeData, LayerParams } from "@/lib/types";
import { FASHION_MNIST } from "@/lib/types";
import { LayerTypeIcon } from "@/components/LayerIcons";

type Props = {
  data: LayerNodeData | null;
  onChangeParams: (params: LayerParams) => void;
};

export function ExplainPanel({ data, onChangeParams }: Props) {
  if (!data) {
    return (
      <div className="flex h-full flex-col justify-center px-4 text-sm text-stone-500 dark:text-stone-400">
        <p className="font-medium text-[var(--edl-ink)]">Select a layer</p>
        <p className="mt-1 text-[13px] leading-relaxed">
          Click any node to see what it does in plain English, why you&apos;d use it, and
          tweak its settings.
        </p>
        <div className="mt-6 rounded-md bg-zinc-900/40 px-3 py-2 text-[12px] leading-relaxed text-zinc-100 dark:bg-zinc-900 dark:text-zinc-100">
          <strong className="font-semibold">Task:</strong> {FASHION_MNIST.description}
        </div>
      </div>
    );
  }

  const def = LAYER_DEFS[data.layerType];

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div
        className="border-b border-[var(--edl-border)] px-4 py-3"
        style={{ background: def.accent }}
      >
        <div className="flex items-start gap-3">
          <LayerTypeIcon type={data.layerType} color={def.color} />
          <div className="min-w-0">
            <p
              className="text-[10px] font-semibold uppercase tracking-wider"
              style={{ color: def.color }}
            >
              {def.category}
            </p>
            <h3 className="text-base font-semibold text-[var(--edl-ink)]">{def.label}</h3>
            <p className="mt-1 font-mono text-[11px] text-[var(--edl-muted)]">
              {formatShape(data.inputShape)} → {formatShape(data.outputShape)}
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-4 px-4 py-3 text-[13px] leading-relaxed text-stone-700 dark:text-stone-300">
        {data.error && (
          <div className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-rose-800 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-200">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-rose-600 dark:text-rose-400">
              Shape problem
            </p>
            <p className="mt-1">{data.error}</p>
          </div>
        )}

        <section>
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-stone-400">
            What it does
          </h4>
          <p className="mt-1">{def.description}</p>
        </section>

        <section>
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-stone-400">
            Why use it
          </h4>
          <p className="mt-1">{def.why}</p>
        </section>

        <section>
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-stone-400">
            Analogy
          </h4>
          <p className="mt-1 italic text-stone-600 dark:text-stone-400">{def.analogy}</p>
        </section>

        {data.layerType === "LoopBlock" && (
          <section className="rounded-md border border-zinc-700 bg-zinc-900/40 px-3 py-2 text-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100">
            <h4 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-300 dark:text-zinc-400">
              Research idea
            </h4>
            <p className="mt-1">
              Looped / recurrent-depth transformers reuse the same block weights for N
              forward passes. You spend more compute at inference time instead of storing
              more parameters — useful when you want a deeper &quot;think harder&quot; mode
              without a bigger model file.
            </p>
          </section>
        )}

        {def.params.length > 0 && (
          <section>
            <h4 className="text-[10px] font-semibold uppercase tracking-wider text-stone-400">
              Settings
            </h4>
            <div className="mt-2 space-y-3">
              {def.params.map((field) => (
                <label key={field.key} className="block">
                  <span className="flex items-baseline justify-between gap-2">
                    <span className="text-xs font-medium text-[var(--edl-ink)]">
                      {field.label}
                    </span>
                    <span className="font-mono text-[11px] text-stone-500">
                      {String(data.params[field.key] ?? field.defaultValue)}
                    </span>
                  </span>
                  <input
                    type="number"
                    className="mt-1 w-full rounded border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2 py-1.5 text-sm text-[var(--edl-ink)] outline-none focus:border-zinc-500 focus:ring-1 focus:ring-zinc-500"
                    min={field.min}
                    max={field.max}
                    step={field.step ?? 1}
                    value={Number(data.params[field.key] ?? field.defaultValue)}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      onChangeParams({ ...data.params, [field.key]: v });
                    }}
                  />
                  <span className="mt-0.5 block text-[11px] text-stone-500 dark:text-stone-400">
                    {field.help}
                  </span>
                </label>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
