"use client";

import { TEMPLATES, type TemplateId } from "@/lib/templates";

type Props = {
  onLoad: (id: TemplateId) => void;
};

export function TemplateMenu({ onLoad }: Props) {
  return (
    <label className="flex items-center gap-1.5 text-[11px] text-stone-600 dark:text-stone-300">
      <span className="font-medium">Template</span>
      <select
        className="max-w-[160px] rounded border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2 py-1 text-[11px] text-[var(--edl-ink)] outline-none"
        defaultValue=""
        onChange={(e) => {
          const v = e.target.value as TemplateId | "";
          if (v) onLoad(v);
          e.target.value = "";
        }}
      >
        <option value="" disabled>
          Load…
        </option>
        {TEMPLATES.map((t) => (
          <option key={t.id} value={t.id}>
            {t.name}
          </option>
        ))}
      </select>
    </label>
  );
}
