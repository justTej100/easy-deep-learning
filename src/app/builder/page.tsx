"use client";

import dynamic from "next/dynamic";

const Builder = dynamic(
  () => import("@/components/Builder").then((m) => m.Builder),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-dvh items-center justify-center bg-[var(--edl-bg)] text-[var(--edl-muted)]">
        <p className="font-[family-name:var(--font-display)] text-xl text-teal-800 dark:text-teal-300">
          easy deep learning
        </p>
      </div>
    ),
  },
);

export default function BuilderPage() {
  return <Builder />;
}
