"use client";

import dynamic from "next/dynamic";

const Builder = dynamic(
  () => import("@/components/Builder").then((m) => m.Builder),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-dvh items-center justify-center bg-[#f3efe6] text-stone-600">
        <p className="font-[family-name:var(--font-display)] text-xl text-teal-900">
          easy deep learning
        </p>
      </div>
    ),
  },
);

export default function Home() {
  return <Builder />;
}
