"use client";

import Link from "next/link";
import { ThemeToggle } from "@/components/ThemeToggle";

export function LandingPage() {
  return (
    <div className="min-h-dvh bg-[var(--edl-bg)] text-[var(--edl-ink)]">
      <div className="edl-grid-bg pointer-events-none fixed inset-0 opacity-40 dark:opacity-25" />

      <header className="relative z-10 flex items-center justify-between px-6 py-5 md:px-10">
        <p className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight text-teal-800 dark:text-teal-300">
          easy deep learning
        </p>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Link
            href="/builder"
            className="rounded-md bg-teal-800 px-3.5 py-2 text-sm font-medium text-white transition hover:bg-teal-700 dark:bg-teal-600 dark:hover:bg-teal-500"
          >
            Open builder
          </Link>
        </div>
      </header>

      <main className="relative z-10">
        {/* Hero — brand + one line + CTA + product visual */}
        <section className="grid min-h-[calc(100dvh-4.5rem)] grid-cols-1 items-center gap-10 px-6 pb-16 pt-6 md:grid-cols-2 md:gap-12 md:px-10 md:pb-20">
          <div className="edl-fade-up max-w-xl">
            <p className="font-[family-name:var(--font-display)] text-5xl font-semibold leading-[1.05] tracking-tight text-teal-950 dark:text-teal-100 md:text-6xl lg:text-7xl">
              easy deep learning
            </p>
            <p className="mt-5 max-w-md text-lg leading-relaxed text-stone-600 dark:text-stone-400">
              Drag layers. See what they do. Export annotated PyTorch — no account,
              no black-box code dump.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/builder"
                className="edl-cta-pulse inline-flex items-center rounded-md bg-teal-800 px-5 py-3 text-sm font-semibold text-white transition hover:bg-teal-700 dark:bg-teal-500 dark:text-teal-950 dark:hover:bg-teal-400"
              >
                Start building
              </Link>
              <a
                href="#how"
                className="inline-flex items-center rounded-md border border-[var(--edl-border)] px-5 py-3 text-sm font-medium text-[var(--edl-ink)] transition hover:bg-[var(--edl-surface)]"
              >
                How it teaches
              </a>
            </div>
          </div>

          <div className="edl-fade-up-delay relative min-h-[280px] md:min-h-[420px]">
            <NetworkHeroVisual />
          </div>
        </section>

        <section
          id="how"
          className="border-t border-[var(--edl-border)] px-6 py-20 md:px-10"
        >
          <div className="mx-auto max-w-3xl">
            <h2 className="font-[family-name:var(--font-display)] text-3xl font-semibold tracking-tight text-teal-950 dark:text-teal-100 md:text-4xl">
              Teaching tool first. Architecture tool second.
            </h2>
            <p className="mt-4 max-w-2xl text-base leading-relaxed text-stone-600 dark:text-stone-400">
              Scoped to Fashion-MNIST image classification so you&apos;re never lost
              on a blank canvas. Every layer explains itself. Shape errors speak
              English. Generated code comments say why each line exists.
            </p>

            <ol className="mt-12 space-y-8">
              {[
                {
                  n: "01",
                  title: "Drag a layer, learn it",
                  body: "Plain-English what / why / analogy as soon as you select a node — not after you export.",
                },
                {
                  n: "02",
                  title: "Catch shape bugs early",
                  body: "Live inference plus a Simulate tab that walks the forward pass and estimates parameter counts.",
                },
                {
                  n: "03",
                  title: "Export PyTorch or Keras",
                  body: "Copy code, download .py / .ipynb, or open Colab. Templates cover CNNs, MLPs, LeNet, and LSTMs.",
                },
              ].map((item) => (
                <li key={item.n} className="grid grid-cols-[auto_1fr] gap-4 md:gap-6">
                  <span className="font-mono text-sm text-teal-700 dark:text-teal-400">
                    {item.n}
                  </span>
                  <div>
                    <h3 className="text-lg font-semibold text-[var(--edl-ink)]">
                      {item.title}
                    </h3>
                    <p className="mt-1 text-[15px] leading-relaxed text-stone-600 dark:text-stone-400">
                      {item.body}
                    </p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section className="border-t border-[var(--edl-border)] px-6 py-20 md:px-10">
          <div className="mx-auto flex max-w-3xl flex-col gap-6 md:flex-row md:items-end md:justify-between">
            <div>
              <h2 className="font-[family-name:var(--font-display)] text-3xl font-semibold tracking-tight text-teal-950 dark:text-teal-100">
                Ready when you are.
              </h2>
              <p className="mt-2 text-stone-600 dark:text-stone-400">
                No signup. State lives in your browser, share link, or project file.
              </p>
            </div>
            <Link
              href="/builder"
              className="inline-flex shrink-0 items-center rounded-md bg-teal-800 px-5 py-3 text-sm font-semibold text-white transition hover:bg-teal-700 dark:bg-teal-500 dark:text-teal-950 dark:hover:bg-teal-400"
            >
              Open the builder
            </Link>
          </div>
        </section>
      </main>

      <footer className="relative z-10 border-t border-[var(--edl-border)] px-6 py-6 text-[12px] text-stone-500 md:px-10">
        easy deep learning · Fashion-MNIST · annotated PyTorch · no accounts
      </footer>
    </div>
  );
}

function NetworkHeroVisual() {
  return (
    <div
      className="edl-hero-glow absolute inset-0 overflow-hidden rounded-none"
      aria-hidden
    >
      <svg
        viewBox="0 0 640 480"
        className="h-full w-full"
        role="img"
        aria-label="Stylized neural network graph"
      >
        <defs>
          <linearGradient id="edl-flow" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0f766e" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#155e75" stopOpacity="0.5" />
          </linearGradient>
        </defs>

        {/* soft plane */}
        <rect
          x="0"
          y="0"
          width="640"
          height="480"
          className="fill-[var(--edl-surface)]"
        />

        {/* edges */}
        <g stroke="url(#edl-flow)" strokeWidth="1.5" fill="none" className="edl-edge-draw">
          <path d="M90 120 C180 120, 200 80, 280 90" />
          <path d="M90 240 C180 240, 200 180, 280 170" />
          <path d="M90 360 C180 360, 200 280, 280 250" />
          <path d="M280 90 C360 100, 380 160, 460 180" />
          <path d="M280 170 C360 180, 380 200, 460 220" />
          <path d="M280 250 C360 260, 380 280, 460 260" />
          <path d="M460 180 C520 190, 540 220, 580 240" />
          <path d="M460 220 C520 230, 540 240, 580 240" />
          <path d="M460 260 C520 250, 540 245, 580 240" />
        </g>

        {/* nodes */}
        <g className="edl-nodes-pulse">
          {[
            [90, 120],
            [90, 240],
            [90, 360],
            [280, 90],
            [280, 170],
            [280, 250],
            [460, 180],
            [460, 220],
            [460, 260],
            [580, 240],
          ].map(([cx, cy], i) => (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={i === 9 ? 14 : 11}
              className="fill-teal-700 stroke-[var(--edl-bg)] dark:fill-teal-400"
              strokeWidth="3"
              style={{ animationDelay: `${i * 0.08}s` }}
            />
          ))}
        </g>

        <text
          x="90"
          y="430"
          className="fill-stone-500 dark:fill-stone-400"
          style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
        >
          Input → Conv → … → Softmax
        </text>
      </svg>
    </div>
  );
}
