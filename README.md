# easy deep learning

**Visual PyTorch Builder** — a Next.js drag-and-drop neural network canvas that teaches how layers work while generating real, annotated PyTorch code. No accounts, no backend, no training hosted here.

## What it does

- Build a Fashion-MNIST image classifier from beginner layers (Conv2d, MaxPool, Flatten, Linear, ReLU, Dropout, Softmax)
- Live shape checking with plain-English errors
- Per-layer explanations (what / why / analogy) plus editable params
- Annotated `nn.Module` code generation (comments explain *why*)
- Export: copy code, download `.py`, download `.ipynb`, or open Colab (downloads notebook + opens Colab upload)
- Share via URL hash (compressed JSON) or download/import a project file
- **Research mode**: Loop Block → shared-weight `for _ in range(N)` (looped / recurrent-depth idea)

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3456](http://localhost:3456) (port **3456** so it doesn’t collide with other apps on 3000).

- `/` — landing page
- `/builder` — visual PyTorch builder

Light/dark mode toggles from the header on both pages (saved in `localStorage`).

## Project shape

This is a **Next.js App Router** app:

- `src/app/page.tsx` — landing
- `src/app/builder/page.tsx` — builder UI
- `src/app/layout.tsx` — root layout / fonts / theme
- `src/components/*` — landing, React Flow canvas, palette, explain + code panels
- `src/lib/*` — layers, shape inference, codegen, persistence, Colab export

There is **no `/api`** — the product is backend-free (state in the browser / URL / files; Colab runs the code).

## Deploy

Deploy to Vercel from this repo. No env vars or database required.

## Out of scope (by design)

- No in-browser or server-side training / code execution
- Verification is offloaded to Google Colab or the user’s own machine

## Stack

Next.js · React · React Flow · Tailwind CSS · TypeScript · lz-string (URL state)
