# easy deep learning

**Visual PyTorch Builder** — a Next.js drag-and-drop neural network canvas that teaches how layers work while generating real, annotated PyTorch code. No accounts, no backend, no training hosted here.

## What it does

- Build a Fashion-MNIST image classifier from beginner layers (Conv2d, MaxPool, Flatten, Linear, ReLU, Dropout, Softmax)
- Expanded catalog: BatchNorm, LSTM/GRU, Attention, Add/Concat, and more
- Live shape checking with plain-English errors + forward-pass Simulate tab
- Per-layer explanations (what / why / analogy) plus editable params
- Annotated code generation for **PyTorch** and **Keras**
- Export: copy code, download `.py`, download `.ipynb`, or open Colab
- Architecture templates (CNN, MLP, LeNet-style, LSTM, …)
- Share via URL hash (compressed JSON) or download/import a project file
- **Research mode**: Loop Block → shared-weight `for _ in range(N)`

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

## Future ideas

- **Google Docs–style live collaborative editor** — multiple people editing the same network canvas in real time (cursors, shared graph state, presence). Not built yet; may land in a later update if we add a lightweight realtime backend.

## Stack

Next.js · React · React Flow · Tailwind CSS · TypeScript · lz-string (URL state)
