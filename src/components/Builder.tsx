"use client";

import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useEdgesState,
  useNodesState,
  useReactFlow,
  ReactFlowProvider,
  type Connection,
  type Edge,
  type Node,
  type OnSelectionChangeParams,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
} from "react";
import Link from "next/link";
import { LayerNode, type LayerFlowNode } from "@/components/LayerNode";
import { Palette } from "@/components/Palette";
import { ExplainPanel } from "@/components/ExplainPanel";
import { CodePanel } from "@/components/CodePanel";
import { SimulatePanel } from "@/components/SimulatePanel";
import { ThemeToggle } from "@/components/ThemeToggle";
import { useTheme } from "@/components/ThemeProvider";
import { propagateShapes } from "@/lib/shapes";
import { LAYER_DEFS } from "@/lib/layers";
import {
  createStarterGraph,
  fromProjectState,
  makeLayerData,
  toProjectState,
} from "@/lib/project";
import { getTemplate, type TemplateId } from "@/lib/templates";
import { generateCode } from "@/lib/codegen";
import {
  downloadProjectFile,
  loadProjectFile,
  readProjectFromLocation,
  writeProjectToUrl,
  downloadPythonFile,
} from "@/lib/persistence";
import { downloadColabProject } from "@/lib/export";
import type { LayerNodeData, LayerParams, LayerType } from "@/lib/types";

const nodeTypes = { layer: LayerNode };

function applyShapes(
  nodes: Node<LayerNodeData>[],
  edges: Edge[],
): Node<LayerNodeData>[] {
  const shapes = propagateShapes(
    nodes.map((n) => ({ id: n.id, data: n.data })),
    edges.map((e) => ({ source: e.source, target: e.target })),
  );
  return nodes.map((n) => {
    const s = shapes.get(n.id);
    if (!s) return n;
    const same =
      JSON.stringify(n.data.inputShape) === JSON.stringify(s.inputShape) &&
      JSON.stringify(n.data.outputShape) === JSON.stringify(s.outputShape) &&
      n.data.error === s.error;
    if (same) return n;
    return {
      ...n,
      data: {
        ...n.data,
        inputShape: s.inputShape,
        outputShape: s.outputShape,
        error: s.error,
      },
    };
  });
}

function loadInitialGraph() {
  const starter = createStarterGraph();
  const fromUrl = readProjectFromLocation();
  if (fromUrl) {
    const restored = fromProjectState(fromUrl);
    return {
      nodes: applyShapes(restored.nodes, restored.edges) as LayerFlowNode[],
      edges: restored.edges,
      mode: restored.mode,
    };
  }
  return {
    nodes: applyShapes(starter.nodes, starter.edges) as LayerFlowNode[],
    edges: starter.edges,
    mode: "beginner" as const,
  };
}

function BuilderInner() {
  const [initial] = useState(loadInitialGraph);
  const [nodes, setNodes, onNodesChange] = useNodesState<LayerFlowNode>(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const [mode, setMode] = useState<"beginner" | "research">(initial.mode);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<"explain" | "code" | "simulate">("explain");
  const [shareCopied, setShareCopied] = useState(false);
  const [downloadOpen, setDownloadOpen] = useState(false);
  const { screenToFlowPosition } = useReactFlow();
  const fileRef = useRef<HTMLInputElement>(null);
  const downloadMenuRef = useRef<HTMLDivElement>(null);
  const idCounter = useRef(0);
  const lastFp = useRef("");
  const shareResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const nodesFingerprint = nodes
    .map((n) => `${n.id}:${n.data.layerType}:${JSON.stringify(n.data.params)}`)
    .join("|");
  const edgesFingerprint = edges.map((e) => `${e.source}->${e.target}`).join("|");

  const structureKey = useMemo(
    () => `${nodesFingerprint}::${edgesFingerprint}`,
    [nodesFingerprint, edgesFingerprint],
  );

  useEffect(() => {
    if (structureKey === lastFp.current) return;
    lastFp.current = structureKey;
    setNodes((nds) => applyShapes(nds, edges) as LayerFlowNode[]);
  }, [structureKey, edges, setNodes]);

  useEffect(() => {
    const t = setTimeout(() => {
      writeProjectToUrl(toProjectState(nodes, edges, mode));
    }, 400);
    return () => clearTimeout(t);
  }, [nodes, edges, mode]);

  useEffect(() => {
    if (!downloadOpen) return;
    function onPointerDown(e: PointerEvent) {
      const target = e.target;
      if (
        downloadMenuRef.current &&
        target instanceof Element &&
        !downloadMenuRef.current.contains(target)
      ) {
        setDownloadOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setDownloadOpen(false);
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [downloadOpen]);

  const projectState = useMemo(
    () => toProjectState(nodes, edges, mode),
    [nodes, edges, mode],
  );

  function getPyTorchCode() {
    const { code, error } = generateCode(
      nodes.map((n) => ({ id: n.id, data: n.data })),
      edges.map((e) => ({ source: e.source, target: e.target })),
      { framework: "pytorch" },
    );
    return { code, error };
  }

  function handleDownloadJson() {
    downloadProjectFile(projectState);
    setDownloadOpen(false);
  }

  function handleDownloadPython() {
    const { code, error } = getPyTorchCode();
    if (!code) {
      alert(error ?? "Could not generate Python for this graph.");
      return;
    }
    downloadPythonFile(code, "fashion_mnist_net.py");
    setDownloadOpen(false);
  }

  function handleDownloadColab() {
    const result = downloadColabProject(projectState);
    if (!result.ok) {
      alert(result.error);
      return;
    }
    setDownloadOpen(false);
  }

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => {
        const targetNode = nodes.find((n) => n.id === connection.target);
        const multiOk =
          targetNode && LAYER_DEFS[targetNode.data.layerType].allowsMultiInput;
        const existing = eds.filter((e) => e.target === connection.target);
        let cleaned = eds;
        if (!multiOk) {
          cleaned = eds.filter((e) => e.target !== connection.target);
        } else if (existing.length >= 2) {
          // keep newest two — drop oldest
          const dropId = existing[0]?.id;
          cleaned = eds.filter((e) => e.id !== dropId);
        }
        return addEdge(
          { ...connection, id: `e-${connection.source}-${connection.target}-${Date.now()}` },
          cleaned,
        );
      });
    },
    [setEdges, nodes],
  );

  const onSelectionChange = useCallback((params: OnSelectionChangeParams) => {
    const n = params.nodes[0];
    setSelectedId(n?.id ?? null);
    if (n) setRightTab("explain");
  }, []);

  const selectedNode = nodes.find((n) => n.id === selectedId) ?? null;

  const addLayer = useCallback(
    (layerType: LayerType, position?: { x: number; y: number }) => {
      idCounter.current += 1;
      const id = `${layerType.toLowerCase()}-${Date.now()}-${idCounter.current}`;
      const pos = position ?? {
        x: 260 + (idCounter.current % 5) * 20,
        y: 120 + idCounter.current * 36,
      };
      const newNode: LayerFlowNode = {
        id,
        type: "layer",
        position: pos,
        data: makeLayerData(layerType),
      };
      lastFp.current = "";
      setNodes((nds) => applyShapes([...nds, newNode], edges) as LayerFlowNode[]);
      setSelectedId(id);
      setRightTab("explain");
    },
    [edges, setNodes],
  );

  const onDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      const layerType = e.dataTransfer.getData("application/edl-layer") as LayerType;
      if (!layerType) return;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      addLayer(layerType, position);
    },
    [addLayer, screenToFlowPosition],
  );

  const updateSelectedParams = useCallback(
    (params: LayerParams) => {
      if (!selectedId) return;
      lastFp.current = "";
      setNodes((nds) => {
        const next = nds.map((n) =>
          n.id === selectedId ? { ...n, data: { ...n.data, params } } : n,
        );
        return applyShapes(next, edges) as LayerFlowNode[];
      });
    },
    [selectedId, edges, setNodes],
  );

  const resetStarter = () => {
    const g = createStarterGraph();
    lastFp.current = "";
    setMode("beginner");
    setNodes(applyShapes(g.nodes, g.edges) as LayerFlowNode[]);
    setEdges(g.edges);
    setSelectedId(null);
  };

  const loadTemplate = (id: TemplateId) => {
    const t = getTemplate(id);
    if (!t) return;
    const g = t.build();
    lastFp.current = "";
    setNodes(applyShapes(g.nodes, g.edges) as LayerFlowNode[]);
    setEdges(g.edges);
    setSelectedId(null);
  };

  const errorCount = nodes.filter((n) => n.data.error).length;
  const { theme } = useTheme();
  const isDark = theme === "dark";

  return (
    <div className="flex h-dvh flex-col bg-[var(--edl-bg)] text-[var(--edl-ink)]">
      <header className="flex shrink-0 flex-wrap items-center gap-3 border-b border-[var(--edl-border)] bg-[var(--edl-surface)] px-4 py-2.5">
        <div className="mr-2">
          <Link
            href="/"
            className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight text-[var(--edl-ink)]"
          >
            easy deep learning
          </Link>
          <p className="text-[11px] text-stone-500 dark:text-stone-400">
            Visual PyTorch builder · Fashion-MNIST · no account
          </p>
        </div>

        <div className="flex rounded-md border border-[var(--edl-border)] p-0.5 text-[12px]">
          <button
            type="button"
            onClick={() => setMode("beginner")}
            className={`rounded px-2.5 py-1 font-medium ${
              mode === "beginner"
                ? "bg-zinc-100 text-zinc-950"
                : "text-stone-400 hover:bg-[var(--edl-surface-2)]"
            }`}
          >
            Beginner
          </button>
          <button
            type="button"
            onClick={() => setMode("research")}
            className={`rounded px-2.5 py-1 font-medium ${
              mode === "research"
                ? "bg-zinc-200 text-zinc-950"
                : "text-stone-400 hover:bg-[var(--edl-surface-2)]"
            }`}
          >
            Research
          </button>
        </div>

        {errorCount > 0 && (
          <span className="rounded-full bg-rose-100 px-2.5 py-0.5 text-[11px] font-medium text-rose-800 dark:bg-rose-950 dark:text-rose-300">
            {errorCount} shape {errorCount === 1 ? "issue" : "issues"}
          </span>
        )}

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <ThemeToggle />
          <button
            type="button"
            onClick={resetStarter}
            className="rounded border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2.5 py-1 text-[11px] font-medium text-stone-700 hover:bg-[var(--edl-surface-2)] dark:text-stone-300"
          >
            Reset starter CNN
          </button>
          <div className="relative" ref={downloadMenuRef}>
            <button
              type="button"
              onClick={() => setDownloadOpen((o) => !o)}
              aria-expanded={downloadOpen}
              aria-haspopup="menu"
              className="rounded border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2.5 py-1 text-[11px] font-medium text-stone-700 hover:bg-[var(--edl-surface-2)] dark:text-stone-300"
            >
              Download project
            </button>
            {downloadOpen && (
              <div
                role="menu"
                className="absolute right-0 z-50 mt-1 w-56 overflow-hidden rounded-md border border-[var(--edl-border)] bg-[var(--edl-surface)] py-1 shadow-lg"
              >
                <button
                  type="button"
                  role="menuitem"
                  onClick={handleDownloadJson}
                  className="flex w-full flex-col items-start px-3 py-2 text-left hover:bg-[var(--edl-surface-2)]"
                >
                  <span className="text-[12px] font-medium text-[var(--edl-ink)]">
                    JSON project
                  </span>
                  <span className="text-[10px] text-stone-500">
                    Re-import into the builder
                  </span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={handleDownloadPython}
                  className="flex w-full flex-col items-start px-3 py-2 text-left hover:bg-[var(--edl-surface-2)]"
                >
                  <span className="text-[12px] font-medium text-[var(--edl-ink)]">
                    Python (.py)
                  </span>
                  <span className="text-[10px] text-stone-500">
                    Annotated PyTorch model file
                  </span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={handleDownloadColab}
                  className="flex w-full flex-col items-start px-3 py-2 text-left hover:bg-[var(--edl-surface-2)]"
                >
                  <span className="text-[12px] font-medium text-[var(--edl-ink)]">
                    Google Colab (.ipynb)
                  </span>
                  <span className="text-[10px] text-stone-500">
                    One notebook — cell per layer block
                  </span>
                </button>
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="rounded border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2.5 py-1 text-[11px] font-medium text-stone-700 hover:bg-[var(--edl-surface-2)] dark:text-stone-300"
          >
            Import project
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              try {
                const state = await loadProjectFile(file);
                const restored = fromProjectState(state);
                lastFp.current = "";
                setMode(restored.mode);
                setNodes(
                  applyShapes(restored.nodes, restored.edges) as LayerFlowNode[],
                );
                setEdges(restored.edges);
              } catch (err) {
                alert(
                  err instanceof Error ? err.message : "Failed to load project",
                );
              }
              e.target.value = "";
            }}
          />
          <button
            type="button"
            onClick={() => {
              writeProjectToUrl(toProjectState(nodes, edges, mode));
              void navigator.clipboard.writeText(window.location.href);
              setShareCopied(true);
              if (shareResetRef.current) clearTimeout(shareResetRef.current);
              shareResetRef.current = setTimeout(() => setShareCopied(false), 1400);
            }}
            className={`relative inline-flex h-7 w-[7.75rem] shrink-0 items-center justify-center rounded bg-zinc-100 text-[11px] font-medium text-zinc-950 transition-colors hover:bg-white ${
              shareCopied ? "edl-btn-clicked bg-emerald-300 hover:bg-emerald-300" : ""
            }`}
          >
            <span className={shareCopied ? "invisible" : undefined}>
              Copy share link
            </span>
            <span
              className={`absolute inset-0 flex items-center justify-center ${
                shareCopied ? "" : "invisible"
              }`}
              aria-live="polite"
            >
              Link copied
            </span>
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <Palette
          mode={mode}
          onDragStart={() => undefined}
          onAdd={(t) => addLayer(t)}
          onLoadTemplate={loadTemplate}
        />

        <div className="relative min-w-0 flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onSelectionChange={onSelectionChange}
            onDragOver={onDragOver}
            onDrop={onDrop}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            deleteKeyCode={["Backspace", "Delete"]}
            proOptions={{ hideAttribution: true }}
            className="bg-[var(--edl-canvas)]"
            colorMode={isDark ? "dark" : "light"}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={18}
              size={1}
              color={isDark ? "#2a3531" : "#d6d3d1"}
            />
            <Controls showInteractive={false} />
            <MiniMap
              nodeColor={() => (isDark ? "#a1a1aa" : "#52525b")}
              maskColor={isDark ? "rgba(0,0,0,0.35)" : "rgba(28,25,23,0.1)"}
              className="!bg-[var(--edl-surface)]/80"
            />
          </ReactFlow>

          {mode === "research" && (
            <div className="pointer-events-none absolute left-3 top-3 max-w-sm rounded-md border border-[var(--edl-border)] bg-[var(--edl-surface)]/95 px-3 py-2 text-[11px] leading-relaxed text-[var(--edl-ink)] shadow-sm">
              <strong className="font-semibold">Research mode:</strong> drop a
              Loop Block to reuse shared weights N times — the code generator
              emits a{" "}
              <code className="rounded bg-zinc-800 px-1 font-mono dark:bg-zinc-100">
                for _ in range(N)
              </code>{" "}
              over one module.
            </div>
          )}
        </div>

        <aside className="flex w-[380px] shrink-0 flex-col border-l border-[var(--edl-border)] bg-[var(--edl-surface)]">
          <div className="flex border-b border-[var(--edl-border)] text-[12px]">
            <button
              type="button"
              onClick={() => setRightTab("explain")}
              className={`flex-1 px-3 py-2 font-medium ${
                rightTab === "explain"
                  ? "border-b-2 border-zinc-400 text-[var(--edl-ink)]"
                  : "text-stone-500 dark:text-stone-400"
              }`}
            >
              Explain
            </button>
            <button
              type="button"
              onClick={() => setRightTab("simulate")}
              className={`flex-1 px-3 py-2 font-medium ${
                rightTab === "simulate"
                  ? "border-b-2 border-zinc-400 text-[var(--edl-ink)]"
                  : "text-stone-500 dark:text-stone-400"
              }`}
            >
              Simulate
            </button>
            <button
              type="button"
              onClick={() => setRightTab("code")}
              className={`flex-1 px-3 py-2 font-medium ${
                rightTab === "code"
                  ? "border-b-2 border-zinc-400 text-[var(--edl-ink)]"
                  : "text-stone-500 dark:text-stone-400"
              }`}
            >
              Code
            </button>
          </div>
          <div className="min-h-0 flex-1">
            {rightTab === "explain" ? (
              <ExplainPanel
                data={selectedNode?.data ?? null}
                onChangeParams={updateSelectedParams}
              />
            ) : rightTab === "simulate" ? (
              <SimulatePanel
                nodes={nodes.map((n) => ({ id: n.id, data: n.data }))}
                edges={edges.map((e) => ({
                  source: e.source,
                  target: e.target,
                }))}
              />
            ) : (
              <CodePanel
                nodes={nodes.map((n) => ({ id: n.id, data: n.data }))}
                edges={edges.map((e) => ({
                  source: e.source,
                  target: e.target,
                }))}
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

export function Builder() {
  return (
    <ReactFlowProvider>
      <BuilderInner />
    </ReactFlowProvider>
  );
}
