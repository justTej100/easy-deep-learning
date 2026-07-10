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
import { ThemeToggle } from "@/components/ThemeToggle";
import { useTheme } from "@/components/ThemeProvider";
import { propagateShapes } from "@/lib/shapes";
import {
  createStarterGraph,
  fromProjectState,
  makeLayerData,
  toProjectState,
} from "@/lib/project";
import {
  downloadProjectFile,
  loadProjectFile,
  readProjectFromLocation,
  writeProjectToUrl,
} from "@/lib/persistence";
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
  const [rightTab, setRightTab] = useState<"explain" | "code">("explain");
  const { screenToFlowPosition } = useReactFlow();
  const fileRef = useRef<HTMLInputElement>(null);
  const idCounter = useRef(0);
  const lastFp = useRef("");

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

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => {
        const cleaned = eds.filter((e) => e.target !== connection.target);
        return addEdge(
          { ...connection, id: `e-${connection.source}-${connection.target}` },
          cleaned,
        );
      });
    },
    [setEdges],
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

  const errorCount = nodes.filter((n) => n.data.error).length;
  const { theme } = useTheme();
  const isDark = theme === "dark";

  return (
    <div className="flex h-dvh flex-col bg-[var(--edl-bg)] text-[var(--edl-ink)]">
      <header className="flex shrink-0 flex-wrap items-center gap-3 border-b border-[var(--edl-border)] bg-[var(--edl-surface)] px-4 py-2.5">
        <div className="mr-2">
          <Link
            href="/"
            className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight text-teal-900 dark:text-teal-300"
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
                ? "bg-teal-800 text-white dark:bg-teal-600"
                : "text-stone-600 hover:bg-[var(--edl-surface-2)] dark:text-stone-300"
            }`}
          >
            Beginner
          </button>
          <button
            type="button"
            onClick={() => setMode("research")}
            className={`rounded px-2.5 py-1 font-medium ${
              mode === "research"
                ? "bg-cyan-800 text-white dark:bg-cyan-700"
                : "text-stone-600 hover:bg-[var(--edl-surface-2)] dark:text-stone-300"
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
          <button
            type="button"
            onClick={() =>
              downloadProjectFile(toProjectState(nodes, edges, mode))
            }
            className="rounded border border-[var(--edl-border)] bg-[var(--edl-surface)] px-2.5 py-1 text-[11px] font-medium text-stone-700 hover:bg-[var(--edl-surface-2)] dark:text-stone-300"
          >
            Download project
          </button>
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
            }}
            className="rounded bg-teal-800 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-teal-700 dark:bg-teal-600 dark:hover:bg-teal-500"
          >
            Copy share link
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <Palette
          mode={mode}
          onDragStart={() => undefined}
          onAdd={(t) => addLayer(t)}
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
              nodeColor={() => (isDark ? "#2dd4bf" : "#0f766e")}
              maskColor={isDark ? "rgba(0,0,0,0.35)" : "rgba(28,25,23,0.1)"}
              className="!bg-[var(--edl-surface)]/80"
            />
          </ReactFlow>

          {mode === "research" && (
            <div className="pointer-events-none absolute left-3 top-3 max-w-sm rounded-md border border-cyan-200 bg-cyan-50/95 px-3 py-2 text-[11px] leading-relaxed text-cyan-950 shadow-sm dark:border-cyan-800 dark:bg-cyan-950/90 dark:text-cyan-100">
              <strong className="font-semibold">Research mode:</strong> drop a
              Loop Block to reuse shared weights N times — the code generator
              emits a{" "}
              <code className="rounded bg-cyan-100 px-1 font-mono dark:bg-cyan-900">
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
                  ? "border-b-2 border-teal-800 text-teal-900 dark:border-teal-400 dark:text-teal-300"
                  : "text-stone-500 dark:text-stone-400"
              }`}
            >
              Explain
            </button>
            <button
              type="button"
              onClick={() => setRightTab("code")}
              className={`flex-1 px-3 py-2 font-medium ${
                rightTab === "code"
                  ? "border-b-2 border-teal-800 text-teal-900 dark:border-teal-400 dark:text-teal-300"
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
