import type { Edge, Node } from "@xyflow/react";
import type { LayerNodeData, ProjectState } from "./types";
import { FASHION_MNIST } from "./types";
import { LAYER_DEFS } from "./layers";

export function makeLayerData(
  layerType: LayerNodeData["layerType"],
  overrides?: Partial<LayerNodeData>,
): LayerNodeData {
  const def = LAYER_DEFS[layerType];
  return {
    layerType,
    label: def.shortLabel,
    params: { ...def.defaultParams },
    inputShape: null,
    outputShape: layerType === "Input" ? [...FASHION_MNIST.inputShape] : null,
    error: null,
    ...overrides,
  };
}

/** Starter: a sensible tiny CNN for Fashion-MNIST so the canvas isn't blank. */
export function createStarterGraph(): {
  nodes: Node<LayerNodeData>[];
  edges: Edge[];
} {
  const nodes: Node<LayerNodeData>[] = [
    {
      id: "input",
      type: "layer",
      position: { x: 280, y: 0 },
      data: makeLayerData("Input"),
      deletable: false,
    },
    {
      id: "conv1",
      type: "layer",
      position: { x: 280, y: 110 },
      data: makeLayerData("Conv2d", {
        params: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 },
      }),
    },
    {
      id: "relu1",
      type: "layer",
      position: { x: 280, y: 220 },
      data: makeLayerData("ReLU"),
    },
    {
      id: "pool1",
      type: "layer",
      position: { x: 280, y: 330 },
      data: makeLayerData("MaxPool2d"),
    },
    {
      id: "flat",
      type: "layer",
      position: { x: 280, y: 440 },
      data: makeLayerData("Flatten"),
    },
    {
      id: "fc1",
      type: "layer",
      position: { x: 280, y: 550 },
      data: makeLayerData("Linear", { params: { outFeatures: 128 } }),
    },
    {
      id: "relu2",
      type: "layer",
      position: { x: 280, y: 660 },
      data: makeLayerData("ReLU"),
    },
    {
      id: "drop",
      type: "layer",
      position: { x: 280, y: 770 },
      data: makeLayerData("Dropout", { params: { p: 0.25 } }),
    },
    {
      id: "out",
      type: "layer",
      position: { x: 280, y: 880 },
      data: makeLayerData("Output"),
      deletable: false,
    },
  ];

  const edges: Edge[] = [
    { id: "e1", source: "input", target: "conv1" },
    { id: "e2", source: "conv1", target: "relu1" },
    { id: "e3", source: "relu1", target: "pool1" },
    { id: "e4", source: "pool1", target: "flat" },
    { id: "e5", source: "flat", target: "fc1" },
    { id: "e6", source: "fc1", target: "relu2" },
    { id: "e7", source: "relu2", target: "drop" },
    { id: "e8", source: "drop", target: "out" },
  ];

  return { nodes, edges };
}

export function toProjectState(
  nodes: Node<LayerNodeData>[],
  edges: Edge[],
  mode: "beginner" | "research",
): ProjectState {
  return {
    version: 1,
    mode,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type ?? "layer",
      position: n.position,
      data: n.data,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
    })),
  };
}

export function fromProjectState(state: ProjectState): {
  nodes: Node<LayerNodeData>[];
  edges: Edge[];
  mode: "beginner" | "research";
} {
  return {
    mode: state.mode,
    nodes: state.nodes.map((n) => ({
      id: n.id,
      type: n.type || "layer",
      position: n.position,
      data: n.data,
      deletable: n.data.layerType !== "Input" && n.data.layerType !== "Output",
    })),
    edges: state.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? undefined,
      targetHandle: e.targetHandle ?? undefined,
    })),
  };
}
