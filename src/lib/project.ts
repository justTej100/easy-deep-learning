import type { Edge, Node } from "@xyflow/react";
import type { LayerNodeData, ProjectState } from "./types";
import { LAYER_DEFS, inputShapeFromParams } from "./layers";
import { getTemplate } from "./templates";

export function makeLayerData(
  layerType: LayerNodeData["layerType"],
  overrides?: Partial<LayerNodeData>,
): LayerNodeData {
  const def = LAYER_DEFS[layerType];
  const params = {
    ...def.defaultParams,
    ...(overrides?.params ?? {}),
  };
  return {
    layerType,
    label: overrides?.label ?? def.shortLabel,
    params,
    inputShape: overrides?.inputShape ?? null,
    outputShape:
      overrides?.outputShape ??
      (layerType === "Input" ? inputShapeFromParams(params) : null),
    error: overrides?.error ?? null,
    childIds: overrides?.childIds,
  };
}

/** Starter: Fashion-MNIST CNN template. */
export function createStarterGraph(): {
  nodes: Node<LayerNodeData>[];
  edges: Edge[];
} {
  return getTemplate("fashion-cnn")!.build();
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
