import type { Edge, Node } from "@xyflow/react";
import { LAYER_DEFS, inputShapeFromParams } from "./layers";
import type { LayerNodeData, LayerType } from "./types";

export type TemplateId =
  | "fashion-cnn"
  | "image-classifier"
  | "mlp"
  | "lenet"
  | "lstm-classifier";

export type ArchitectureTemplate = {
  id: TemplateId;
  name: string;
  description: string;
  tags: string[];
  build: () => { nodes: Node<LayerNodeData>[]; edges: Edge[] };
};

function data(type: LayerType, params?: LayerNodeData["params"]): LayerNodeData {
  const def = LAYER_DEFS[type];
  const merged = { ...def.defaultParams, ...params };
  return {
    layerType: type,
    label: def.shortLabel,
    params: merged,
    inputShape: null,
    outputShape: type === "Input" ? inputShapeFromParams(merged) : null,
    error: null,
  };
}

function chain(
  layers: Array<{
    id: string;
    type: LayerType;
    x: number;
    y: number;
    params?: LayerNodeData["params"];
  }>,
): { nodes: Node<LayerNodeData>[]; edges: Edge[] } {
  const nodes: Node<LayerNodeData>[] = layers.map((l) => ({
    id: l.id,
    type: "layer",
    position: { x: l.x, y: l.y },
    data: data(l.type, l.params),
    deletable: l.type !== "Input" && l.type !== "Output",
  }));
  const edges: Edge[] = [];
  for (let i = 0; i < layers.length - 1; i++) {
    edges.push({
      id: `e-${layers[i].id}-${layers[i + 1].id}`,
      source: layers[i].id,
      target: layers[i + 1].id,
    });
  }
  return { nodes, edges };
}

export const TEMPLATES: ArchitectureTemplate[] = [
  {
    id: "fashion-cnn",
    name: "Fashion-MNIST CNN",
    description: "Starter teaching CNN for 28×28 grayscale clothing classification.",
    tags: ["beginner", "vision"],
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: 0, params: { inChannels: 1, height: 28, width: 28 } },
        { id: "conv1", type: "Conv2d", x: 280, y: 100, params: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 } },
        { id: "relu1", type: "ReLU", x: 280, y: 200 },
        { id: "pool1", type: "MaxPool2d", x: 280, y: 300 },
        { id: "flat", type: "Flatten", x: 280, y: 400 },
        { id: "fc1", type: "Linear", x: 280, y: 500, params: { outFeatures: 128 } },
        { id: "relu2", type: "ReLU", x: 280, y: 600 },
        { id: "drop", type: "Dropout", x: 280, y: 700, params: { p: 0.25 } },
        { id: "out", type: "Output", x: 280, y: 800, params: { numClasses: 10 } },
      ]),
  },
  {
    id: "image-classifier",
    name: "Image Classifier (CNN)",
    description: "RGB CNN with BatchNorm — inspired by NeuralFlows’ template (224×224 → 10).",
    tags: ["vision"],
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: 0, params: { inChannels: 3, height: 224, width: 224 } },
        { id: "conv1", type: "Conv2d", x: 280, y: 100, params: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 } },
        { id: "bn1", type: "BatchNorm2d", x: 280, y: 200 },
        { id: "relu1", type: "ReLU", x: 280, y: 300 },
        { id: "pool1", type: "MaxPool2d", x: 280, y: 400 },
        { id: "conv2", type: "Conv2d", x: 280, y: 500, params: { outChannels: 128, kernelSize: 3, stride: 1, padding: 1 } },
        { id: "relu2", type: "ReLU", x: 280, y: 600 },
        { id: "drop", type: "Dropout", x: 280, y: 700, params: { p: 0.3 } },
        { id: "flat", type: "Flatten", x: 280, y: 800 },
        { id: "fc1", type: "Linear", x: 280, y: 900, params: { outFeatures: 256 } },
        { id: "relu3", type: "ReLU", x: 280, y: 1000 },
        { id: "out", type: "Output", x: 280, y: 1100, params: { numClasses: 10 } },
      ]),
  },
  {
    id: "mlp",
    name: "Deep MLP",
    description: "Flatten → Dense stack for Fashion-MNIST.",
    tags: ["beginner", "mlp"],
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: 0, params: { inChannels: 1, height: 28, width: 28 } },
        { id: "flat", type: "Flatten", x: 280, y: 100 },
        { id: "fc1", type: "Linear", x: 280, y: 200, params: { outFeatures: 256 } },
        { id: "relu1", type: "ReLU", x: 280, y: 300 },
        { id: "drop1", type: "Dropout", x: 280, y: 400, params: { p: 0.3 } },
        { id: "fc2", type: "Linear", x: 280, y: 500, params: { outFeatures: 128 } },
        { id: "relu2", type: "ReLU", x: 280, y: 600 },
        { id: "out", type: "Output", x: 280, y: 700, params: { numClasses: 10 } },
      ]),
  },
  {
    id: "lenet",
    name: "LeNet-style",
    description: "Classic small CNN inspired by LeNet.",
    tags: ["vision", "classic"],
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: 0, params: { inChannels: 1, height: 28, width: 28 } },
        { id: "conv1", type: "Conv2d", x: 280, y: 100, params: { outChannels: 6, kernelSize: 5, stride: 1, padding: 0 } },
        { id: "relu1", type: "ReLU", x: 280, y: 200 },
        { id: "pool1", type: "AvgPool2d", x: 280, y: 300 },
        { id: "conv2", type: "Conv2d", x: 280, y: 400, params: { outChannels: 16, kernelSize: 5, stride: 1, padding: 0 } },
        { id: "relu2", type: "ReLU", x: 280, y: 500 },
        { id: "pool2", type: "AvgPool2d", x: 280, y: 600 },
        { id: "flat", type: "Flatten", x: 280, y: 700 },
        { id: "fc1", type: "Linear", x: 280, y: 800, params: { outFeatures: 120 } },
        { id: "relu3", type: "ReLU", x: 280, y: 900 },
        { id: "fc2", type: "Linear", x: 280, y: 1000, params: { outFeatures: 84 } },
        { id: "relu4", type: "ReLU", x: 280, y: 1100 },
        { id: "out", type: "Output", x: 280, y: 1200, params: { numClasses: 10 } },
      ]),
  },
  {
    id: "lstm-classifier",
    name: "LSTM Classifier",
    description: "Token sequence → Embedding → LSTM → Dense head.",
    tags: ["sequence", "nlp"],
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: 0, params: { inChannels: 0, height: 1, width: 128 } },
        { id: "emb", type: "Embedding", x: 280, y: 100, params: { numEmbeddings: 1000, embeddingDim: 64 } },
        { id: "lstm", type: "LSTM", x: 280, y: 200, params: { hiddenSize: 128, numLayers: 1, bidirectional: 0 } },
        { id: "flat", type: "Flatten", x: 280, y: 300 },
        { id: "fc1", type: "Linear", x: 280, y: 400, params: { outFeatures: 64 } },
        { id: "relu1", type: "ReLU", x: 280, y: 500 },
        { id: "out", type: "Output", x: 280, y: 600, params: { numClasses: 10 } },
      ]),
  },
];

export function getTemplate(id: TemplateId): ArchitectureTemplate | undefined {
  return TEMPLATES.find((t) => t.id === id);
}
