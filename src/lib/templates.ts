import type { Edge, Node } from "@xyflow/react";
import { LAYER_DEFS, inputShapeFromParams } from "./layers";
import type { LayerNodeData, LayerType } from "./types";

export type TemplateId =
  | "simple-mlp"
  | "cnn"
  | "rnn-lstm"
  | "transformer-bert"
  | "autoencoder"
  | "text-classifier"
  | "resnet-block"
  | "fashion-cnn"
  | "lenet";

export type ArchitectureTemplate = {
  id: TemplateId;
  name: string;
  description: string;
  tags: string[];
  icon: string;
  color: string;
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

function yAt(i: number, step = 100) {
  return i * step;
}

export const TEMPLATES: ArchitectureTemplate[] = [
  {
    id: "simple-mlp",
    name: "Simple MLP",
    description: "Multi-layer perceptron for tabular data.",
    tags: ["beginner", "mlp"],
    icon: "mlp",
    color: "#a78bfa",
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: yAt(0), params: { inChannels: 1, height: 28, width: 28 } },
        { id: "flat", type: "Flatten", x: 280, y: yAt(1) },
        { id: "fc1", type: "Linear", x: 280, y: yAt(2), params: { outFeatures: 256 } },
        { id: "relu1", type: "ReLU", x: 280, y: yAt(3) },
        { id: "out", type: "Output", x: 280, y: yAt(4), params: { numClasses: 10 } },
      ]),
  },
  {
    id: "cnn",
    name: "CNN",
    description: "For image classification.",
    tags: ["vision"],
    icon: "cnn",
    color: "#38bdf8",
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: yAt(0), params: { inChannels: 3, height: 224, width: 224 } },
        { id: "conv1", type: "Conv2d", x: 280, y: yAt(1), params: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 } },
        { id: "bn1", type: "BatchNorm2d", x: 280, y: yAt(2) },
        { id: "relu1", type: "ReLU", x: 280, y: yAt(3) },
        { id: "pool1", type: "MaxPool2d", x: 280, y: yAt(4) },
        { id: "conv2", type: "Conv2d", x: 280, y: yAt(5), params: { outChannels: 128, kernelSize: 3, stride: 1, padding: 1 } },
        { id: "drop", type: "Dropout", x: 280, y: yAt(6), params: { p: 0.3 } },
        { id: "flat", type: "Flatten", x: 280, y: yAt(7) },
        { id: "out", type: "Output", x: 280, y: yAt(8), params: { numClasses: 10 } },
      ]),
  },
  {
    id: "rnn-lstm",
    name: "RNN (LSTM)",
    description: "For sequence & time-series.",
    tags: ["sequence"],
    icon: "lstm",
    color: "#4ade80",
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: yAt(0), params: { inChannels: 0, height: 1, width: 128 } },
        { id: "emb", type: "Embedding", x: 280, y: yAt(1), params: { numEmbeddings: 1000, embeddingDim: 64 } },
        { id: "lstm", type: "LSTM", x: 280, y: yAt(2), params: { hiddenSize: 128, numLayers: 1, bidirectional: 0 } },
        { id: "flat", type: "Flatten", x: 280, y: yAt(3) },
        { id: "fc1", type: "Linear", x: 280, y: yAt(4), params: { outFeatures: 64 } },
        { id: "out", type: "Output", x: 280, y: yAt(5), params: { numClasses: 10 } },
      ]),
  },
  {
    id: "transformer-bert",
    name: "Transformer (BERT-style)",
    description: "Encoder-only for classification.",
    tags: ["sequence", "transformer"],
    icon: "transformer",
    color: "#fbbf24",
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: yAt(0), params: { inChannels: 0, height: 1, width: 64 } },
        { id: "emb", type: "Embedding", x: 280, y: yAt(1), params: { numEmbeddings: 1000, embeddingDim: 64 } },
        { id: "pe", type: "PositionalEncoding", x: 280, y: yAt(2), params: { maxLen: 512 } },
        { id: "enc", type: "TransformerEncoder", x: 280, y: yAt(3), params: { embedDim: 64, numHeads: 4, ffDim: 256, numLayers: 2 } },
        { id: "flat", type: "Flatten", x: 280, y: yAt(4) },
        { id: "out", type: "Output", x: 280, y: yAt(5), params: { numClasses: 2 } },
      ]),
  },
  {
    id: "autoencoder",
    name: "Autoencoder",
    description: "For dimensionality reduction.",
    tags: ["mlp"],
    icon: "ae",
    color: "#fb7185",
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: yAt(0), params: { inChannels: 1, height: 28, width: 28 } },
        { id: "flat", type: "Flatten", x: 280, y: yAt(1) },
        { id: "enc1", type: "Linear", x: 280, y: yAt(2), params: { outFeatures: 128 } },
        { id: "relu1", type: "ReLU", x: 280, y: yAt(3) },
        { id: "bottleneck", type: "Linear", x: 280, y: yAt(4), params: { outFeatures: 32 } },
        { id: "relu2", type: "ReLU", x: 280, y: yAt(5) },
        { id: "dec1", type: "Linear", x: 280, y: yAt(6), params: { outFeatures: 784 } },
      ]),
  },
  {
    id: "text-classifier",
    name: "Text Classifier",
    description: "For sentiment & NLP.",
    tags: ["nlp", "sequence"],
    icon: "text",
    color: "#34d399",
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: yAt(0), params: { inChannels: 0, height: 1, width: 128 } },
        { id: "emb", type: "Embedding", x: 280, y: yAt(1), params: { numEmbeddings: 5000, embeddingDim: 64 } },
        { id: "lstm", type: "LSTM", x: 280, y: yAt(2), params: { hiddenSize: 128, numLayers: 1, bidirectional: 1 } },
        { id: "drop", type: "Dropout", x: 280, y: yAt(3), params: { p: 0.3 } },
        { id: "flat", type: "Flatten", x: 280, y: yAt(4) },
        { id: "out", type: "Output", x: 280, y: yAt(5), params: { numClasses: 2 } },
      ]),
  },
  {
    id: "resnet-block",
    name: "ResNet Block",
    description: "CNN with skip connections.",
    tags: ["vision", "residual"],
    icon: "resnet",
    color: "#f87171",
    build: () => ({
      nodes: [
        {
          id: "input",
          type: "layer",
          position: { x: 200, y: 0 },
          data: data("Input", { inChannels: 3, height: 32, width: 32 }),
          deletable: false,
        },
        {
          id: "stem",
          type: "layer",
          position: { x: 200, y: 100 },
          data: data("Conv2d", { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 }),
        },
        {
          id: "stem_relu",
          type: "layer",
          position: { x: 200, y: 200 },
          data: data("ReLU"),
        },
        {
          id: "conv1",
          type: "layer",
          position: { x: 80, y: 320 },
          data: data("Conv2d", { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 }),
        },
        {
          id: "bn1",
          type: "layer",
          position: { x: 80, y: 420 },
          data: data("BatchNorm2d"),
        },
        {
          id: "relu1",
          type: "layer",
          position: { x: 80, y: 520 },
          data: data("ReLU"),
        },
        {
          id: "conv2",
          type: "layer",
          position: { x: 80, y: 620 },
          data: data("Conv2d", { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 }),
        },
        {
          id: "bn2",
          type: "layer",
          position: { x: 80, y: 720 },
          data: data("BatchNorm2d"),
        },
        {
          id: "add",
          type: "layer",
          position: { x: 200, y: 840 },
          data: data("Add"),
        },
      ],
      edges: [
        { id: "e1", source: "input", target: "stem" },
        { id: "e2", source: "stem", target: "stem_relu" },
        { id: "e3", source: "stem_relu", target: "conv1" },
        { id: "e4", source: "conv1", target: "bn1" },
        { id: "e5", source: "bn1", target: "relu1" },
        { id: "e6", source: "relu1", target: "conv2" },
        { id: "e7", source: "conv2", target: "bn2" },
        { id: "e8", source: "bn2", target: "add" },
        { id: "e9", source: "stem_relu", target: "add" },
      ],
    }),
  },
  {
    id: "fashion-cnn",
    name: "Fashion-MNIST CNN",
    description: "Starter teaching CNN for 28×28 grayscale.",
    tags: ["beginner", "vision"],
    icon: "cnn",
    color: "#38bdf8",
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: yAt(0), params: { inChannels: 1, height: 28, width: 28 } },
        { id: "conv1", type: "Conv2d", x: 280, y: yAt(1), params: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 } },
        { id: "relu1", type: "ReLU", x: 280, y: yAt(2) },
        { id: "pool1", type: "MaxPool2d", x: 280, y: yAt(3) },
        { id: "flat", type: "Flatten", x: 280, y: yAt(4) },
        { id: "fc1", type: "Linear", x: 280, y: yAt(5), params: { outFeatures: 128 } },
        { id: "relu2", type: "ReLU", x: 280, y: yAt(6) },
        { id: "drop", type: "Dropout", x: 280, y: yAt(7), params: { p: 0.25 } },
        { id: "out", type: "Output", x: 280, y: yAt(8), params: { numClasses: 10 } },
      ]),
  },
  {
    id: "lenet",
    name: "LeNet-style",
    description: "Classic small CNN inspired by LeNet.",
    tags: ["vision", "classic"],
    icon: "cnn",
    color: "#38bdf8",
    build: () =>
      chain([
        { id: "input", type: "Input", x: 280, y: yAt(0), params: { inChannels: 1, height: 28, width: 28 } },
        { id: "conv1", type: "Conv2d", x: 280, y: yAt(1), params: { outChannels: 6, kernelSize: 5, stride: 1, padding: 0 } },
        { id: "relu1", type: "ReLU", x: 280, y: yAt(2) },
        { id: "pool1", type: "AvgPool2d", x: 280, y: yAt(3) },
        { id: "conv2", type: "Conv2d", x: 280, y: yAt(4), params: { outChannels: 16, kernelSize: 5, stride: 1, padding: 0 } },
        { id: "relu2", type: "ReLU", x: 280, y: yAt(5) },
        { id: "pool2", type: "AvgPool2d", x: 280, y: yAt(6) },
        { id: "flat", type: "Flatten", x: 280, y: yAt(7) },
        { id: "fc1", type: "Linear", x: 280, y: yAt(8), params: { outFeatures: 120 } },
        { id: "out", type: "Output", x: 280, y: yAt(9), params: { numClasses: 10 } },
      ]),
  },
];

/** Templates shown in the NeuralFlows-style Templates panel. */
export const SIDEBAR_TEMPLATES: TemplateId[] = [
  "simple-mlp",
  "cnn",
  "rnn-lstm",
  "transformer-bert",
  "autoencoder",
  "text-classifier",
  "resnet-block",
];

export function getTemplate(id: TemplateId): ArchitectureTemplate | undefined {
  return TEMPLATES.find((t) => t.id === id);
}

export function countTemplateNodes(id: TemplateId): number {
  return getTemplate(id)?.build().nodes.length ?? 0;
}
