export type LayerType =
  | "Input"
  | "Conv2d"
  | "Conv1d"
  | "MaxPool2d"
  | "AvgPool2d"
  | "AdaptiveAvgPool2d"
  | "BatchNorm2d"
  | "LayerNorm"
  | "Flatten"
  | "Reshape"
  | "Linear"
  | "Embedding"
  | "LSTM"
  | "GRU"
  | "MultiheadAttention"
  | "ReLU"
  | "LeakyReLU"
  | "GELU"
  | "Sigmoid"
  | "Tanh"
  | "Dropout"
  | "Softmax"
  | "Add"
  | "Concat"
  | "Output"
  | "LoopBlock";

export type TensorShape = number[];

export type LayerParams = {
  // Conv
  outChannels?: number;
  kernelSize?: number;
  stride?: number;
  padding?: number;
  // Pool
  poolKernel?: number;
  poolStride?: number;
  // Adaptive pool
  outputSize?: number;
  // Linear / Dense
  outFeatures?: number;
  // Dropout
  p?: number;
  // Softmax / LayerNorm
  dim?: number;
  // LeakyReLU
  negativeSlope?: number;
  // Embedding
  numEmbeddings?: number;
  embeddingDim?: number;
  // LSTM / GRU
  hiddenSize?: number;
  numLayers?: number;
  bidirectional?: number; // 0 | 1 for number input
  // Attention
  numHeads?: number;
  embedDim?: number;
  // Reshape
  reshapeChannels?: number;
  reshapeHeight?: number;
  reshapeWidth?: number;
  // Concat
  concatDim?: number;
  // LoopBlock
  repeats?: number;
  // Input presets
  inChannels?: number;
  height?: number;
  width?: number;
  seqLen?: number;
  // Output
  numClasses?: number;
};

export type LayerNodeData = {
  layerType: LayerType;
  label: string;
  params: LayerParams;
  inputShape?: TensorShape | null;
  outputShape?: TensorShape | null;
  error?: string | null;
  childIds?: string[];
  [key: string]: unknown;
};

export type ProjectState = {
  version: 1;
  mode: "beginner" | "research";
  nodes: Array<{
    id: string;
    type: string;
    position: { x: number; y: number };
    data: LayerNodeData;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    sourceHandle?: string | null;
    targetHandle?: string | null;
  }>;
};

export const FASHION_MNIST = {
  name: "Fashion-MNIST",
  description:
    "28×28 grayscale clothing images (T-shirt, dress, sneaker, …) — 10 classes.",
  inputShape: [1, 28, 28] as TensorShape,
  numClasses: 10,
  classNames: [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
  ],
};

export const INPUT_PRESETS = {
  fashionMnist: { label: "Fashion-MNIST", shape: [1, 28, 28] as TensorShape },
  mnist: { label: "MNIST", shape: [1, 28, 28] as TensorShape },
  rgb224: { label: "RGB 224×224", shape: [3, 224, 224] as TensorShape },
  rgb32: { label: "CIFAR-like 32×32", shape: [3, 32, 32] as TensorShape },
  sequence: { label: "Sequence (tokens)", shape: [128] as TensorShape },
} as const;
