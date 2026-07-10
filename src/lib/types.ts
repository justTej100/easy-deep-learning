export type LayerType =
  | "Input"
  | "Conv2d"
  | "MaxPool2d"
  | "Flatten"
  | "Linear"
  | "ReLU"
  | "Dropout"
  | "Softmax"
  | "Output"
  | "LoopBlock";

export type TensorShape = number[]; // e.g. [1, 28, 28] or [10]

export type LayerParams = {
  // Conv2d
  outChannels?: number;
  kernelSize?: number;
  stride?: number;
  padding?: number;
  // MaxPool2d
  poolKernel?: number;
  poolStride?: number;
  // Linear
  outFeatures?: number;
  // Dropout
  p?: number;
  // Softmax
  dim?: number;
  // LoopBlock
  repeats?: number;
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
  /** For LoopBlock: ordered child node ids inside the loop */
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
