import type { LayerParams, LayerType, TensorShape } from "./types";
import { FASHION_MNIST } from "./types";

export type ParamField = {
  key: keyof LayerParams;
  label: string;
  type: "number";
  min?: number;
  max?: number;
  step?: number;
  defaultValue: number;
  help: string;
};

export type LayerCategory =
  | "io"
  | "conv"
  | "pool"
  | "norm"
  | "dense"
  | "sequence"
  | "activation"
  | "ops"
  | "research";

export type LayerDefinition = {
  type: LayerType;
  label: string;
  shortLabel: string;
  category: LayerCategory;
  color: string;
  accent: string;
  beginner: boolean;
  researchOnly?: boolean;
  allowsMultiInput?: boolean;
  description: string;
  why: string;
  analogy: string;
  params: ParamField[];
  defaultParams: LayerParams;
};

const num = (
  key: keyof LayerParams,
  label: string,
  defaultValue: number,
  help: string,
  opts: Partial<ParamField> = {},
): ParamField => ({
  key,
  label,
  type: "number",
  defaultValue,
  help,
  ...opts,
});

export const LAYER_DEFS: Record<LayerType, LayerDefinition> = {
  Input: {
    type: "Input",
    label: "Input",
    shortLabel: "Input",
    category: "io",
    color: "#0f766e",
    accent: "#ccfbf1",
    beginner: true,
    description:
      "Network entry point. Set channels × height × width (images) or a sequence length (tokens).",
    why: "Every model needs a fixed input shape so later layers know how many numbers to expect.",
    analogy: "Like the front door of a house — everything else is built around its size.",
    params: [
      num("inChannels", "Channels", 1, "1 = grayscale, 3 = RGB. Set 0 for 1D sequence input.", {
        min: 0,
        max: 16,
      }),
      num("height", "Height", 28, "Image height (ignored for sequences).", { min: 1, max: 512 }),
      num("width", "Width", 28, "Image width, or sequence length when channels = 0.", {
        min: 1,
        max: 2048,
      }),
    ],
    defaultParams: { inChannels: 1, height: 28, width: 28 },
  },
  Conv2d: {
    type: "Conv2d",
    label: "Conv2d",
    shortLabel: "Conv2d",
    category: "conv",
    color: "#1d4ed8",
    accent: "#dbeafe",
    beginner: true,
    description:
      "Slides filters over an image to detect local patterns like edges, textures, and shapes.",
    why: "Images have spatial structure — convolutions learn local patterns with far fewer parameters than a giant Linear layer.",
    analogy: "Scanning a photo with a small magnifying glass looking for the same pattern everywhere.",
    params: [
      num("outChannels", "Filters", 32, "How many feature maps to learn.", { min: 1, max: 512 }),
      num("kernelSize", "Kernel", 3, "Filter window size (usually 3).", {
        min: 1,
        max: 11,
        step: 2,
      }),
      num("stride", "Stride", 1, "Pixels the filter jumps each step.", { min: 1, max: 4 }),
      num("padding", "Padding", 1, "Zeros around the border.", { min: 0, max: 8 }),
    ],
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 },
  },
  Conv1d: {
    type: "Conv1d",
    label: "Conv1d",
    shortLabel: "Conv1d",
    category: "conv",
    color: "#2563eb",
    accent: "#dbeafe",
    beginner: false,
    description: "1D convolution over sequences (time series, audio frames, token channels).",
    why: "Same local-filter idea as Conv2d, but along one axis — great for sequential signals.",
    analogy: "A sliding window along a timeline.",
    params: [
      num("outChannels", "Filters", 64, "Output channels.", { min: 1, max: 512 }),
      num("kernelSize", "Kernel", 3, "Window length.", { min: 1, max: 31 }),
      num("stride", "Stride", 1, "Step size.", { min: 1, max: 8 }),
      num("padding", "Padding", 1, "Padding along the sequence.", { min: 0, max: 16 }),
    ],
    defaultParams: { outChannels: 64, kernelSize: 3, stride: 1, padding: 1 },
  },
  MaxPool2d: {
    type: "MaxPool2d",
    label: "MaxPool2d",
    shortLabel: "MaxPool",
    category: "pool",
    color: "#0369a1",
    accent: "#e0f2fe",
    beginner: true,
    description: "Keeps the strongest value in each window — shrinks height/width.",
    why: "Reduces compute and makes features more robust to tiny shifts.",
    analogy: "Summarizing a neighborhood by its tallest building.",
    params: [
      num("poolKernel", "Kernel", 2, "Window size.", { min: 2, max: 4 }),
      num("poolStride", "Stride", 2, "Usually matches kernel.", { min: 1, max: 4 }),
    ],
    defaultParams: { poolKernel: 2, poolStride: 2 },
  },
  AvgPool2d: {
    type: "AvgPool2d",
    label: "AvgPool2d",
    shortLabel: "AvgPool",
    category: "pool",
    color: "#0284c7",
    accent: "#e0f2fe",
    beginner: false,
    description: "Averages each window instead of taking the max — smoother downsampling.",
    why: "Often used before classification heads when you want a gentler spatial reduction.",
    analogy: "Averaging neighborhood heights instead of picking the tallest.",
    params: [
      num("poolKernel", "Kernel", 2, "Window size.", { min: 2, max: 4 }),
      num("poolStride", "Stride", 2, "Step size.", { min: 1, max: 4 }),
    ],
    defaultParams: { poolKernel: 2, poolStride: 2 },
  },
  AdaptiveAvgPool2d: {
    type: "AdaptiveAvgPool2d",
    label: "AdaptiveAvgPool2d",
    shortLabel: "AdaptPool",
    category: "pool",
    color: "#0e7490",
    accent: "#cffafe",
    beginner: false,
    description: "Pools any spatial size down to a fixed H×W (often 1×1 for a global vector).",
    why: "Lets you accept variable image sizes and still feed a fixed-size classifier.",
    analogy: "Always summarizing the whole map into the same number of bins.",
    params: [
      num("outputSize", "Output size", 1, "Target H and W (square).", { min: 1, max: 16 }),
    ],
    defaultParams: { outputSize: 1 },
  },
  BatchNorm2d: {
    type: "BatchNorm2d",
    label: "BatchNorm2d",
    shortLabel: "BatchNorm",
    category: "norm",
    color: "#7c3aed",
    accent: "#ede9fe",
    beginner: false,
    description: "Normalizes each channel across the batch — stabilizes and speeds up training.",
    why: "Keeps activations in a healthy range as weights change; common after Conv layers.",
    analogy: "Re-centering every feature map so none dominate the others.",
    params: [],
    defaultParams: {},
  },
  LayerNorm: {
    type: "LayerNorm",
    label: "LayerNorm",
    shortLabel: "LayerNorm",
    category: "norm",
    color: "#6d28d9",
    accent: "#ede9fe",
    beginner: false,
    description: "Normalizes across features for each sample — standard in transformers.",
    why: "Works well with variable sequence lengths and doesn’t depend on batch size.",
    analogy: "Rescaling each example’s feature vector independently.",
    params: [],
    defaultParams: {},
  },
  Flatten: {
    type: "Flatten",
    label: "Flatten",
    shortLabel: "Flatten",
    category: "dense",
    color: "#7c2d12",
    accent: "#ffedd5",
    beginner: true,
    description: "Turns a multi-dimensional tensor into one long vector.",
    why: "Linear layers expect a 1D list of features, not a grid.",
    analogy: "Unrolling a folded map into a straight line.",
    params: [],
    defaultParams: {},
  },
  Reshape: {
    type: "Reshape",
    label: "Reshape",
    shortLabel: "Reshape",
    category: "ops",
    color: "#9a3412",
    accent: "#ffedd5",
    beginner: false,
    description: "Reinterprets the same numbers into a new C×H×W layout (product must match).",
    why: "Useful when bridging sequence and image-like layouts.",
    analogy: "Repacking the same bricks into a different-shaped box.",
    params: [
      num("reshapeChannels", "Channels", 1, "New channel count.", { min: 1, max: 512 }),
      num("reshapeHeight", "Height", 28, "New height.", { min: 1, max: 512 }),
      num("reshapeWidth", "Width", 28, "New width.", { min: 1, max: 512 }),
    ],
    defaultParams: { reshapeChannels: 1, reshapeHeight: 28, reshapeWidth: 28 },
  },
  Linear: {
    type: "Linear",
    label: "Linear (Dense)",
    shortLabel: "Dense",
    category: "dense",
    color: "#b45309",
    accent: "#fef3c7",
    beginner: true,
    description: "Fully connected layer — every input connects to every output.",
    why: "Combines all features for higher-level decisions and class scores.",
    analogy: "A voting booth where every feature gets a weighted vote.",
    params: [
      num("outFeatures", "Units", 128, "How many numbers this layer outputs.", {
        min: 1,
        max: 4096,
      }),
    ],
    defaultParams: { outFeatures: 128 },
  },
  Embedding: {
    type: "Embedding",
    label: "Embedding",
    shortLabel: "Embed",
    category: "sequence",
    color: "#a16207",
    accent: "#fef9c3",
    beginner: false,
    description: "Maps token IDs to dense vectors — the usual first step for text models.",
    why: "Turns discrete symbols into continuous features the network can learn from.",
    analogy: "A dictionary that looks up a meaning-vector for each word ID.",
    params: [
      num("numEmbeddings", "Vocab size", 1000, "Number of distinct tokens.", {
        min: 2,
        max: 100000,
      }),
      num("embeddingDim", "Embed dim", 64, "Vector size per token.", { min: 1, max: 1024 }),
    ],
    defaultParams: { numEmbeddings: 1000, embeddingDim: 64 },
  },
  LSTM: {
    type: "LSTM",
    label: "LSTM",
    shortLabel: "LSTM",
    category: "sequence",
    color: "#c2410c",
    accent: "#ffedd5",
    beginner: false,
    description: "Recurrent layer with memory gates — processes sequences step by step.",
    why: "Captures long-range temporal dependencies better than a plain RNN.",
    analogy: "Reading a story while keeping notes about what still matters.",
    params: [
      num("hiddenSize", "Hidden size", 128, "Hidden state dimension.", { min: 1, max: 1024 }),
      num("numLayers", "Layers", 1, "Stacked LSTM depth.", { min: 1, max: 4 }),
      num("bidirectional", "Bidirectional", 0, "1 = forward+backward, 0 = forward only.", {
        min: 0,
        max: 1,
      }),
    ],
    defaultParams: { hiddenSize: 128, numLayers: 1, bidirectional: 0 },
  },
  GRU: {
    type: "GRU",
    label: "GRU",
    shortLabel: "GRU",
    category: "sequence",
    color: "#ea580c",
    accent: "#ffedd5",
    beginner: false,
    description: "A lighter recurrent alternative to LSTM with fewer gates.",
    why: "Often similar accuracy to LSTM with less compute.",
    analogy: "A simpler notepad for reading sequences.",
    params: [
      num("hiddenSize", "Hidden size", 128, "Hidden state dimension.", { min: 1, max: 1024 }),
      num("numLayers", "Layers", 1, "Stacked GRU depth.", { min: 1, max: 4 }),
      num("bidirectional", "Bidirectional", 0, "1 = bidirectional.", { min: 0, max: 1 }),
    ],
    defaultParams: { hiddenSize: 128, numLayers: 1, bidirectional: 0 },
  },
  MultiheadAttention: {
    type: "MultiheadAttention",
    label: "MultiheadAttention",
    shortLabel: "Attention",
    category: "sequence",
    color: "#db2777",
    accent: "#fce7f3",
    beginner: false,
    description:
      "Lets every position look at every other position — the core of transformers.",
    why: "Models long-range relationships without stepping through time one token at a time.",
    analogy: "A meeting where everyone can shout to everyone else at once.",
    params: [
      num("embedDim", "Embed dim", 64, "Must match feature size.", { min: 8, max: 1024 }),
      num("numHeads", "Heads", 4, "Parallel attention heads (must divide embed dim).", {
        min: 1,
        max: 16,
      }),
    ],
    defaultParams: { embedDim: 64, numHeads: 4 },
  },
  ReLU: {
    type: "ReLU",
    label: "ReLU",
    shortLabel: "ReLU",
    category: "activation",
    color: "#15803d",
    accent: "#dcfce7",
    beginner: true,
    description: "Zeros negatives, keeps positives — the default non-linearity.",
    why: "Without activations, stacked layers collapse into one linear map.",
    analogy: "A one-way valve for signals.",
    params: [],
    defaultParams: {},
  },
  LeakyReLU: {
    type: "LeakyReLU",
    label: "LeakyReLU",
    shortLabel: "LeakyReLU",
    category: "activation",
    color: "#16a34a",
    accent: "#dcfce7",
    beginner: false,
    description: "Like ReLU but lets a small negative slope through.",
    why: "Avoids completely “dead” neurons that never fire again.",
    analogy: "A mostly one-way valve with a tiny leak.",
    params: [
      num("negativeSlope", "Negative slope", 0.01, "Slope for negative values.", {
        min: 0,
        max: 0.5,
        step: 0.01,
      }),
    ],
    defaultParams: { negativeSlope: 0.01 },
  },
  GELU: {
    type: "GELU",
    label: "GELU",
    shortLabel: "GELU",
    category: "activation",
    color: "#059669",
    accent: "#d1fae5",
    beginner: false,
    description: "Smooth activation used in modern transformers (BERT, GPT-style).",
    why: "Often trains more stably than hard ReLU on transformer blocks.",
    analogy: "A soft gate instead of an on/off switch.",
    params: [],
    defaultParams: {},
  },
  Sigmoid: {
    type: "Sigmoid",
    label: "Sigmoid",
    shortLabel: "Sigmoid",
    category: "activation",
    color: "#0d9488",
    accent: "#ccfbf1",
    beginner: false,
    description: "Squashes each value into (0, 1).",
    why: "Classic for independent probabilities / gates (not usually the final multi-class head).",
    analogy: "Turning raw scores into “percent yes.”",
    params: [],
    defaultParams: {},
  },
  Tanh: {
    type: "Tanh",
    label: "Tanh",
    shortLabel: "Tanh",
    category: "activation",
    color: "#0f766e",
    accent: "#ccfbf1",
    beginner: false,
    description: "Squashes values into (−1, 1), zero-centered.",
    why: "Common in older RNNs; sometimes nicer than sigmoid for hidden states.",
    analogy: "A balanced soft clamp around zero.",
    params: [],
    defaultParams: {},
  },
  Dropout: {
    type: "Dropout",
    label: "Dropout",
    shortLabel: "Dropout",
    category: "activation",
    color: "#a16207",
    accent: "#fef9c3",
    beginner: true,
    description: "Randomly zeros activations during training to reduce overfitting.",
    why: "Forces the network not to rely on any single neuron.",
    analogy: "Practicing with random teammates benched.",
    params: [
      num("p", "Drop probability", 0.25, "Fraction dropped during training.", {
        min: 0,
        max: 0.9,
        step: 0.05,
      }),
    ],
    defaultParams: { p: 0.25 },
  },
  Softmax: {
    type: "Softmax",
    label: "Softmax",
    shortLabel: "Softmax",
    category: "activation",
    color: "#6d28d9",
    accent: "#ede9fe",
    beginner: true,
    description: "Turns logits into probabilities that sum to 1.",
    why: "Useful for interpreting class scores. CrossEntropyLoss already includes log-softmax.",
    analogy: "Turning contest scores into win probabilities.",
    params: [
      num("dim", "Dimension", -1, "Axis to normalize over (usually -1).", { min: -1, max: 3 }),
    ],
    defaultParams: { dim: -1 },
  },
  Add: {
    type: "Add",
    label: "Add (Residual)",
    shortLabel: "Add",
    category: "ops",
    color: "#be185d",
    accent: "#fce7f3",
    beginner: false,
    allowsMultiInput: true,
    description: "Element-wise sum of two tensors with the same shape — residual skip connections.",
    why: "Lets gradients flow through deep nets (ResNet-style).",
    analogy: "Merging two identical roads back into one.",
    params: [],
    defaultParams: {},
  },
  Concat: {
    type: "Concat",
    label: "Concatenate",
    shortLabel: "Concat",
    category: "ops",
    color: "#9d174d",
    accent: "#fce7f3",
    beginner: false,
    allowsMultiInput: true,
    description: "Stacks two tensors along a chosen dimension (channels by default).",
    why: "Combines feature maps from parallel branches (U-Net / DenseNet style).",
    analogy: "Zippering two stacks of papers into one thicker stack.",
    params: [
      num("concatDim", "Dimension", 0, "0 = channels for C×H×W feature maps.", {
        min: 0,
        max: 2,
      }),
    ],
    defaultParams: { concatDim: 0 },
  },
  Output: {
    type: "Output",
    label: "Output",
    shortLabel: "Output",
    category: "io",
    color: "#be123c",
    accent: "#ffe4e6",
    beginner: true,
    description: `Final classification head — Linear to num_classes (default ${FASHION_MNIST.numClasses}).`,
    why: "Classification needs one score per class.",
    analogy: "Scoreboards — one per label.",
    params: [
      num("numClasses", "Classes", FASHION_MNIST.numClasses, "Number of output classes.", {
        min: 2,
        max: 1000,
      }),
    ],
    defaultParams: { numClasses: FASHION_MNIST.numClasses },
  },
  LoopBlock: {
    type: "LoopBlock",
    label: "Loop Block",
    shortLabel: "Loop",
    category: "research",
    color: "#0e7490",
    accent: "#cffafe",
    beginner: false,
    researchOnly: true,
    description:
      "Runs a shared-weight block N times — looped / recurrent-depth transformers.",
    why: "Trade inference compute for parameter count.",
    analogy: "Re-reading the same paragraph N times with the same brain.",
    params: [
      num("repeats", "Repeat count (N)", 2, "Shared forward passes.", { min: 1, max: 16 }),
    ],
    defaultParams: { repeats: 2 },
  },
};

export const CATEGORY_LABELS: Record<LayerCategory, string> = {
  io: "I/O",
  conv: "Convolution",
  pool: "Pooling",
  norm: "Normalization",
  dense: "Dense",
  sequence: "Sequence",
  activation: "Activation",
  ops: "Operations",
  research: "Research",
};

export const BEGINNER_PALETTE: LayerType[] = [
  "Conv2d",
  "MaxPool2d",
  "Flatten",
  "Linear",
  "ReLU",
  "Dropout",
  "Softmax",
  "BatchNorm2d",
];

export const ADVANCED_PALETTE: LayerType[] = [
  "Conv1d",
  "AvgPool2d",
  "AdaptiveAvgPool2d",
  "LayerNorm",
  "Reshape",
  "Embedding",
  "LSTM",
  "GRU",
  "MultiheadAttention",
  "LeakyReLU",
  "GELU",
  "Sigmoid",
  "Tanh",
  "Add",
  "Concat",
];

export const RESEARCH_PALETTE: LayerType[] = ["LoopBlock"];

export function formatShape(shape: TensorShape | null | undefined): string {
  if (!shape || shape.length === 0) return "—";
  return shape.join(" × ");
}

export function shapeProduct(shape: TensorShape): number {
  return shape.reduce((a, b) => a * b, 1);
}

export function inputShapeFromParams(params: LayerParams): TensorShape {
  const c = params.inChannels ?? 1;
  const h = params.height ?? 28;
  const w = params.width ?? 28;
  if (c <= 0) return [w]; // treat width as sequence length
  return [c, h, w];
}
