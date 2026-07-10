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

export type LayerDefinition = {
  type: LayerType;
  label: string;
  shortLabel: string;
  category: "io" | "conv" | "dense" | "activation" | "research";
  color: string;
  accent: string;
  beginner: boolean;
  researchOnly?: boolean;
  description: string;
  why: string;
  analogy: string;
  params: ParamField[];
  defaultParams: LayerParams;
};

export const LAYER_DEFS: Record<LayerType, LayerDefinition> = {
  Input: {
    type: "Input",
    label: "Input (Fashion-MNIST)",
    shortLabel: "Input",
    category: "io",
    color: "#0f766e",
    accent: "#ccfbf1",
    beginner: true,
    description:
      "The starting point of your network. Fashion-MNIST images are 28×28 pixels with 1 grayscale channel.",
    why: "Every model needs a fixed input shape so later layers know how many numbers to expect.",
    analogy: "Like the front door of a house — everything else is built around its size.",
    params: [],
    defaultParams: {},
  },
  Conv2d: {
    type: "Conv2d",
    label: "Conv2d",
    shortLabel: "Conv",
    category: "conv",
    color: "#1d4ed8",
    accent: "#dbeafe",
    beginner: true,
    description:
      "A convolutional layer slides small filters over the image to detect local patterns like edges, textures, and shapes.",
    why: "Images have spatial structure — nearby pixels matter together. Convolutions learn those local patterns with far fewer parameters than a giant Linear layer.",
    analogy: "Like scanning a photo with a small magnifying glass, looking for the same pattern everywhere.",
    params: [
      {
        key: "outChannels",
        label: "Output channels",
        type: "number",
        min: 1,
        max: 256,
        defaultValue: 32,
        help: "How many different filters (feature maps) to learn.",
      },
      {
        key: "kernelSize",
        label: "Kernel size",
        type: "number",
        min: 1,
        max: 11,
        step: 2,
        defaultValue: 3,
        help: "Width/height of each filter window (usually 3).",
      },
      {
        key: "stride",
        label: "Stride",
        type: "number",
        min: 1,
        max: 4,
        defaultValue: 1,
        help: "How many pixels the filter jumps each step.",
      },
      {
        key: "padding",
        label: "Padding",
        type: "number",
        min: 0,
        max: 5,
        defaultValue: 1,
        help: "Zeros added around the border so the spatial size can stay the same.",
      },
    ],
    defaultParams: { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 },
  },
  MaxPool2d: {
    type: "MaxPool2d",
    label: "MaxPool2d",
    shortLabel: "Pool",
    category: "conv",
    color: "#0369a1",
    accent: "#e0f2fe",
    beginner: true,
    description:
      "Downsamples the feature map by keeping only the strongest value in each small window. Shrinks height/width and makes features more robust to tiny shifts.",
    why: "Reduces computation and helps the network focus on “is this pattern present?” rather than exact pixel location.",
    analogy: "Like summarizing a neighborhood by its tallest building.",
    params: [
      {
        key: "poolKernel",
        label: "Kernel size",
        type: "number",
        min: 2,
        max: 4,
        defaultValue: 2,
        help: "Window size for taking the max.",
      },
      {
        key: "poolStride",
        label: "Stride",
        type: "number",
        min: 1,
        max: 4,
        defaultValue: 2,
        help: "Usually matches kernel size for non-overlapping pools.",
      },
    ],
    defaultParams: { poolKernel: 2, poolStride: 2 },
  },
  Flatten: {
    type: "Flatten",
    label: "Flatten",
    shortLabel: "Flat",
    category: "dense",
    color: "#7c2d12",
    accent: "#ffedd5",
    beginner: true,
    description:
      "Turns a multi-dimensional feature map (channels × height × width) into one long vector of numbers.",
    why: "Linear (fully connected) layers expect a 1D list of features, not a 2D/3D grid.",
    analogy: "Unrolling a folded map into a single straight line.",
    params: [],
    defaultParams: {},
  },
  Linear: {
    type: "Linear",
    label: "Linear (Dense)",
    shortLabel: "Linear",
    category: "dense",
    color: "#b45309",
    accent: "#fef3c7",
    beginner: true,
    description:
      "A fully connected layer: every input number connects to every output number via learned weights.",
    why: "After Flatten, Linear layers combine all features to make higher-level decisions (and eventually class scores).",
    analogy: "A voting booth where every feature gets a weighted vote on each output.",
    params: [
      {
        key: "outFeatures",
        label: "Output features",
        type: "number",
        min: 1,
        max: 2048,
        defaultValue: 128,
        help: "How many numbers this layer outputs.",
      },
    ],
    defaultParams: { outFeatures: 128 },
  },
  ReLU: {
    type: "ReLU",
    label: "ReLU",
    shortLabel: "ReLU",
    category: "activation",
    color: "#15803d",
    accent: "#dcfce7",
    beginner: true,
    description:
      "An activation that sets negative values to zero and keeps positives. Adds non-linearity so the network can learn curved decision boundaries.",
    why: "Without activations, stacking Linear/Conv layers is just one big linear transform — not powerful enough for images.",
    analogy: "A one-way valve: only positive signals pass through.",
    params: [],
    defaultParams: {},
  },
  Dropout: {
    type: "Dropout",
    label: "Dropout",
    shortLabel: "Drop",
    category: "activation",
    color: "#a16207",
    accent: "#fef9c3",
    beginner: true,
    description:
      "During training, randomly zeros a fraction of activations so the network can’t rely on any single neuron.",
    why: "A simple regularizer that reduces overfitting — especially useful on small datasets like Fashion-MNIST demos.",
    analogy: "Practicing a team sport with random players benched so everyone learns to contribute.",
    params: [
      {
        key: "p",
        label: "Drop probability",
        type: "number",
        min: 0,
        max: 0.9,
        step: 0.05,
        defaultValue: 0.25,
        help: "Fraction of units dropped during training (0 = off).",
      },
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
    description:
      "Converts a vector of raw scores (logits) into probabilities that sum to 1.",
    why: "Useful for interpreting class scores. Note: PyTorch’s CrossEntropyLoss already includes log-softmax, so many training scripts skip Softmax in the model.",
    analogy: "Turning contest scores into “percent chance of winning.”",
    params: [
      {
        key: "dim",
        label: "Dimension",
        type: "number",
        min: -1,
        max: 3,
        defaultValue: -1,
        help: "Which axis to normalize over (usually -1 = last).",
      },
    ],
    defaultParams: { dim: -1 },
  },
  Output: {
    type: "Output",
    label: "Output (10 classes)",
    shortLabel: "Output",
    category: "io",
    color: "#be123c",
    accent: "#ffe4e6",
    beginner: true,
    description: `The final prediction head for Fashion-MNIST’s ${FASHION_MNIST.numClasses} clothing classes. Typically a Linear layer with ${FASHION_MNIST.numClasses} outputs.`,
    why: "Classification needs one score per class so you can pick the most likely label.",
    analogy: "Ten scoreboards — one for each clothing type.",
    params: [
      {
        key: "numClasses",
        label: "Classes",
        type: "number",
        min: 2,
        max: 1000,
        defaultValue: FASHION_MNIST.numClasses,
        help: "Fashion-MNIST has 10 classes; leave this at 10 for the starter task.",
      },
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
      "Runs a shared-weight sub-block N times. Same parameters, more compute — the idea behind looped / recurrent-depth transformers.",
    why: "You can deepen reasoning at inference time without adding new parameters. Trade compute for capacity.",
    analogy: "Re-reading the same paragraph N times with the same brain, refining the answer each pass.",
    params: [
      {
        key: "repeats",
        label: "Repeat count (N)",
        type: "number",
        min: 1,
        max: 16,
        defaultValue: 2,
        help: "How many times to apply the shared block in the forward pass.",
      },
    ],
    defaultParams: { repeats: 2 },
  },
};

export const BEGINNER_PALETTE: LayerType[] = [
  "Conv2d",
  "MaxPool2d",
  "Flatten",
  "Linear",
  "ReLU",
  "Dropout",
  "Softmax",
];

export const RESEARCH_PALETTE: LayerType[] = ["LoopBlock"];

export function formatShape(shape: TensorShape | null | undefined): string {
  if (!shape || shape.length === 0) return "—";
  return shape.join(" × ");
}

export function shapeProduct(shape: TensorShape): number {
  return shape.reduce((a, b) => a * b, 1);
}
