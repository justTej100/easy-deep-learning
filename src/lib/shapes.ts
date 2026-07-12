import { LAYER_DEFS, inputShapeFromParams, shapeProduct } from "./layers";
import type { LayerNodeData, LayerParams, TensorShape } from "./types";
import { FASHION_MNIST } from "./types";

function convOutDim(size: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1;
}

function poolOutDim(size: number, kernel: number, stride: number): number {
  return Math.floor((size - kernel) / stride) + 1;
}

export function inferOutputShape(
  layerType: LayerNodeData["layerType"],
  params: LayerParams,
  inputShape: TensorShape | null,
  extraInputs: TensorShape[] = [],
): { shape: TensorShape | null; error: string | null } {
  if (layerType === "Input") {
    return { shape: inputShapeFromParams(params), error: null };
  }

  if (layerType === "Add" || layerType === "Concat") {
    const shapes = [inputShape, ...extraInputs].filter(Boolean) as TensorShape[];
    if (shapes.length < 2) {
      return {
        shape: null,
        error: `${LAYER_DEFS[layerType].label} needs two incoming connections (e.g. a residual skip).`,
      };
    }
    if (layerType === "Add") {
      const [a, b] = shapes;
      if (a.length !== b.length || a.some((v, i) => v !== b[i])) {
        return {
          shape: null,
          error: `Add needs identical shapes. Got ${a.join("×")} and ${b.join("×")}.`,
        };
      }
      return { shape: [...a], error: null };
    }
    // Concat
    const dim = params.concatDim ?? 0;
    const [a, b] = shapes;
    if (a.length !== b.length) {
      return {
        shape: null,
        error: `Concat tensors must have the same rank. Got ${a.length}D and ${b.length}D.`,
      };
    }
    if (dim < 0 || dim >= a.length) {
      return { shape: null, error: `concatDim ${dim} is out of range for shape ${a.join("×")}.` };
    }
    for (let i = 0; i < a.length; i++) {
      if (i !== dim && a[i] !== b[i]) {
        return {
          shape: null,
          error: `Concat requires matching sizes on non-concat dims. Dim ${i}: ${a[i]} vs ${b[i]}.`,
        };
      }
    }
    const out = [...a];
    out[dim] = a[dim] + b[dim];
    return { shape: out, error: null };
  }

  if (!inputShape) {
    return {
      shape: null,
      error: "Connect an upstream layer first so we know the input size.",
    };
  }

  switch (layerType) {
    case "Conv2d": {
      if (inputShape.length !== 3) {
        return {
          shape: null,
          error: `Conv2d expects C×H×W, got ${inputShape.join("×")}.`,
        };
      }
      const [, h, w] = inputShape;
      const k = params.kernelSize ?? 3;
      const s = params.stride ?? 1;
      const p = params.padding ?? 0;
      const outC = params.outChannels ?? 32;
      const outH = convOutDim(h, k, s, p);
      const outW = convOutDim(w, k, s, p);
      if (outH <= 0 || outW <= 0) {
        return {
          shape: null,
          error: `Conv2d would produce invalid spatial size ${outH}×${outW}.`,
        };
      }
      return { shape: [outC, outH, outW], error: null };
    }
    case "Conv1d": {
      if (inputShape.length !== 2) {
        return {
          shape: null,
          error: `Conv1d expects C×L, got ${inputShape.join("×")}. Use Embedding/reshape first if needed.`,
        };
      }
      const [, L] = inputShape;
      const k = params.kernelSize ?? 3;
      const s = params.stride ?? 1;
      const p = params.padding ?? 0;
      const outC = params.outChannels ?? 64;
      const outL = convOutDim(L, k, s, p);
      if (outL <= 0) {
        return { shape: null, error: `Conv1d would produce invalid length ${outL}.` };
      }
      return { shape: [outC, outL], error: null };
    }
    case "MaxPool2d":
    case "AvgPool2d": {
      if (inputShape.length !== 3) {
        return {
          shape: null,
          error: `${layerType} needs C×H×W, got ${inputShape.join("×")}.`,
        };
      }
      const [c, h, w] = inputShape;
      const k = params.poolKernel ?? 2;
      const s = params.poolStride ?? 2;
      const outH = poolOutDim(h, k, s);
      const outW = poolOutDim(w, k, s);
      if (outH <= 0 || outW <= 0) {
        return {
          shape: null,
          error: `Pooling would produce ${outH}×${outW} — feature map too small.`,
        };
      }
      return { shape: [c, outH, outW], error: null };
    }
    case "AdaptiveAvgPool2d": {
      if (inputShape.length !== 3) {
        return {
          shape: null,
          error: `AdaptiveAvgPool2d needs C×H×W, got ${inputShape.join("×")}.`,
        };
      }
      const size = params.outputSize ?? 1;
      return { shape: [inputShape[0], size, size], error: null };
    }
    case "BatchNorm2d": {
      if (inputShape.length !== 3) {
        return {
          shape: null,
          error: `BatchNorm2d expects C×H×W feature maps, got ${inputShape.join("×")}.`,
        };
      }
      return { shape: [...inputShape], error: null };
    }
    case "LayerNorm":
    case "ReLU":
    case "LeakyReLU":
    case "GELU":
    case "Sigmoid":
    case "Tanh":
    case "Dropout":
    case "LoopBlock":
      return { shape: [...inputShape], error: null };
    case "Flatten":
      if (inputShape.length === 1) return { shape: inputShape, error: null };
      return { shape: [shapeProduct(inputShape)], error: null };
    case "Reshape": {
      const c = params.reshapeChannels ?? 1;
      const h = params.reshapeHeight ?? 28;
      const w = params.reshapeWidth ?? 28;
      const target = c * h * w;
      const src = shapeProduct(inputShape);
      if (target !== src) {
        return {
          shape: null,
          error: `Reshape to ${c}×${h}×${w} needs ${target} values, but input has ${src}.`,
        };
      }
      return { shape: [c, h, w], error: null };
    }
    case "Linear": {
      if (inputShape.length !== 1) {
        return {
          shape: null,
          error: `Linear expects a flat vector, got ${inputShape.join("×")}. Insert Flatten first.`,
        };
      }
      return { shape: [params.outFeatures ?? 128], error: null };
    }
    case "Embedding": {
      // Input is token ids: [seq] → [seq, embed]
      if (inputShape.length !== 1) {
        return {
          shape: null,
          error: `Embedding expects a 1D sequence of token ids, got ${inputShape.join("×")}.`,
        };
      }
      return { shape: [inputShape[0], params.embeddingDim ?? 64], error: null };
    }
    case "LSTM":
    case "GRU": {
      // Accept [seq, feat] or [feat] treated as seq=1
      let seq: number;
      let feat: number;
      if (inputShape.length === 2) {
        [seq, feat] = inputShape;
      } else if (inputShape.length === 1) {
        seq = 1;
        feat = inputShape[0];
      } else {
        return {
          shape: null,
          error: `${layerType} expects [seq, features], got ${inputShape.join("×")}.`,
        };
      }
      void feat;
      const hidden = params.hiddenSize ?? 128;
      const dirs = (params.bidirectional ?? 0) === 1 ? 2 : 1;
      return { shape: [seq, hidden * dirs], error: null };
    }
    case "MultiheadAttention": {
      if (inputShape.length !== 2) {
        return {
          shape: null,
          error: `Attention expects [seq, embed], got ${inputShape.join("×")}.`,
        };
      }
      const embed = params.embedDim ?? inputShape[1];
      const heads = params.numHeads ?? 4;
      if (embed !== inputShape[1]) {
        return {
          shape: null,
          error: `embedDim (${embed}) must match input features (${inputShape[1]}).`,
        };
      }
      if (embed % heads !== 0) {
        return {
          shape: null,
          error: `embedDim (${embed}) must be divisible by numHeads (${heads}).`,
        };
      }
      return { shape: [...inputShape], error: null };
    }
    case "Softmax": {
      if (inputShape.length !== 1) {
        return {
          shape: null,
          error: `Softmax for classification usually sits on a class-score vector. Got ${inputShape.join("×")}.`,
        };
      }
      return { shape: [...inputShape], error: null };
    }
    case "Output": {
      const n = params.numClasses ?? FASHION_MNIST.numClasses;
      if (inputShape.length !== 1) {
        return {
          shape: null,
          error: `Output needs a flat vector. Got ${inputShape.join("×")}. Add Flatten first.`,
        };
      }
      return { shape: [n], error: null };
    }
    default:
      return { shape: null, error: "Unknown layer type." };
  }
}

export function mismatchMessage(
  sourceLabel: string,
  sourceShape: TensorShape,
  targetType: LayerNodeData["layerType"],
  targetLabel: string,
): string | null {
  const def = LAYER_DEFS[targetType];
  if (targetType === "Linear" && sourceShape.length !== 1) {
    return `${sourceLabel} outputs ${sourceShape.join("×")} (${shapeProduct(sourceShape)} numbers). ${targetLabel} (${def.label}) expects a flat list — add Flatten.`;
  }
  if (targetType === "Conv2d" && sourceShape.length !== 3) {
    return `${sourceLabel} outputs ${sourceShape.join("×")}, but Conv2d needs C×H×W.`;
  }
  if (
    (targetType === "MaxPool2d" || targetType === "AvgPool2d" || targetType === "BatchNorm2d") &&
    sourceShape.length !== 3
  ) {
    return `${sourceLabel} outputs ${sourceShape.join("×")}, but ${def.label} needs a 2D feature map.`;
  }
  if (targetType === "Output" && sourceShape.length !== 1) {
    return `${sourceLabel} still has spatial dims (${sourceShape.join("×")}). Flatten before Output.`;
  }
  return null;
}

export function propagateShapes(
  nodes: Array<{ id: string; data: LayerNodeData }>,
  edges: Array<{ source: string; target: string }>,
): Map<
  string,
  { inputShape: TensorShape | null; outputShape: TensorShape | null; error: string | null }
> {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const n of nodes) {
    incoming.set(n.id, []);
    outgoing.set(n.id, []);
  }
  for (const e of edges) {
    if (!byId.has(e.source) || !byId.has(e.target)) continue;
    incoming.get(e.target)!.push(e.source);
    outgoing.get(e.source)!.push(e.target);
  }

  const result = new Map<
    string,
    { inputShape: TensorShape | null; outputShape: TensorShape | null; error: string | null }
  >();

  const indeg = new Map<string, number>();
  for (const n of nodes) indeg.set(n.id, incoming.get(n.id)!.length);
  const queue = nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
  const order: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const t of outgoing.get(id)!) {
      indeg.set(t, indeg.get(t)! - 1);
      if (indeg.get(t) === 0) queue.push(t);
    }
  }

  const hasCycle = order.length !== nodes.length;
  const computedOut = new Map<string, TensorShape | null>();

  for (const id of order) {
    const node = byId.get(id)!;
    const preds = incoming.get(id)!;
    const multiOk = LAYER_DEFS[node.data.layerType].allowsMultiInput;

    let inputShape: TensorShape | null = null;
    let extraInputs: TensorShape[] = [];
    let error: string | null = null;

    if (node.data.layerType === "Input") {
      inputShape = null;
    } else if (preds.length === 0) {
      error = "This layer isn’t connected to anything upstream.";
    } else if (preds.length > 1 && !multiOk) {
      error =
        "Multiple inputs aren’t supported on this layer — use Add/Concat for merges, or keep a single path.";
    } else if (preds.length > 2 && multiOk) {
      error = `${LAYER_DEFS[node.data.layerType].label} currently supports exactly two inputs.`;
    } else {
      const shapes = preds.map((srcId) => computedOut.get(srcId) ?? null);
      if (shapes.some((s) => !s)) {
        error = "Fix the upstream layer first.";
      } else {
        inputShape = shapes[0];
        extraInputs = shapes.slice(1) as TensorShape[];
        if (shapes[0] && !multiOk) {
          const mm = mismatchMessage(
            byId.get(preds[0])!.data.label,
            shapes[0],
            node.data.layerType,
            node.data.label,
          );
          if (mm) error = mm;
        }
      }
    }

    if (hasCycle && !error && node.data.layerType !== "LoopBlock") {
      // still compute reachable nodes; cycle leftovers handled below
    }

    const inferred = inferOutputShape(
      node.data.layerType,
      node.data.params,
      inputShape,
      extraInputs,
    );
    if (!error && inferred.error) error = inferred.error;

    computedOut.set(id, inferred.shape);
    result.set(id, {
      inputShape,
      outputShape: inferred.shape,
      error,
    });
  }

  for (const n of nodes) {
    if (!result.has(n.id)) {
      result.set(n.id, {
        inputShape: null,
        outputShape: null,
        error:
          "The graph has a cycle. Use a Loop Block for intentional repetition, or remove the cycle.",
      });
    }
  }

  return result;
}
