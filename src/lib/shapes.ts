import { LAYER_DEFS, shapeProduct } from "./layers";
import type { LayerNodeData, LayerParams, TensorShape } from "./types";
import { FASHION_MNIST } from "./types";

function convOutDim(
  size: number,
  kernel: number,
  stride: number,
  padding: number,
): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1;
}

function poolOutDim(size: number, kernel: number, stride: number): number {
  return Math.floor((size - kernel) / stride) + 1;
}

export function inferOutputShape(
  layerType: LayerNodeData["layerType"],
  params: LayerParams,
  inputShape: TensorShape | null,
): { shape: TensorShape | null; error: string | null } {
  if (layerType === "Input") {
    return { shape: [...FASHION_MNIST.inputShape], error: null };
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
          error: `Conv2d expects an image-like tensor (channels × height × width), but got shape ${inputShape.join("×")}. Add Flatten/Linear only after you're done with convolutions — or connect Conv2d right after Input / another Conv / Pool.`,
        };
      }
      const [c, h, w] = inputShape;
      const k = params.kernelSize ?? 3;
      const s = params.stride ?? 1;
      const p = params.padding ?? 0;
      const outC = params.outChannels ?? 32;
      const outH = convOutDim(h, k, s, p);
      const outW = convOutDim(w, k, s, p);
      if (outH <= 0 || outW <= 0) {
        return {
          shape: null,
          error: `This Conv2d shrinks the image to ${outH}×${outW}, which is invalid. Try smaller kernel, larger padding, or pool less aggressively earlier.`,
        };
      }
      void c;
      return { shape: [outC, outH, outW], error: null };
    }
    case "MaxPool2d": {
      if (inputShape.length !== 3) {
        return {
          shape: null,
          error: `MaxPool2d only works on feature maps (C×H×W). Got ${inputShape.join("×")}.`,
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
          error: `Pooling would produce ${outH}×${outW}. The feature map is already too small — remove a pool or reduce kernel/stride.`,
        };
      }
      return { shape: [c, outH, outW], error: null };
    }
    case "Flatten": {
      if (inputShape.length === 1) {
        return {
          shape: inputShape,
          error: null,
        };
      }
      return { shape: [shapeProduct(inputShape)], error: null };
    }
    case "Linear": {
      if (inputShape.length !== 1) {
        return {
          shape: null,
          error: `Linear expects a flat vector, but got shape ${inputShape.join("×")} (${shapeProduct(inputShape)} numbers total). Insert a Flatten layer first.`,
        };
      }
      const out = params.outFeatures ?? 128;
      return { shape: [out], error: null };
    }
    case "ReLU":
    case "Dropout":
      return { shape: [...inputShape], error: null };
    case "Softmax": {
      if (inputShape.length !== 1) {
        return {
          shape: null,
          error: `Softmax for classification usually sits on a class-score vector. Got ${inputShape.join("×")}. Flatten + Linear to ${FASHION_MNIST.numClasses} first.`,
        };
      }
      return { shape: [...inputShape], error: null };
    }
    case "Output": {
      const n = params.numClasses ?? FASHION_MNIST.numClasses;
      if (inputShape.length !== 1) {
        return {
          shape: null,
          error: `The Output head needs a flat vector. Got ${inputShape.join("×")}. Add Flatten (and usually Linear layers) before Output.`,
        };
      }
      // Output is implemented as Linear(in, numClasses)
      return { shape: [n], error: null };
    }
    case "LoopBlock": {
      // Loop preserves shape: shared block must map shape → same shape.
      // We treat LoopBlock as identity for outer shape; inner validation is separate.
      return { shape: [...inputShape], error: null };
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
    return `${sourceLabel} outputs a ${sourceShape.join("×")} feature map (${shapeProduct(sourceShape)} numbers). ${targetLabel} (${def.label}) expects a flat list — add Flatten in between.`;
  }
  if (targetType === "Conv2d" && sourceShape.length !== 3) {
    return `${sourceLabel} outputs shape ${sourceShape.join("×")}, but Conv2d needs channels×height×width. You may have already flattened.`;
  }
  if (targetType === "MaxPool2d" && sourceShape.length !== 3) {
    return `${sourceLabel} outputs shape ${sourceShape.join("×")}, but MaxPool2d needs a 2D feature map.`;
  }
  if (targetType === "Output" && sourceShape.length !== 1) {
    return `${sourceLabel} still has spatial dimensions (${sourceShape.join("×")}). Flatten before the Output head.`;
  }
  return null;
}

/** Propagate shapes through a DAG given adjacency. */
export function propagateShapes(
  nodes: Array<{ id: string; data: LayerNodeData }>,
  edges: Array<{ source: string; target: string }>,
): Map<string, { inputShape: TensorShape | null; outputShape: TensorShape | null; error: string | null }> {
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

  // Kahn topological order
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
  if (hasCycle) {
    for (const n of nodes) {
      result.set(n.id, {
        inputShape: null,
        outputShape: null,
        error:
          n.data.layerType === "LoopBlock"
            ? null
            : "The graph has a cycle. In beginner mode the network must be a straight feed-forward path (use a Loop Block in Research mode for intentional repetition).",
      });
    }
    // Still try to compute what we can for nodes in order
  }

  const computedOut = new Map<string, TensorShape | null>();

  for (const id of order) {
    const node = byId.get(id)!;
    const preds = incoming.get(id)!;

    let inputShape: TensorShape | null = null;
    let error: string | null = null;

    if (node.data.layerType === "Input") {
      inputShape = null;
    } else if (preds.length === 0) {
      error = "This layer isn’t connected to anything upstream.";
    } else if (preds.length > 1) {
      error =
        "Multiple inputs aren’t supported yet — connect only one incoming edge (V1 is a single path / DAG without merges).";
    } else {
      const srcId = preds[0];
      const srcOut = computedOut.get(srcId) ?? null;
      const srcNode = byId.get(srcId)!;
      inputShape = srcOut;
      if (srcOut) {
        const mm = mismatchMessage(
          srcNode.data.label,
          srcOut,
          node.data.layerType,
          node.data.label,
        );
        if (mm) error = mm;
      } else if (!result.get(srcId)?.error) {
        error = "Upstream shape couldn’t be determined.";
      } else {
        error = "Fix the upstream layer first.";
      }
    }

    const inferred = inferOutputShape(node.data.layerType, node.data.params, inputShape);
    if (!error && inferred.error) error = inferred.error;

    // Soft warning: Output should be 10 classes for Fashion-MNIST task
    if (
      !error &&
      node.data.layerType === "Output" &&
      inferred.shape &&
      inferred.shape[0] !== FASHION_MNIST.numClasses
    ) {
      error = `Fashion-MNIST has ${FASHION_MNIST.numClasses} classes, but Output is set to ${inferred.shape[0]}.`;
    }

    computedOut.set(id, inferred.shape);
    result.set(id, {
      inputShape,
      outputShape: inferred.shape,
      error,
    });
  }

  // Nodes not in order (cycle leftovers)
  for (const n of nodes) {
    if (!result.has(n.id)) {
      result.set(n.id, {
        inputShape: null,
        outputShape: null,
        error: "Unreachable due to a cycle in the graph.",
      });
    }
  }

  return result;
}
