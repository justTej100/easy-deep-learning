import { formatShape, shapeProduct } from "./layers";
import { propagateShapes } from "./shapes";
import type { LayerNodeData, LayerType, TensorShape } from "./types";

export type SimStep = {
  id: string;
  label: string;
  layerType: LayerType;
  inputShape: TensorShape | null;
  outputShape: TensorShape | null;
  paramsHint: string;
  error: string | null;
  approxParams: number;
};

export type SimulationResult = {
  ok: boolean;
  steps: SimStep[];
  totalApproxParams: number;
  summary: string;
};

function approxLayerParams(data: LayerNodeData): number {
  const p = data.params;
  const inShape = data.inputShape;
  switch (data.layerType) {
    case "Conv2d": {
      const inC = inShape?.[0] ?? 1;
      const k = p.kernelSize ?? 3;
      const outC = p.outChannels ?? 32;
      return inC * outC * k * k + outC;
    }
    case "Conv1d": {
      const inC = inShape?.[0] ?? 1;
      const k = p.kernelSize ?? 3;
      const outC = p.outChannels ?? 64;
      return inC * outC * k + outC;
    }
    case "Linear":
    case "Output": {
      const inF = inShape?.[0] ?? 0;
      const outF =
        data.layerType === "Output" ? (p.numClasses ?? 10) : (p.outFeatures ?? 128);
      return inF * outF + outF;
    }
    case "BatchNorm2d":
      return (inShape?.[0] ?? 0) * 2;
    case "Embedding":
      return (p.numEmbeddings ?? 1000) * (p.embeddingDim ?? 64);
    case "LSTM": {
      const feat = inShape?.length === 2 ? inShape[1] : (inShape?.[0] ?? 64);
      const h = p.hiddenSize ?? 128;
      // 4 gates
      return 4 * ((feat + h) * h + h);
    }
    case "GRU": {
      const feat = inShape?.length === 2 ? inShape[1] : (inShape?.[0] ?? 64);
      const h = p.hiddenSize ?? 128;
      return 3 * ((feat + h) * h + h);
    }
    default:
      return 0;
  }
}

/** Simulate a forward pass by walking shapes — no weights, no training. */
export function simulateForwardPass(
  nodes: Array<{ id: string; data: LayerNodeData }>,
  edges: Array<{ source: string; target: string }>,
): SimulationResult {
  const shapes = propagateShapes(nodes, edges);
  const byId = new Map(nodes.map((n) => [n.id, n]));

  // topo from shapes order via Kahn again
  const incoming = new Map<string, number>();
  const outgoing = new Map<string, string[]>();
  for (const n of nodes) {
    incoming.set(n.id, 0);
    outgoing.set(n.id, []);
  }
  for (const e of edges) {
    if (!byId.has(e.source) || !byId.has(e.target)) continue;
    incoming.set(e.target, (incoming.get(e.target) ?? 0) + 1);
    outgoing.get(e.source)!.push(e.target);
  }
  const q = nodes.filter((n) => incoming.get(n.id) === 0).map((n) => n.id);
  const order: string[] = [];
  while (q.length) {
    const id = q.shift()!;
    order.push(id);
    for (const t of outgoing.get(id)!) {
      incoming.set(t, incoming.get(t)! - 1);
      if (incoming.get(t) === 0) q.push(t);
    }
  }

  const steps: SimStep[] = [];
  let total = 0;
  let ok = order.length === nodes.length;

  for (const id of order) {
    const node = byId.get(id)!;
    const s = shapes.get(id);
    const approx = approxLayerParams({
      ...node.data,
      inputShape: s?.inputShape,
      outputShape: s?.outputShape,
    });
    total += approx;
    if (s?.error) ok = false;
    steps.push({
      id,
      label: node.data.label,
      layerType: node.data.layerType,
      inputShape: s?.inputShape ?? null,
      outputShape: s?.outputShape ?? null,
      paramsHint: formatShape(s?.outputShape),
      error: s?.error ?? null,
      approxParams: approx,
    });
  }

  const last = steps[steps.length - 1];
  const summary = ok
    ? `Forward pass OK · final shape ${formatShape(last?.outputShape)} · ~${total.toLocaleString()} trainable params`
    : `Forward pass blocked — fix ${steps.filter((s) => s.error).length} shape issue(s).`;

  return { ok, steps, totalApproxParams: total, summary };
}

export function describeTensor(shape: TensorShape | null): string {
  if (!shape) return "unknown";
  return `${formatShape(shape)} (${shapeProduct(shape).toLocaleString()} values / sample)`;
}
