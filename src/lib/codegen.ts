import { LAYER_DEFS, inputShapeFromParams } from "./layers";
import type { LayerNodeData, ProjectState } from "./types";
import { FASHION_MNIST } from "./types";
import { collectPyTorchBlocks, joinPyTorchClass } from "./colab";

type GenNode = { id: string; data: LayerNodeData };
type GenEdge = { source: string; target: string };

function topoSort(nodes: GenNode[], edges: GenEdge[]): string[] | null {
  const ids = new Set(nodes.map((n) => n.id));
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const id of ids) {
    incoming.set(id, []);
    outgoing.set(id, []);
  }
  for (const e of edges) {
    if (!ids.has(e.source) || !ids.has(e.target)) continue;
    incoming.get(e.target)!.push(e.source);
    outgoing.get(e.source)!.push(e.target);
  }
  const indeg = new Map<string, number>();
  for (const id of ids) indeg.set(id, incoming.get(id)!.length);
  const q = [...ids].filter((id) => indeg.get(id) === 0);
  const order: string[] = [];
  while (q.length) {
    const id = q.shift()!;
    order.push(id);
    for (const t of outgoing.get(id)!) {
      indeg.set(t, indeg.get(t)! - 1);
      if (indeg.get(t) === 0) q.push(t);
    }
  }
  if (order.length !== ids.size) return null;
  return order;
}

function safeName(layerType: string, index: number): string {
  return `${layerType.toLowerCase().replace(/[^a-z0-9]/g, "_")}_${index}`;
}

function getInChannels(data: LayerNodeData): number | null {
  if (data.inputShape?.length === 3) return data.inputShape[0];
  if (data.inputShape?.length === 2) return data.inputShape[0];
  return null;
}

function getInFeatures(data: LayerNodeData): number | null {
  if (data.inputShape?.length === 1) return data.inputShape[0];
  return null;
}

export type Framework = "pytorch" | "keras";

export type CodeGenResult = {
  code: string;
  error: string | null;
};

function generatePyTorch(
  nodes: GenNode[],
  edges: GenEdge[],
  order: string[],
  className: string,
  includeTrainStub: boolean,
): string {
  const { blocks, retVar, inputShape } = collectPyTorchBlocks(nodes, edges, order);
  return joinPyTorchClass(className, blocks, retVar, includeTrainStub, inputShape);
}

function generateKeras(
  nodes: GenNode[],
  edges: GenEdge[],
  order: string[],
  className: string,
): string {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const incoming = new Map<string, string[]>();
  for (const n of nodes) incoming.set(n.id, []);
  for (const e of edges) {
    if (byId.has(e.source) && byId.has(e.target)) {
      incoming.get(e.target)!.push(e.source);
    }
  }

  const names = new Map<string, string>();
  let idx = 0;
  for (const id of order) {
    if (byId.get(id)!.data.layerType === "Input") continue;
    names.set(id, safeName(byId.get(id)!.data.layerType, idx++));
  }

  const varNames = new Map<string, string>();
  const lines: string[] = [];

  const inputNode = nodes.find((n) => n.data.layerType === "Input");
  const inShape = inputShapeFromParams(inputNode?.data.params ?? {});
  // Keras uses channels_last for images
  let kerasInputShape: string;
  if (inShape.length === 3) {
    kerasInputShape = `(${inShape[1]}, ${inShape[2]}, ${inShape[0]})  # H, W, C`;
  } else if (inShape.length === 2) {
    kerasInputShape = `(${inShape[0]}, ${inShape[1]})`;
  } else {
    kerasInputShape = `(${inShape[0]},)`;
  }

  lines.push(`inputs = keras.Input(shape=${kerasInputShape})`);
  if (inputNode) varNames.set(inputNode.id, "inputs");

  for (const id of order) {
    const n = byId.get(id)!;
    const { layerType, params } = n.data;
    if (layerType === "Input") continue;

    const name = names.get(id)!;
    const preds = incoming.get(id)!;
    const srcVars = preds.map((p) => varNames.get(p) ?? "inputs");
    let expr = "";

    switch (layerType) {
      case "Conv2d":
        expr = `layers.Conv2D(${params.outChannels ?? 32}, ${params.kernelSize ?? 3}, strides=${params.stride ?? 1}, padding="${(params.padding ?? 0) > 0 ? "same" : "valid"}", name="${name}")(${srcVars[0]})`;
        break;
      case "Conv1d":
        expr = `layers.Conv1D(${params.outChannels ?? 64}, ${params.kernelSize ?? 3}, strides=${params.stride ?? 1}, padding="${(params.padding ?? 0) > 0 ? "same" : "valid"}", name="${name}")(${srcVars[0]})`;
        break;
      case "MaxPool2d":
        expr = `layers.MaxPooling2D(${params.poolKernel ?? 2}, strides=${params.poolStride ?? 2}, name="${name}")(${srcVars[0]})`;
        break;
      case "AvgPool2d":
        expr = `layers.AveragePooling2D(${params.poolKernel ?? 2}, strides=${params.poolStride ?? 2}, name="${name}")(${srcVars[0]})`;
        break;
      case "AdaptiveAvgPool2d":
        expr = `layers.GlobalAveragePooling2D(name="${name}")(${srcVars[0]})` ;
        break;
      case "BatchNorm2d":
        expr = `layers.BatchNormalization(name="${name}")(${srcVars[0]})`;
        break;
      case "BatchNorm1d":
        expr = `layers.BatchNormalization(name="${name}")(${srcVars[0]})`;
        break;
      case "GlobalAvgPool1d":
        expr = `layers.GlobalAveragePooling1D(name="${name}")(${srcVars[0]})`;
        break;
      case "LayerNorm":
        expr = `layers.LayerNormalization(name="${name}")(${srcVars[0]})`;
        break;
      case "Flatten":
        expr = `layers.Flatten(name="${name}")(${srcVars[0]})`;
        break;
      case "Reshape":
        expr = `layers.Reshape((${params.reshapeHeight ?? 28}, ${params.reshapeWidth ?? 28}, ${params.reshapeChannels ?? 1}), name="${name}")(${srcVars[0]})`;
        break;
      case "Linear":
        expr = `layers.Dense(${params.outFeatures ?? 128}, name="${name}")(${srcVars[0]})`;
        break;
      case "Output":
        expr = `layers.Dense(${params.numClasses ?? FASHION_MNIST.numClasses}, name="${name}")(${srcVars[0]})`;
        break;
      case "Embedding":
        expr = `layers.Embedding(${params.numEmbeddings ?? 1000}, ${params.embeddingDim ?? 64}, name="${name}")(${srcVars[0]})`;
        break;
      case "LSTM":
        expr = `layers.LSTM(${params.hiddenSize ?? 128}, return_sequences=True, name="${name}")(${srcVars[0]})`;
        break;
      case "GRU":
        expr = `layers.GRU(${params.hiddenSize ?? 128}, return_sequences=True, name="${name}")(${srcVars[0]})`;
        break;
      case "MultiheadAttention":
        expr = `layers.MultiHeadAttention(${params.numHeads ?? 4}, key_dim=${Math.floor((params.embedDim ?? 64) / (params.numHeads ?? 4))}, name="${name}")(${srcVars[0]}, ${srcVars[0]})`;
        break;
      case "PositionalEncoding":
        lines.push(`# Positional encoding — add a learned/sinusoidal PE in a custom layer if needed`);
        expr = `${srcVars[0]}  # TODO: add PositionalEncoding`;
        break;
      case "TransformerEncoder":
        expr = `layers.TransformerEncoder(num_layers=${params.numLayers ?? 2}, intermediate_dim=${params.ffDim ?? 256}, num_heads=${params.numHeads ?? 4}, name="${name}")(${srcVars[0]})`;
        break;
      case "TransformerDecoder":
        lines.push(`# Keras has no single Decoder layer matching PyTorch — approximate with another encoder stack`);
        expr = `layers.TransformerEncoder(num_layers=${params.numLayers ?? 2}, intermediate_dim=${params.ffDim ?? 256}, num_heads=${params.numHeads ?? 4}, name="${name}")(${srcVars[0]})`;
        break;
      case "ReLU":
        expr = `layers.ReLU(name="${name}")(${srcVars[0]})`;
        break;
      case "LeakyReLU":
        expr = `layers.LeakyReLU(negative_slope=${params.negativeSlope ?? 0.01}, name="${name}")(${srcVars[0]})`;
        break;
      case "GELU":
        expr = `layers.Activation("gelu", name="${name}")(${srcVars[0]})`;
        break;
      case "Sigmoid":
        expr = `layers.Activation("sigmoid", name="${name}")(${srcVars[0]})`;
        break;
      case "Tanh":
        expr = `layers.Activation("tanh", name="${name}")(${srcVars[0]})`;
        break;
      case "Dropout":
        expr = `layers.Dropout(${params.p ?? 0.25}, name="${name}")(${srcVars[0]})`;
        break;
      case "Softmax":
        expr = `layers.Softmax(name="${name}")(${srcVars[0]})`;
        break;
      case "Add":
        expr = `layers.Add(name="${name}")([${srcVars.join(", ")}])`;
        break;
      case "Concat":
        expr = `layers.Concatenate(axis=${(params.concatDim ?? 0) === 0 ? -1 : (params.concatDim ?? 0) + 1}, name="${name}")([${srcVars.join(", ")}])`;
        break;
      case "LoopBlock": {
        const repeats = params.repeats ?? 2;
        lines.push(`# Loop block ${name}: shared dense refine ×${repeats}`);
        lines.push(`${name}_h = ${srcVars[0]}`);
        lines.push(`for _ in range(${repeats}):`);
        lines.push(`    ${name}_h = layers.Dense(${n.data.inputShape?.[n.data.inputShape.length - 1] ?? 64}, activation="relu")(${name}_h)`);
        varNames.set(id, `${name}_h`);
        continue;
      }
      default:
        expr = `${srcVars[0]}  # unsupported ${layerType}`;
    }

    lines.push(`${name} = ${expr}  # ${LAYER_DEFS[layerType].shortLabel}`);
    varNames.set(id, name);
  }

  const lastId = [...order].reverse().find((id) => byId.get(id)!.data.layerType !== "Input");
  const outVar = lastId ? (varNames.get(lastId) ?? "inputs") : "inputs";

  return `"""
Generated by easy deep learning — annotated Keras / TensorFlow export
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_${className.toLowerCase()}():
${lines.map((l) => "    " + l).join("\n")}
    model = keras.Model(inputs=inputs, outputs=${outVar}, name="${className}")
    return model


if __name__ == "__main__":
    model = build_${className.toLowerCase()}()
    model.summary()
`;
}

export function generateCode(
  nodes: GenNode[],
  edges: GenEdge[],
  options?: {
    className?: string;
    includeTrainStub?: boolean;
    framework?: Framework;
  },
): CodeGenResult {
  const className = options?.className ?? "VisualNet";
  const framework = options?.framework ?? "pytorch";

  if (nodes.length === 0) {
    return { code: "", error: "Add some layers to the canvas first." };
  }

  const order = topoSort(nodes, edges);
  if (!order) {
    return {
      code: "",
      error: "Can't generate code: the graph has a cycle. Use a Loop Block instead.",
    };
  }

  if (!nodes.some((n) => n.data.layerType === "Input")) {
    return { code: "", error: "Add an Input node to start the network." };
  }

  if (framework === "keras") {
    return { code: generateKeras(nodes, edges, order, className), error: null };
  }

  return {
    code: generatePyTorch(
      nodes,
      edges,
      order,
      className,
      options?.includeTrainStub ?? true,
    ),
    error: null,
  };
}

/** @deprecated use generateCode */
export function generatePyTorchCode(
  nodes: GenNode[],
  edges: GenEdge[],
  options?: { className?: string; includeTrainStub?: boolean },
): CodeGenResult {
  return generateCode(nodes, edges, { ...options, framework: "pytorch" });
}

export function projectToGenInput(state: ProjectState) {
  return {
    nodes: state.nodes.map((n) => ({ id: n.id, data: n.data })),
    edges: state.edges.map((e) => ({ source: e.source, target: e.target })),
  };
}
