import { LAYER_DEFS, inputShapeFromParams } from "./layers";
import type { LayerNodeData, ProjectState } from "./types";
import { FASHION_MNIST } from "./types";

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
  const initLines: string[] = [];
  const forwardLines: string[] = [];

  for (const id of order) {
    const n = byId.get(id)!;
    const { layerType, params } = n.data;

    if (layerType === "Input") {
      const shape = inputShapeFromParams(params);
      forwardLines.push(`# x batch shape: (N, ${shape.join(", ")})`);
      varNames.set(id, "x");
      continue;
    }

    const name = names.get(id)!;
    const preds = incoming.get(id)!;
    const srcVars = preds.map((p) => varNames.get(p) ?? "x");

    const emitInit = (code: string, comment: string) => {
      initLines.push(`        # ${comment}`);
      initLines.push(`        ${code}`);
    };

    switch (layerType) {
      case "Conv2d":
        emitInit(
          `self.${name} = nn.Conv2d(${getInChannels(n.data) ?? 1}, ${params.outChannels ?? 32}, kernel_size=${params.kernelSize ?? 3}, stride=${params.stride ?? 1}, padding=${params.padding ?? 0})`,
          LAYER_DEFS.Conv2d.why,
        );
        break;
      case "Conv1d":
        emitInit(
          `self.${name} = nn.Conv1d(${getInChannels(n.data) ?? 1}, ${params.outChannels ?? 64}, kernel_size=${params.kernelSize ?? 3}, stride=${params.stride ?? 1}, padding=${params.padding ?? 0})`,
          LAYER_DEFS.Conv1d.why,
        );
        break;
      case "MaxPool2d":
        emitInit(
          `self.${name} = nn.MaxPool2d(kernel_size=${params.poolKernel ?? 2}, stride=${params.poolStride ?? 2})`,
          LAYER_DEFS.MaxPool2d.why,
        );
        break;
      case "AvgPool2d":
        emitInit(
          `self.${name} = nn.AvgPool2d(kernel_size=${params.poolKernel ?? 2}, stride=${params.poolStride ?? 2})`,
          LAYER_DEFS.AvgPool2d.why,
        );
        break;
      case "AdaptiveAvgPool2d":
        emitInit(
          `self.${name} = nn.AdaptiveAvgPool2d((${params.outputSize ?? 1}, ${params.outputSize ?? 1}))`,
          LAYER_DEFS.AdaptiveAvgPool2d.why,
        );
        break;
      case "BatchNorm2d":
        emitInit(
          `self.${name} = nn.BatchNorm2d(${getInChannels(n.data) ?? 32})`,
          LAYER_DEFS.BatchNorm2d.why,
        );
        break;
      case "LayerNorm": {
        const shape = n.data.inputShape ?? [64];
        emitInit(
          `self.${name} = nn.LayerNorm(${shape.length === 1 ? shape[0] : shape[shape.length - 1]})`,
          LAYER_DEFS.LayerNorm.why,
        );
        break;
      }
      case "Flatten":
        emitInit(`self.${name} = nn.Flatten()`, LAYER_DEFS.Flatten.why);
        break;
      case "Reshape":
        // handled in forward
        break;
      case "Linear":
      case "Output": {
        const out =
          layerType === "Output"
            ? (params.numClasses ?? FASHION_MNIST.numClasses)
            : (params.outFeatures ?? 128);
        emitInit(
          `self.${name} = nn.Linear(${getInFeatures(n.data) ?? "in_features"}, ${out})`,
          layerType === "Output" ? LAYER_DEFS.Output.why : LAYER_DEFS.Linear.why,
        );
        break;
      }
      case "Embedding":
        emitInit(
          `self.${name} = nn.Embedding(${params.numEmbeddings ?? 1000}, ${params.embeddingDim ?? 64})`,
          LAYER_DEFS.Embedding.why,
        );
        break;
      case "LSTM":
      case "GRU": {
        const Cls = layerType === "LSTM" ? "LSTM" : "GRU";
        const inSize =
          n.data.inputShape?.length === 2
            ? n.data.inputShape[1]
            : (n.data.inputShape?.[0] ?? 64);
        emitInit(
          `self.${name} = nn.${Cls}(${inSize}, ${params.hiddenSize ?? 128}, num_layers=${params.numLayers ?? 1}, batch_first=True, bidirectional=${(params.bidirectional ?? 0) === 1})`,
          LAYER_DEFS[layerType].why,
        );
        break;
      }
      case "MultiheadAttention":
        emitInit(
          `self.${name} = nn.MultiheadAttention(${params.embedDim ?? 64}, ${params.numHeads ?? 4}, batch_first=True)`,
          LAYER_DEFS.MultiheadAttention.why,
        );
        break;
      case "ReLU":
        emitInit(`self.${name} = nn.ReLU()`, LAYER_DEFS.ReLU.why);
        break;
      case "LeakyReLU":
        emitInit(
          `self.${name} = nn.LeakyReLU(negative_slope=${params.negativeSlope ?? 0.01})`,
          LAYER_DEFS.LeakyReLU.why,
        );
        break;
      case "GELU":
        emitInit(`self.${name} = nn.GELU()`, LAYER_DEFS.GELU.why);
        break;
      case "Sigmoid":
        emitInit(`self.${name} = nn.Sigmoid()`, LAYER_DEFS.Sigmoid.why);
        break;
      case "Tanh":
        emitInit(`self.${name} = nn.Tanh()`, LAYER_DEFS.Tanh.why);
        break;
      case "Dropout":
        emitInit(`self.${name} = nn.Dropout(p=${params.p ?? 0.25})`, LAYER_DEFS.Dropout.why);
        break;
      case "Softmax":
        emitInit(
          `self.${name} = nn.Softmax(dim=${params.dim ?? -1})`,
          LAYER_DEFS.Softmax.why,
        );
        break;
      case "LoopBlock": {
        const repeats = params.repeats ?? 2;
        const shape = n.data.inputShape;
        initLines.push(`        # Loop Block: shared weights ×${repeats}`);
        if (shape?.length === 3) {
          const c = shape[0];
          initLines.push(`        self.${name}_shared = nn.Sequential(`);
          initLines.push(`            nn.Conv2d(${c}, ${c}, 3, padding=1),`);
          initLines.push(`            nn.ReLU(),`);
          initLines.push(`        )`);
        } else if (shape?.length === 1) {
          const d = shape[0];
          initLines.push(`        self.${name}_shared = nn.Sequential(`);
          initLines.push(`            nn.Linear(${d}, ${d}),`);
          initLines.push(`            nn.ReLU(),`);
          initLines.push(`        )`);
        } else {
          initLines.push(`        self.${name}_shared = nn.Identity()`);
        }
        break;
      }
      case "Add":
      case "Concat":
        break;
      default:
        break;
    }

    // Forward
    if (layerType === "Add") {
      forwardLines.push(
        `${name}_out = ${srcVars[0]} + ${srcVars[1] ?? srcVars[0]}  # residual add`,
      );
      varNames.set(id, `${name}_out`);
      continue;
    }
    if (layerType === "Concat") {
      const dim = (params.concatDim ?? 0) + 1; // +1 for batch dim
      forwardLines.push(
        `${name}_out = torch.cat([${srcVars.join(", ")}], dim=${dim})  # concat`,
      );
      varNames.set(id, `${name}_out`);
      continue;
    }
    if (layerType === "Reshape") {
      forwardLines.push(
        `${name}_out = ${srcVars[0]}.view(-1, ${params.reshapeChannels ?? 1}, ${params.reshapeHeight ?? 28}, ${params.reshapeWidth ?? 28})  # reshape`,
      );
      varNames.set(id, `${name}_out`);
      continue;
    }
    if (layerType === "LSTM" || layerType === "GRU") {
      forwardLines.push(`${name}_out, _ = self.${name}(${srcVars[0]})  # ${layerType}`);
      varNames.set(id, `${name}_out`);
      continue;
    }
    if (layerType === "MultiheadAttention") {
      forwardLines.push(
        `${name}_out, _ = self.${name}(${srcVars[0]}, ${srcVars[0]}, ${srcVars[0]})  # self-attention`,
      );
      varNames.set(id, `${name}_out`);
      continue;
    }
    if (layerType === "LoopBlock") {
      const repeats = params.repeats ?? 2;
      forwardLines.push(`${name}_out = ${srcVars[0]}`);
      forwardLines.push(`for _ in range(${repeats}):`);
      forwardLines.push(`    ${name}_out = self.${name}_shared(${name}_out)`);
      varNames.set(id, `${name}_out`);
      continue;
    }

    forwardLines.push(
      `${name}_out = self.${name}(${srcVars[0]})  # ${LAYER_DEFS[layerType].label}`,
    );
    varNames.set(id, `${name}_out`);
  }

  const lastId = [...order].reverse().find((id) => byId.get(id)!.data.layerType !== "Input");
  const retVar = lastId ? (varNames.get(lastId) ?? "x") : "x";

  const trainStub = includeTrainStub
    ? `

if __name__ == "__main__":
    import torch
    model = ${className}()
    # Dummy batch — replace with your DataLoader
    x = torch.randn(2, ${inputShapeFromParams(nodes.find((n) => n.data.layerType === "Input")?.data.params ?? {}).join(", ")})
    y = model(x)
    print(y.shape)
`
    : "";

  return `"""
Generated by easy deep learning — annotated PyTorch export
"""
import torch
import torch.nn as nn


class ${className}(nn.Module):
    def __init__(self):
        super().__init__()
${initLines.length ? initLines.join("\n") : "        pass"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
${forwardLines.map((l) => "        " + l).join("\n")}
        return ${retVar}
${trainStub}`;
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
