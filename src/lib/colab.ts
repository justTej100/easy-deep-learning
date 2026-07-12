import { LAYER_DEFS, formatShape, inputShapeFromParams } from "./layers";
import type { LayerNodeData, LayerType, ProjectState, TensorShape } from "./types";
import { FASHION_MNIST } from "./types";

type GenNode = { id: string; data: LayerNodeData };
type GenEdge = { source: string; target: string };

export type NotebookCell = {
  cell_type: "markdown" | "code";
  source: string[];
};

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

function toSources(text: string): string[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  return lines.map((line, i) => (i < lines.length - 1 ? `${line}\n` : line));
}

function md(text: string): NotebookCell {
  return { cell_type: "markdown", source: toSources(text.trim() + "\n") };
}

function code(text: string): NotebookCell {
  return { cell_type: "code", source: toSources(text.replace(/^\n+/, "").replace(/\n+$/, "") + "\n") };
}

type LayerBlock = {
  id: string;
  layerType: LayerType;
  name: string | null;
  label: string;
  why: string;
  analogy: string;
  inputShape: TensorShape | null;
  outputShape: TensorShape | null;
  /** Lines registered in __init__ (already indented with 8 spaces for class body) */
  initLines: string[];
  /** Forward lines without class indent */
  forwardLines: string[];
};

/**
 * Walk the graph once and collect per-layer init/forward snippets
 * used by both .py export and Colab cell-per-block notebooks.
 */
function collectPyTorchBlocks(
  nodes: GenNode[],
  edges: GenEdge[],
  order: string[],
): { blocks: LayerBlock[]; retVar: string; inputShape: number[] } {
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
  const blocks: LayerBlock[] = [];
  const inputParams =
    nodes.find((n) => n.data.layerType === "Input")?.data.params ?? {};
  const inputShape = inputShapeFromParams(inputParams);

  for (const id of order) {
    const n = byId.get(id)!;
    const { layerType, params } = n.data;
    const def = LAYER_DEFS[layerType];
    const initLines: string[] = [];
    const forwardLines: string[] = [];

    const emitInit = (line: string, comment: string) => {
      initLines.push(`        # ${comment}`);
      initLines.push(`        ${line}`);
    };

    if (layerType === "Input") {
      forwardLines.push(`# x batch shape: (N, ${inputShape.join(", ")})`);
      varNames.set(id, "x");
      blocks.push({
        id,
        layerType,
        name: null,
        label: def.label,
        why: def.why,
        analogy: def.analogy,
        inputShape: n.data.inputShape ?? null,
        outputShape: n.data.outputShape ?? null,
        initLines: [],
        forwardLines,
      });
      continue;
    }

    const name = names.get(id)!;
    const preds = incoming.get(id)!;
    const srcVars = preds.map((p) => varNames.get(p) ?? "x");

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
      case "BatchNorm1d": {
        const ch =
          n.data.inputShape?.length === 2
            ? n.data.inputShape[0]
            : (n.data.inputShape?.[0] ?? 64);
        emitInit(`self.${name} = nn.BatchNorm1d(${ch})`, LAYER_DEFS.BatchNorm1d.why);
        break;
      }
      case "GlobalAvgPool1d":
        emitInit(
          `self.${name} = nn.AdaptiveAvgPool1d(1)  # global avg pool over length`,
          LAYER_DEFS.GlobalAvgPool1d.why,
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
      case "PositionalEncoding": {
        const d =
          n.data.inputShape?.length === 2
            ? n.data.inputShape[1]
            : (params.embedDim ?? 64);
        const maxLen = params.maxLen ?? 512;
        initLines.push(`        # ${LAYER_DEFS.PositionalEncoding.why}`);
        initLines.push(`        pe = torch.zeros(1, ${maxLen}, ${d})`);
        initLines.push(
          `        position = torch.arange(0, ${maxLen}, dtype=torch.float32).unsqueeze(1)`,
        );
        initLines.push(
          `        div_term = torch.exp(torch.arange(0, ${d}, 2).float() * (-math.log(10000.0) / ${d}))`,
        );
        initLines.push(`        pe[0, :, 0::2] = torch.sin(position * div_term)`);
        initLines.push(`        pe[0, :, 1::2] = torch.cos(position * div_term)`);
        initLines.push(`        self.register_buffer("${name}_pe", pe)`);
        break;
      }
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
      case "TransformerEncoder": {
        const d = params.embedDim ?? 64;
        const heads = params.numHeads ?? 4;
        const ff = params.ffDim ?? 256;
        const layers = params.numLayers ?? 2;
        initLines.push(`        # ${LAYER_DEFS.TransformerEncoder.why}`);
        initLines.push(
          `        _enc = nn.TransformerEncoderLayer(d_model=${d}, nhead=${heads}, dim_feedforward=${ff}, batch_first=True)`,
        );
        initLines.push(
          `        self.${name} = nn.TransformerEncoder(_enc, num_layers=${layers})`,
        );
        break;
      }
      case "TransformerDecoder": {
        const d = params.embedDim ?? 64;
        const heads = params.numHeads ?? 4;
        const ff = params.ffDim ?? 256;
        const layers = params.numLayers ?? 2;
        initLines.push(`        # ${LAYER_DEFS.TransformerDecoder.why}`);
        initLines.push(
          `        _dec = nn.TransformerDecoderLayer(d_model=${d}, nhead=${heads}, dim_feedforward=${ff}, batch_first=True)`,
        );
        initLines.push(
          `        self.${name} = nn.TransformerDecoder(_dec, num_layers=${layers})`,
        );
        break;
      }
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

    if (layerType === "Add") {
      forwardLines.push(
        `${name}_out = ${srcVars[0]} + ${srcVars[1] ?? srcVars[0]}  # residual add`,
      );
      varNames.set(id, `${name}_out`);
    } else if (layerType === "Concat") {
      const dim = (params.concatDim ?? 0) + 1;
      forwardLines.push(
        `${name}_out = torch.cat([${srcVars.join(", ")}], dim=${dim})  # concat`,
      );
      varNames.set(id, `${name}_out`);
    } else if (layerType === "Reshape") {
      forwardLines.push(
        `${name}_out = ${srcVars[0]}.view(-1, ${params.reshapeChannels ?? 1}, ${params.reshapeHeight ?? 28}, ${params.reshapeWidth ?? 28})  # reshape`,
      );
      varNames.set(id, `${name}_out`);
    } else if (layerType === "LSTM" || layerType === "GRU") {
      forwardLines.push(`${name}_out, _ = self.${name}(${srcVars[0]})  # ${layerType}`);
      varNames.set(id, `${name}_out`);
    } else if (layerType === "MultiheadAttention") {
      forwardLines.push(
        `${name}_out, _ = self.${name}(${srcVars[0]}, ${srcVars[0]}, ${srcVars[0]})  # self-attention`,
      );
      varNames.set(id, `${name}_out`);
    } else if (layerType === "PositionalEncoding") {
      forwardLines.push(
        `${name}_out = ${srcVars[0]} + self.${name}_pe[:, : ${srcVars[0]}.size(1)]  # add positions`,
      );
      varNames.set(id, `${name}_out`);
    } else if (layerType === "TransformerEncoder") {
      forwardLines.push(`${name}_out = self.${name}(${srcVars[0]})  # encoder`);
      varNames.set(id, `${name}_out`);
    } else if (layerType === "TransformerDecoder") {
      forwardLines.push(
        `${name}_out = self.${name}(${srcVars[0]}, ${srcVars[0]})  # decoder (tgt, memory)`,
      );
      varNames.set(id, `${name}_out`);
    } else if (layerType === "GlobalAvgPool1d") {
      forwardLines.push(
        `${name}_out = self.${name}(${srcVars[0]}).squeeze(-1)  # Global Avg Pool 1D`,
      );
      varNames.set(id, `${name}_out`);
    } else if (layerType === "LoopBlock") {
      const repeats = params.repeats ?? 2;
      forwardLines.push(`${name}_out = ${srcVars[0]}`);
      forwardLines.push(`for _ in range(${repeats}):`);
      forwardLines.push(`    ${name}_out = self.${name}_shared(${name}_out)`);
      varNames.set(id, `${name}_out`);
    } else {
      forwardLines.push(
        `${name}_out = self.${name}(${srcVars[0]})  # ${LAYER_DEFS[layerType].label}`,
      );
      varNames.set(id, `${name}_out`);
    }

    blocks.push({
      id,
      layerType,
      name,
      label: def.label,
      why: def.why,
      analogy: def.analogy,
      inputShape: n.data.inputShape ?? null,
      outputShape: n.data.outputShape ?? null,
      initLines,
      forwardLines,
    });
  }

  const lastId = [...order]
    .reverse()
    .find((id) => byId.get(id)!.data.layerType !== "Input");
  const retVar = lastId ? (varNames.get(lastId) ?? "x") : "x";

  return { blocks, retVar, inputShape };
}

function joinPyTorchClass(
  className: string,
  blocks: LayerBlock[],
  retVar: string,
  includeTrainStub: boolean,
  inputShape: number[],
): string {
  const initLines = blocks.flatMap((b) => b.initLines);
  const forwardLines = blocks.flatMap((b) => b.forwardLines);

  const trainStub = includeTrainStub
    ? `

if __name__ == "__main__":
    import torch
    model = ${className}()
    # Dummy batch — replace with your DataLoader
    x = torch.randn(2, ${inputShape.join(", ")})
    y = model(x)
    print(y.shape)
`
    : "";

  return `"""
Generated by easy deep learning — annotated PyTorch export
"""
import math
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

/**
 * Build a Colab .ipynb for the whole project: intro, imports,
 * one markdown + code cell per layer block, then assemble + smoke test.
 */
export function buildColabNotebook(
  nodes: GenNode[],
  edges: GenEdge[],
  options?: { className?: string },
): { notebook: object; error: string | null } {
  const className = options?.className ?? "VisualNet";

  if (nodes.length === 0) {
    return { notebook: {}, error: "Add some layers to the canvas first." };
  }
  const order = topoSort(nodes, edges);
  if (!order) {
    return {
      notebook: {},
      error: "Can't export: the graph has a cycle. Use a Loop Block instead.",
    };
  }
  if (!nodes.some((n) => n.data.layerType === "Input")) {
    return { notebook: {}, error: "Add an Input node to start the network." };
  }

  const { blocks, retVar, inputShape } = collectPyTorchBlocks(nodes, edges, order);
  const cells: NotebookCell[] = [];

  cells.push(
    md(`# Fashion-MNIST model — easy deep learning

Generated from your visual graph. Each **block** below is one layer from the canvas.

1. Runtime → **Run all**
2. Or run cells top-to-bottom: imports → each block → assemble → smoke test

PyTorch is pre-installed on Colab.`),
  );

  cells.push(
    code(`import math
import torch
import torch.nn as nn

print("torch", torch.__version__)`),
  );

  cells.push(
    md(`## Layer blocks

Run each cell in order. Init lines register modules; forward lines show how tensors move.`),
  );

  blocks.forEach((b, i) => {
    const shapeLine = `${formatShape(b.inputShape)} → ${formatShape(b.outputShape)}`;
    cells.push(
      md(`### ${i + 1}. ${b.label}

**Shape:** \`${shapeLine}\`

${b.why}

*${b.analogy}*`),
    );

    if (b.layerType === "Input") {
      cells.push(
        code(`# Block: Input
# Batch tensor enters the network here.
# Expected feature shape (no batch): ${inputShape.join(" × ")}
# In forward(): x has shape (N, ${inputShape.join(", ")})`),
      );
      return;
    }

    const initBody =
      b.initLines.length > 0
        ? b.initLines.map((l) => l.replace(/^        /, "")).join("\n")
        : `# (no parameters — handled in forward)`;
    const fwdBody = b.forwardLines.map((l) => `# ${l}`).join("\n");

    cells.push(
      code(`# Block: ${b.label}${b.name ? ` (${b.name})` : ""}
# --- __init__ ---
${initBody}

# --- forward (wired in Assemble cell) ---
${fwdBody}`),
    );
  });

  const initJoined = blocks.flatMap((b) => b.initLines);
  const forwardJoined = blocks.flatMap((b) => b.forwardLines);

  cells.push(
    md(`## Assemble model

Builds the full \`nn.Module\` from every block above.`),
  );

  cells.push(
    code(`class ${className}(nn.Module):
    def __init__(self):
        super().__init__()
${initJoined.length ? initJoined.join("\n") : "        pass"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
${forwardJoined.map((l) => "        " + l).join("\n")}
        return ${retVar}

model = ${className}()
print(model)`),
  );

  cells.push(
    md(`## Smoke test

Dummy batch — replace with a real Fashion-MNIST DataLoader when you train.`),
  );

  cells.push(
    code(`x = torch.randn(2, ${inputShape.join(", ")})
y = model(x)
print("input ", tuple(x.shape))
print("output", tuple(y.shape))`),
  );

  cells.push(
    md(`## Optional: Fashion-MNIST training sketch

Uncomment and run when you are ready to train on GPU.`),
  );

  cells.push(
    code(`# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,)),
# ])
# train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
# train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ${className}().to(device)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()
#
# model.train()
# for images, labels in train_loader:
#     images, labels = images.to(device), labels.to(device)
#     opt.zero_grad()
#     logits = model(images)
#     loss = loss_fn(logits, labels)
#     loss.backward()
#     opt.step()
#     break  # remove to train a full epoch
# print("one-step loss", float(loss))`),
  );

  const notebook = {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
      language_info: {
        name: "python",
        version: "3.10.0",
      },
      colab: { provenance: [], toc_visible: true },
    },
    cells: cells.map((c) =>
      c.cell_type === "code"
        ? {
            cell_type: "code",
            metadata: {},
            execution_count: null,
            outputs: [],
            source: c.source,
          }
        : {
            cell_type: "markdown",
            metadata: {},
            source: c.source,
          },
    ),
  };

  return { notebook, error: null };
}

export function buildColabNotebookFromProject(
  state: ProjectState,
): { notebook: object; error: string | null } {
  return buildColabNotebook(
    state.nodes.map((n) => ({ id: n.id, data: n.data })),
    state.edges.map((e) => ({ source: e.source, target: e.target })),
  );
}

// Re-export helpers used by codegen.py path after we wire collect into generateCode
export { collectPyTorchBlocks, joinPyTorchClass, topoSort };
export type { GenNode, GenEdge, LayerBlock };
