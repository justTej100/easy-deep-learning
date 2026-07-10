import { LAYER_DEFS } from "./layers";
import type { LayerNodeData, LayerParams, ProjectState } from "./types";
import { FASHION_MNIST } from "./types";

type GenNode = {
  id: string;
  data: LayerNodeData;
};

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
  const base = layerType.toLowerCase().replace(/[^a-z0-9]/g, "_");
  return `${base}_${index}`;
}

function initLine(
  name: string,
  layerType: LayerNodeData["layerType"],
  params: LayerParams,
  inFeatures: number | null,
): { code: string; comment: string } | null {
  switch (layerType) {
    case "Input":
      return null;
    case "Conv2d":
      return {
        code: `self.${name} = nn.Conv2d(in_channels, ${params.outChannels ?? 32}, kernel_size=${params.kernelSize ?? 3}, stride=${params.stride ?? 1}, padding=${params.padding ?? 0})`,
        comment: `Learn ${params.outChannels ?? 32} local filters (edges/textures). in_channels is set dynamically from the previous tensor.`,
      };
    case "MaxPool2d":
      return {
        code: `self.${name} = nn.MaxPool2d(kernel_size=${params.poolKernel ?? 2}, stride=${params.poolStride ?? 2})`,
        comment: "Downsample spatial size; keep the strongest activation in each window.",
      };
    case "Flatten":
      return {
        code: `self.${name} = nn.Flatten()`,
        comment: "Collapse C×H×W into a single feature vector for Linear layers.",
      };
    case "Linear":
      return {
        code: `self.${name} = nn.Linear(${inFeatures ?? "in_features"}, ${params.outFeatures ?? 128})`,
        comment: `Fully connect every input feature to ${params.outFeatures ?? 128} outputs.`,
      };
    case "ReLU":
      return {
        code: `self.${name} = nn.ReLU()`,
        comment: "Non-linearity: zero out negatives so stacked layers aren't just one linear map.",
      };
    case "Dropout":
      return {
        code: `self.${name} = nn.Dropout(p=${params.p ?? 0.25})`,
        comment: `Randomly drop ${Math.round((params.p ?? 0.25) * 100)}% of activations during training to reduce overfitting.`,
      };
    case "Softmax":
      return {
        code: `self.${name} = nn.Softmax(dim=${params.dim ?? -1})`,
        comment:
          "Turn logits into probabilities. Optional for training with CrossEntropyLoss (which already applies log-softmax).",
      };
    case "Output":
      return {
        code: `self.${name} = nn.Linear(${inFeatures ?? "in_features"}, ${params.numClasses ?? FASHION_MNIST.numClasses})`,
        comment: `Classification head: one score per Fashion-MNIST class (${params.numClasses ?? FASHION_MNIST.numClasses} total).`,
      };
    case "LoopBlock":
      return {
        code: `# Loop block "${name}" — shared weights applied ${params.repeats ?? 2}× (see forward)`,
        comment:
          "Recurrent depth: reuse the same parameters to trade inference compute for capacity.",
      };
    default:
      return null;
  }
}

function forwardLine(
  name: string,
  layerType: LayerNodeData["layerType"],
  params: LayerParams,
): string | null {
  switch (layerType) {
    case "Input":
      return null;
    case "LoopBlock": {
      const n = params.repeats ?? 2;
      return [
        `# Shared-weight loop (research idea: looped / recurrent-depth transformers)`,
        `for _ in range(${n}):`,
        `    x = self.${name}_shared(x)  # same weights each pass`,
      ].join("\n        ");
    }
    default:
      return `x = self.${name}(x)`;
  }
}

/** Infer in_channels / in_features from propagated shapes on node data. */
function getInFeatures(data: LayerNodeData): number | null {
  if (!data.inputShape) return null;
  if (data.layerType === "Linear" || data.layerType === "Output") {
    if (data.inputShape.length === 1) return data.inputShape[0];
    return null;
  }
  if (data.layerType === "Conv2d") {
    if (data.inputShape.length === 3) return data.inputShape[0];
  }
  return null;
}

function resolveInFeatures(
  data: LayerNodeData,
  prevOut: number[] | null | undefined,
): number | null {
  const fromSelf = getInFeatures(data);
  if (fromSelf != null) return fromSelf;
  if (!prevOut) return null;
  if (data.layerType === "Linear" || data.layerType === "Output") {
    if (prevOut.length === 1) return prevOut[0];
    return prevOut.reduce((a, b) => a * b, 1);
  }
  if (data.layerType === "Conv2d" && prevOut.length === 3) return prevOut[0];
  return null;
}

export type CodeGenResult = {
  code: string;
  error: string | null;
};

export function generatePyTorchCode(
  nodes: GenNode[],
  edges: GenEdge[],
  options?: { className?: string; includeTrainStub?: boolean },
): CodeGenResult {
  const className = options?.className ?? "FashionMNISTNet";
  const includeTrainStub = options?.includeTrainStub ?? true;

  if (nodes.length === 0) {
    return { code: "", error: "Add some layers to the canvas first." };
  }

  const order = topoSort(nodes, edges);
  if (!order) {
    return {
      code: "",
      error:
        "Can't generate code: the graph has a cycle. Remove the cycle, or use a Loop Block in Research mode.",
    };
  }

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const inputNodes = nodes.filter((n) => n.data.layerType === "Input");
  if (inputNodes.length === 0) {
    return { code: "", error: "Add an Input node (Fashion-MNIST) to start the network." };
  }

  // Walk path and assign module names
  const names = new Map<string, string>();
  let idx = 0;
  for (const id of order) {
    const n = byId.get(id)!;
    if (n.data.layerType === "Input") continue;
    names.set(id, safeName(n.data.layerType, idx++));
  }

  const initLines: string[] = [];
  const forwardLines: string[] = [];
  let prevOut: number[] | null = [...FASHION_MNIST.inputShape];

  for (const id of order) {
    const n = byId.get(id)!;
    const { layerType, params } = n.data;
    if (layerType === "Input") {
      forwardLines.push(
        `# x starts as a Fashion-MNIST batch: (N, ${FASHION_MNIST.inputShape.join(", ")})`,
      );
      prevOut = n.data.outputShape ? [...n.data.outputShape] : [...FASHION_MNIST.inputShape];
      continue;
    }

    const name = names.get(id)!;
    const inFeat = resolveInFeatures(n.data, prevOut);

    if (layerType === "Conv2d") {
      const inCh = inFeat ?? FASHION_MNIST.inputShape[0];
      initLines.push(`        # ${LAYER_DEFS.Conv2d.why}`);
      initLines.push(
        `        self.${name} = nn.Conv2d(${inCh}, ${params.outChannels ?? 32}, kernel_size=${params.kernelSize ?? 3}, stride=${params.stride ?? 1}, padding=${params.padding ?? 0})  # ${params.outChannels ?? 32} filters looking for local patterns`,
      );
      forwardLines.push(`x = self.${name}(x)  # conv → feature maps`);
      prevOut = n.data.outputShape ? [...n.data.outputShape] : null;
      continue;
    }

    if (layerType === "LoopBlock") {
      const repeats = params.repeats ?? 2;
      const shape = n.data.inputShape ?? prevOut;
      initLines.push(`        # --- Loop Block: shared weights, applied ${repeats}× ---`);
      initLines.push(
        `        # Idea: looped/recurrent-depth models spend more compute at inference without new parameters.`,
      );
      if (shape && shape.length === 3) {
        const c = shape[0];
        initLines.push(`        self.${name}_shared = nn.Sequential(`);
        initLines.push(
          `            nn.Conv2d(${c}, ${c}, kernel_size=3, padding=1),  # shape-preserving conv`,
        );
        initLines.push(`            nn.ReLU(),`);
        initLines.push(`        )`);
      } else if (shape && shape.length === 1) {
        const d = shape[0];
        initLines.push(`        self.${name}_shared = nn.Sequential(`);
        initLines.push(`            nn.Linear(${d}, ${d}),  # shape-preserving linear`);
        initLines.push(`            nn.ReLU(),`);
        initLines.push(`        )`);
      } else {
        initLines.push(
          `        self.${name}_shared = nn.Identity()  # connect the loop so shapes resolve`,
        );
      }
      forwardLines.push(
        `# Recurrent depth: refine representation ${repeats} times with shared weights`,
      );
      forwardLines.push(`for _ in range(${repeats}):`);
      forwardLines.push(`    x = self.${name}_shared(x)`);
      prevOut = n.data.outputShape ? [...n.data.outputShape] : shape ? [...shape] : prevOut;
      continue;
    }

    const line = initLine(name, layerType, params, inFeat);
    if (line) {
      initLines.push(`        # ${line.comment}`);
      initLines.push(`        ${line.code}`);
    }
    const fwd = forwardLine(name, layerType, params);
    if (fwd) {
      if (fwd.includes("\n")) {
        forwardLines.push(fwd);
      } else {
        forwardLines.push(`${fwd}  # ${LAYER_DEFS[layerType].label}`);
      }
    }
    prevOut = n.data.outputShape ? [...n.data.outputShape] : prevOut;
  }

  const trainStub = includeTrainStub
    ? `

# ---------------------------------------------------------------------------
# Optional: tiny training sketch for Fashion-MNIST (run in Colab / locally)
# CrossEntropyLoss expects raw logits — if you included Softmax, remove it
# from the model when training, or switch to NLLLoss on log-probabilities.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST-ish stats
    ])
    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ${className}().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1):  # bump this when you're ready
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            # If the model ends with Softmax, this is not ideal — prefer raw logits.
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1} loss={total / len(train_loader):.4f}")
`
    : "";

  const code = `"""
Fashion-MNIST image classifier — generated by easy deep learning
Task: classify 28×28 grayscale clothing images into ${FASHION_MNIST.numClasses} classes.
Every comment below explains *why* the line exists, not just what it does.
"""
import torch
import torch.nn as nn


class ${className}(nn.Module):
    """Visual-builder export: layers follow your canvas order (topological sort)."""

    def __init__(self):
        super().__init__()
        # Layers are registered here so PyTorch tracks their parameters for training.
${initLines.length ? initLines.join("\n") : "        pass  # empty model"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Data flows top → bottom, matching edges on the canvas.
${forwardLines.map((l) => "        " + l).join("\n")}
        return x
${trainStub}`;

  return { code, error: null };
}

export function projectToGenInput(state: ProjectState) {
  return {
    nodes: state.nodes.map((n) => ({ id: n.id, data: n.data })),
    edges: state.edges.map((e) => ({ source: e.source, target: e.target })),
  };
}
