"use client";

import type { LayerType } from "@/lib/types";
import type { TemplateId } from "@/lib/templates";

const svgProps = {
  width: 18,
  height: 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true as const,
};

function IconWrap({
  color,
  children,
}: {
  color: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md"
      style={{ background: `${color}22`, color }}
    >
      {children}
    </span>
  );
}

const ICONS: Record<LayerType, React.ReactNode> = {
  Input: (
    <svg {...svgProps}>
      <path d="M4 12h10" />
      <path d="M14 8l4 4-4 4" />
      <path d="M4 7v10" />
    </svg>
  ),
  Output: (
    <svg {...svgProps}>
      <circle cx="12" cy="12" r="3" />
      <circle cx="12" cy="12" r="7" />
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2" />
    </svg>
  ),
  Linear: (
    <svg {...svgProps}>
      <rect x="4" y="4" width="6" height="6" rx="1" />
      <rect x="14" y="4" width="6" height="6" rx="1" />
      <rect x="4" y="14" width="6" height="6" rx="1" />
      <rect x="14" y="14" width="6" height="6" rx="1" />
      <path d="M10 7h4M10 17h4M7 10v4M17 10v4" />
    </svg>
  ),
  Conv2d: (
    <svg {...svgProps}>
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </svg>
  ),
  Conv1d: (
    <svg {...svgProps}>
      <path d="M3 12h4l2-5 3 10 2-5h7" />
    </svg>
  ),
  MaxPool2d: (
    <svg {...svgProps}>
      <path d="M8 4v6M12 4v10M16 4v6" />
      <path d="M6 16h12" />
      <path d="M9 16l3 4 3-4" />
    </svg>
  ),
  AvgPool2d: (
    <svg {...svgProps}>
      <path d="M5 8h14M7 12h10M9 16h6" />
      <path d="M12 4v16" />
    </svg>
  ),
  AdaptiveAvgPool2d: (
    <svg {...svgProps}>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" rx="1" />
    </svg>
  ),
  GlobalAvgPool1d: (
    <svg {...svgProps}>
      <path d="M5 5v14M19 5v14" />
      <path d="M5 12h14" />
      <path d="M9 8h6M9 16h6" />
    </svg>
  ),
  BatchNorm1d: (
    <svg {...svgProps}>
      <path d="M4 16c3-8 5-8 8 0s5 8 8 0" />
      <path d="M4 12h16" />
    </svg>
  ),
  BatchNorm2d: (
    <svg {...svgProps}>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 8v4l3 2" />
    </svg>
  ),
  LayerNorm: (
    <svg {...svgProps}>
      <path d="M6 8h12M6 12h12M6 16h12" />
      <path d="M9 6v12M15 6v12" />
    </svg>
  ),
  Flatten: (
    <svg {...svgProps}>
      <path d="M7 7h10v6H7z" />
      <path d="M5 17h14" />
      <path d="M9 13l-2 4M15 13l2 4" />
    </svg>
  ),
  Reshape: (
    <svg {...svgProps}>
      <rect x="4" y="4" width="7" height="16" rx="1" />
      <rect x="13" y="8" width="7" height="8" rx="1" />
    </svg>
  ),
  Embedding: (
    <svg {...svgProps}>
      <path d="M8 6h8v4H8zM8 14h8v4H8z" />
      <path d="M12 10v4" />
      <path d="M7 4l5 2 5-2" />
    </svg>
  ),
  PositionalEncoding: (
    <svg {...svgProps}>
      <path d="M5 7h14M5 12h10M5 17h14" />
      <path d="M17 10v4M15 12h4" />
    </svg>
  ),
  LSTM: (
    <svg {...svgProps}>
      <path d="M4 8c2 0 2 8 4 8s2-8 4-8 2 8 4 8 2-8 4-8" />
    </svg>
  ),
  GRU: (
    <svg {...svgProps}>
      <path d="M7 8h7a3 3 0 010 6H9" />
      <path d="M9 14H7a3 3 0 010-6" />
      <path d="M14 8l2-2M14 14l2 2" />
    </svg>
  ),
  MultiheadAttention: (
    <svg {...svgProps}>
      <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" />
      <path d="M18 14l.8 2.2L21 17l-2.2.8L18 20l-.8-2.2L15 17l2.2-.8L18 14z" />
    </svg>
  ),
  TransformerEncoder: (
    <svg {...svgProps}>
      <rect x="5" y="4" width="14" height="16" rx="2" />
      <path d="M9 8h6M9 12h6M9 16h4" />
    </svg>
  ),
  TransformerDecoder: (
    <svg {...svgProps}>
      <circle cx="12" cy="12" r="7" />
      <path d="M12 5v14" />
    </svg>
  ),
  ReLU: (
    <svg {...svgProps}>
      <path d="M4 16h6l6-8h4" />
    </svg>
  ),
  LeakyReLU: (
    <svg {...svgProps}>
      <path d="M4 14l5 2 5-8h6" />
    </svg>
  ),
  GELU: (
    <svg {...svgProps}>
      <path d="M4 15c4-1 5-7 8-7s4 6 8 7" />
    </svg>
  ),
  Sigmoid: (
    <svg {...svgProps}>
      <path d="M4 16c4 0 4-8 8-8s4 8 8 8" />
    </svg>
  ),
  Tanh: (
    <svg {...svgProps}>
      <path d="M4 12c3-6 5-6 8 0s5 6 8 0" />
    </svg>
  ),
  Dropout: (
    <svg {...svgProps}>
      <path d="M6 6l12 12M18 6L6 18" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  ),
  Softmax: (
    <svg {...svgProps}>
      <path d="M5 17l4-10 3 6 3-4 4 8" />
    </svg>
  ),
  Add: (
    <svg {...svgProps}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  ),
  Concat: (
    <svg {...svgProps}>
      <rect x="3" y="7" width="7" height="10" rx="1" />
      <rect x="14" y="7" width="7" height="10" rx="1" />
      <path d="M10 12h4" />
    </svg>
  ),
  LoopBlock: (
    <svg {...svgProps}>
      <path d="M7 7h7a4 4 0 010 8H9" />
      <path d="M9 15H7a4 4 0 010-8" />
      <path d="M15 7l2-2M15 15l2 2" />
    </svg>
  ),
};

const TEMPLATE_ICONS: Record<string, React.ReactNode> = {
  mlp: (
    <svg {...svgProps}>
      <circle cx="6" cy="6" r="2" />
      <circle cx="6" cy="18" r="2" />
      <circle cx="18" cy="6" r="2" />
      <circle cx="18" cy="18" r="2" />
      <circle cx="12" cy="12" r="2" />
      <path d="M8 7l3 4M8 17l3-4M16 7l-3 4M16 17l-3-4" />
    </svg>
  ),
  cnn: (
    <svg {...svgProps}>
      <rect x="4" y="5" width="16" height="3" rx="1" />
      <rect x="6" y="10" width="12" height="3" rx="1" />
      <rect x="8" y="15" width="8" height="3" rx="1" />
    </svg>
  ),
  lstm: (
    <svg {...svgProps}>
      <path d="M4 12h4l2-4 3 8 2-4h5" />
      <path d="M18 8v8" />
    </svg>
  ),
  transformer: (
    <svg {...svgProps}>
      <path d="M12 3c2 3 2 6 0 9-2 3-2 6 0 9" />
      <path d="M8 6c1.5 2 1.5 4 0 6M16 6c-1.5 2-1.5 4 0 6" />
      <circle cx="12" cy="12" r="1.5" />
    </svg>
  ),
  ae: (
    <svg {...svgProps}>
      <path d="M4 12h5l3-6 3 12 2-6h3" />
    </svg>
  ),
  text: (
    <svg {...svgProps}>
      <path d="M6 4h12v16H6z" />
      <path d="M9 8h6M9 12h6M9 16h4" />
    </svg>
  ),
  resnet: (
    <svg {...svgProps}>
      <circle cx="7" cy="12" r="2.5" />
      <circle cx="17" cy="12" r="2.5" />
      <path d="M9.5 12h5" />
      <path d="M7 9.5C7 6 17 6 17 9.5" />
    </svg>
  ),
};

export function LayerTypeIcon({
  type,
  color,
}: {
  type: LayerType;
  color: string;
}) {
  return <IconWrap color={color}>{ICONS[type]}</IconWrap>;
}

export function TemplateTypeIcon({
  icon,
  color,
}: {
  icon: string;
  color: string;
}) {
  return (
    <IconWrap color={color}>{TEMPLATE_ICONS[icon] ?? TEMPLATE_ICONS.mlp}</IconWrap>
  );
}

export type { TemplateId };
