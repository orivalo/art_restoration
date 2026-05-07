r"""Generate the PConv U-Net architecture diagram for the diploma thesis.

Renders a publication-quality figure showing the 7-layer encoder /
7-layer decoder PConv U-Net architecture used in the project, with skip
connections drawn as dashed horizontal arrows between corresponding
encoder and decoder rows.

Outputs (saved next to this script):
    pconv_unet_architecture.png   — 300 DPI raster (slides)
    pconv_unet_architecture.pdf   — vector (LaTeX \includegraphics)
    pconv_unet_architecture.svg   — vector (web / editing)

Usage:
    python thesis/figures/generate_pconv_unet_diagram.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ──────────────────────────────────────────────────────────────────────────────
#  Layout constants
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).resolve().parent

FIG_W, FIG_H = 14.0, 14.5      # inches
ENC_X = 0.6                    # left edge of encoder column
DEC_X = 7.6                    # left edge of decoder column
COL_W = 5.8                    # block width
BLK_H = 1.05                   # block height
ROW_GAP = 1.32                 # vertical centre-to-centre row pitch
TOP_Y = 13.4                   # y of the top row's top edge

# ── Colour palette (subdued, paper-ready) ────────────────────────────────────
ENC_FACE, ENC_EDGE = "#dbeafe", "#1e40af"   # blue
DEC_FACE, DEC_EDGE = "#dcfce7", "#15803d"   # green
INP_FACE, INP_EDGE = "#fef3c7", "#a16207"   # amber
OUT_FACE, OUT_EDGE = "#fed7aa", "#9a3412"   # orange
SKIP_COLOR = "#94a3b8"                       # gray (dashed)
FLOW_COLOR = "#475569"                       # slate
BTLNK_COLOR = "#b91c1c"                      # red (bottleneck)


# ──────────────────────────────────────────────────────────────────────────────
#  Architecture specification
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Block:
    """Single layer block: title, operation, output shape."""

    name: str
    op: str
    shape: str


INPUT_BLOCK = Block(
    name="Input",
    op="RGB image I  +  binary mask M",
    shape="I : 3 × 256²        M : 1 × 256²",
)

# Encoder ordered top → bottom (E1 first, E7 = bottleneck last)
ENCODER: list[Block] = [
    Block("Encoder 1", "PConv 7×7, s=2   +   BN + ReLU", "→  64  × 128²"),
    Block("Encoder 2", "PConv 5×5, s=2   +   BN + ReLU", "→ 128 ×  64²"),
    Block("Encoder 3", "PConv 5×5, s=2   +   BN + ReLU", "→ 256 ×  32²"),
    Block("Encoder 4", "PConv 3×3, s=2   +   BN + ReLU", "→ 512 ×  16²"),
    Block("Encoder 5", "PConv 3×3, s=2   +   BN + ReLU", "→ 512 ×   8²"),
    Block("Encoder 6", "PConv 3×3, s=2   +   BN + ReLU", "→ 512 ×   4²"),
    Block("Encoder 7", "PConv 3×3, s=2   +   BN + ReLU", "→ 512 ×   2²    (bottleneck)"),
]

# Decoder ordered top → bottom (D1 first, D7 last).  In forward order D7 runs
# first and D1 runs last, but for a U-shape diagram we display D1 at the top
# (closest to the OUTPUT) and D7 at the bottom (closest to the bottleneck).
DECODER: list[Block] = [
    Block("Decoder 1", "Upsample×2   cat(Input)   PConv 3×3, s=1", "→  3  × 256²    (no activation)"),
    Block("Decoder 2", "Upsample×2   cat(E1)      PConv  +  LeakyReLU", "→  64  × 128²"),
    Block("Decoder 3", "Upsample×2   cat(E2)      PConv  +  LeakyReLU", "→ 128 ×  64²"),
    Block("Decoder 4", "Upsample×2   cat(E3)      PConv  +  LeakyReLU", "→ 256 ×  32²"),
    Block("Decoder 5", "Upsample×2   cat(E4)      PConv  +  LeakyReLU", "→ 512 ×  16²"),
    Block("Decoder 6", "Upsample×2   cat(E5)      PConv  +  LeakyReLU", "→ 512 ×   8²"),
    Block("Decoder 7", "Upsample×2   cat(E6)      PConv  +  LeakyReLU", "→ 512 ×   4²"),
]

OUTPUT_BLOCK = Block(
    name="Output",
    op="σ(·)   —   sigmoid clamp",
    shape="Restored  Î : 3 × 256²    ∈  [0, 1]",
)


# ──────────────────────────────────────────────────────────────────────────────
#  Drawing primitives
# ──────────────────────────────────────────────────────────────────────────────


def draw_block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    face: str,
    edge: str,
    blk: Block,
    fontsize: float = 9.5,
) -> None:
    """Draw a rounded rectangular block with three lines of text."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)

    cx = x + w / 2
    ax.text(cx, y + h * 0.78, blk.name,
            ha="center", va="center",
            fontsize=fontsize + 1.5, fontweight="bold", color=edge)
    ax.text(cx, y + h * 0.46, blk.op,
            ha="center", va="center",
            fontsize=fontsize - 0.7, color="#1f2937", family="monospace")
    ax.text(cx, y + h * 0.16, blk.shape,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="#0f172a", style="italic", family="monospace")


def arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    *,
    dashed: bool = False,
    width: float = 1.4,
    head: float = 14.0,
    connection: str = "arc3,rad=0",
) -> None:
    """Draw a directed arrow from (x0, y0) to (x1, y1)."""
    ls = (0, (4, 2)) if dashed else "-"
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=width,
        color=color,
        linestyle=ls,
        connectionstyle=connection,
    ))


# ──────────────────────────────────────────────────────────────────────────────
#  Diagram assembly
# ──────────────────────────────────────────────────────────────────────────────


def build_diagram() -> plt.Figure:
    """Construct the matplotlib figure containing the architecture diagram.

    Layout (9 rows, top → bottom):

        row 0:  -----------          OUTPUT
        row 1:  INPUT       --skip-> Decoder 1
        row 2:  Encoder 1   --skip-> Decoder 2
        row 3:  Encoder 2   --skip-> Decoder 3
        row 4:  Encoder 3   --skip-> Decoder 4
        row 5:  Encoder 4   --skip-> Decoder 5
        row 6:  Encoder 5   --skip-> Decoder 6
        row 7:  Encoder 6   --skip-> Decoder 7
        row 8:  Encoder 7   ---bottleneck--> Decoder 7  (diagonal arrow)
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────────
    ax.text(FIG_W / 2, FIG_H - 0.30,
            "PConv U-Net Architecture",
            ha="center", va="top",
            fontsize=16, fontweight="bold", color="#0f172a")
    ax.text(FIG_W / 2, FIG_H - 0.78,
            "Partial Convolution U-Net for art inpainting (Liu et al., ECCV 2018)   —   "
            "≈ 25.8 M parameters",
            ha="center", va="top",
            fontsize=10.5, style="italic", color="#475569")

    # ── Row y-positions (TOP edge of each row) ──────────────────────────
    # row_top[i] is the y-coordinate of the TOP edge of the block at row i.
    row_top = [TOP_Y - i * ROW_GAP for i in range(9)]
    row_bot = [t - BLK_H for t in row_top]                 # BOTTOM edge

    enc_cx = ENC_X + COL_W / 2                              # encoder column centre
    dec_cx = DEC_X + COL_W / 2                              # decoder column centre

    # ── Draw blocks ──────────────────────────────────────────────────────
    # Row 0 — OUTPUT (decoder side only)
    draw_block(ax, DEC_X, row_bot[0], COL_W, BLK_H, OUT_FACE, OUT_EDGE, OUTPUT_BLOCK)

    # Row 1 — INPUT (encoder), Decoder 1 (decoder)
    draw_block(ax, ENC_X, row_bot[1], COL_W, BLK_H, INP_FACE, INP_EDGE, INPUT_BLOCK)
    draw_block(ax, DEC_X, row_bot[1], COL_W, BLK_H, DEC_FACE, DEC_EDGE, DECODER[0])

    # Rows 2–7 — Encoder 1..6 paired with Decoder 2..7
    for i in range(6):
        row = i + 2
        draw_block(ax, ENC_X, row_bot[row], COL_W, BLK_H, ENC_FACE, ENC_EDGE, ENCODER[i])
        draw_block(ax, DEC_X, row_bot[row], COL_W, BLK_H, DEC_FACE, DEC_EDGE, DECODER[i + 1])

    # Row 8 — Encoder 7 (bottleneck), no decoder-side block
    draw_block(ax, ENC_X, row_bot[8], COL_W, BLK_H, ENC_FACE, ENC_EDGE, ENCODER[6])

    # ── Forward flow arrows on the encoder side (top → bottom) ──────────
    for i in range(8):
        arrow(ax,
              enc_cx, row_bot[i],          # tail at bottom of upper block
              enc_cx, row_top[i + 1],      # head at top of next block
              FLOW_COLOR)

    # ── Forward flow arrows on the decoder side (bottom → top) ──────────
    for i in range(7, 0, -1):              # rows 7 → 1
        arrow(ax,
              dec_cx, row_top[i],          # tail at top of lower block
              dec_cx, row_bot[i - 1],      # head at bottom of upper block
              FLOW_COLOR)

    # ── Skip connections (dashed, encoder → decoder) ────────────────────
    skip_pairs = [
        (1, 1),  # INPUT  → Decoder 1
        (2, 2),  # E1     → Decoder 2
        (3, 3),  # E2     → Decoder 3
        (4, 4),  # E3     → Decoder 4
        (5, 5),  # E4     → Decoder 5
        (6, 6),  # E5     → Decoder 6
        (7, 7),  # E6     → Decoder 7
    ]
    for enc_row, dec_row in skip_pairs:
        y = row_bot[enc_row] + BLK_H / 2  # vertical centre of the row
        arrow(ax,
              ENC_X + COL_W, y,
              DEC_X, y,
              SKIP_COLOR, dashed=True, width=1.1, head=11.0)

    # ── Bottleneck arrow: E7 (row 8, encoder side) → D7 (row 7, decoder) ─
    arrow(ax,
          ENC_X + COL_W, row_bot[8] + BLK_H / 2,
          DEC_X, row_bot[7] + BLK_H / 2,
          BTLNK_COLOR, width=1.6, head=15.0,
          connection="arc3,rad=-0.18")

    # Bottleneck label
    bx = (ENC_X + COL_W + DEC_X) / 2
    by = (row_bot[8] + row_bot[7] + BLK_H) / 2
    ax.text(bx, by + 0.18, "bottleneck flow",
            ha="center", va="center",
            fontsize=9, color=BTLNK_COLOR, style="italic")

    # ── Legend (bottom of figure) ───────────────────────────────────────
    legend_y = 0.55
    legend_x_start = 0.6
    spacing = 3.2

    def legend_swatch(x: float, face: str, edge: str, label: str) -> None:
        ax.add_patch(FancyBboxPatch(
            (x, legend_y - 0.18), 0.45, 0.36,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0, edgecolor=edge, facecolor=face,
        ))
        ax.text(x + 0.55, legend_y, label,
                ha="left", va="center",
                fontsize=9.5, color="#1f2937")

    legend_swatch(legend_x_start, ENC_FACE, ENC_EDGE, "Encoder block (PConv + BN + ReLU)")
    legend_swatch(legend_x_start + spacing + 1.4, DEC_FACE, DEC_EDGE,
                  "Decoder block (Upsample + cat + PConv + LeakyReLU)")
    legend_swatch(legend_x_start + 2 * (spacing + 1.4) + 2.2, INP_FACE, INP_EDGE, "Input  /  Output")

    # Skip-connection legend line
    arrow(ax,
          legend_x_start + 0.05, 0.15,
          legend_x_start + 0.95, 0.15,
          SKIP_COLOR, dashed=True, width=1.1, head=11.0)
    ax.text(legend_x_start + 1.05, 0.15, "skip connection (dashed)",
            ha="left", va="center", fontsize=9.5, color="#1f2937")

    arrow(ax,
          legend_x_start + 5.4, 0.15,
          legend_x_start + 6.3, 0.15,
          BTLNK_COLOR, width=1.6, head=14.0)
    ax.text(legend_x_start + 6.4, 0.15, "bottleneck forward flow",
            ha="left", va="center", fontsize=9.5, color="#1f2937")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Render the figure and save to PNG, PDF, and SVG."""
    fig = build_diagram()
    fig.tight_layout(pad=0.4)

    paths = {
        "png": OUT_DIR / "pconv_unet_architecture.png",
        "pdf": OUT_DIR / "pconv_unet_architecture.pdf",
        "svg": OUT_DIR / "pconv_unet_architecture.svg",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for fmt, p in paths.items():
        size_kb = p.stat().st_size / 1024
        print(f"  {fmt.upper():<4} → {p.relative_to(OUT_DIR.parent.parent)}   ({size_kb:.1f} KB)")
    print("\nArchitecture diagram generated.")


if __name__ == "__main__":
    main()
