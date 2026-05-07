"""PConv U-Net — Partial Convolution U-Net for art inpainting.

Implements the U-Net architecture from Liu et al. (ECCV 2018, arXiv:1804.07723)
using the ``PartialConv2d`` layer.  All convolutions in both encoder and
decoder are partial — the binary mask is propagated end-to-end and updated
at every layer via the logical-OR rule.

Architecture summary
--------------------

Encoder (7 layers, all use PConv + BN + ReLU)::

    Layer 1:  PConv(  3 →  64, k=7, s=2, p=3)  256 → 128
    Layer 2:  PConv( 64 → 128, k=5, s=2, p=2)  128 →  64
    Layer 3:  PConv(128 → 256, k=5, s=2, p=2)   64 →  32
    Layer 4:  PConv(256 → 512, k=3, s=2, p=1)   32 →  16
    Layer 5:  PConv(512 → 512, k=3, s=2, p=1)   16 →   8
    Layer 6:  PConv(512 → 512, k=3, s=2, p=1)    8 →   4
    Layer 7:  PConv(512 → 512, k=3, s=2, p=1)    4 →   2

Decoder (7 layers, NearestUpsample → Concat(skip) → PConv + LeakyReLU(0.2))::

    Layer 7:  upsample → cat(E6) → PConv(1024 → 512, k=3, s=1, p=1)   2 →   4
    Layer 6:  upsample → cat(E5) → PConv(1024 → 512, k=3, s=1, p=1)   4 →   8
    Layer 5:  upsample → cat(E4) → PConv(1024 → 512, k=3, s=1, p=1)   8 →  16
    Layer 4:  upsample → cat(E3) → PConv( 768 → 256, k=3, s=1, p=1)  16 →  32
    Layer 3:  upsample → cat(E2) → PConv( 384 → 128, k=3, s=1, p=1)  32 →  64
    Layer 2:  upsample → cat(E1) → PConv( 192 →  64, k=3, s=1, p=1)  64 → 128
    Layer 1:  upsample → cat(I)  → PConv(  67 →   3, k=3, s=1, p=1) 128 → 256

No BatchNorm in the decoder.  Final layer has no activation and no BN; the
output of layer 1 is passed through ``torch.sigmoid`` to clamp values into
``[0, 1]``.

Skip connections
~~~~~~~~~~~~~~~~

Features are concatenated along the channel axis as usual.  Single-channel
masks are combined via **logical OR** (``(m1 + m2).clamp(max=1)``) before
being fed to the decoder PConv — a position is valid if either the encoder
skip or the upsampled decoder feature is valid there.

VRAM estimate (256×256, batch=4, fp16)
--------------------------------------

* Model parameters:    ~26 M  →  ~52 MB (fp16)  +  ~100 MB master copy (fp32)
* Optimizer state (Adam, m+v):                  ~200 MB (fp32)
* Forward activations (encoder + decoder):     ~250 MB
* Gradients:                                   ~100 MB
* Skip-connection caches (E1…E6):               ~50 MB
* Plus VGG16 perceptual loss (frozen):         ~120 MB
* **Total ≈ 0.9–1.2 GB**, well within the T4 16 GB / P100 16 GB budget.

References
----------
[1] Liu et al., "Image Inpainting for Irregular Holes Using Partial
    Convolutions", ECCV 2018.  arXiv:1804.07723.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.partial_conv import PartialConv2d


# ──────────────────────────────────────────────────────────────────────────────
#  Encoder block — PConv + BN + ReLU
# ──────────────────────────────────────────────────────────────────────────────


class _PConvEncoderBlock(nn.Module):
    """Single encoder stage: PartialConv2d → BatchNorm2d → ReLU.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        kernel_size:  Convolution kernel size.
        stride:       Convolution stride.
        padding:      Zero padding on both sides.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.pconv = PartialConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        x = self.act(x)
        return x, mask


# ──────────────────────────────────────────────────────────────────────────────
#  Decoder block — Upsample → Concat(skip) → PConv [+ LeakyReLU(0.2)]
# ──────────────────────────────────────────────────────────────────────────────


class _PConvDecoderBlock(nn.Module):
    """Single decoder stage with skip connection.

    The previous decoder feature is upsampled (nearest neighbour, ×2),
    concatenated with the corresponding encoder skip along the channel
    dimension, and processed by a PartialConv2d.  Masks are combined via
    logical OR before the partial convolution.

    Args:
        in_channels:  Channels in the upsampled lower-resolution feature.
        skip_channels: Channels in the encoder skip feature.
        out_channels: Output channels of the PartialConv2d.
        kernel_size:  Convolution kernel size (default 3).
        stride:       Convolution stride (default 1).
        padding:      Zero padding (default 1).
        use_activation: If ``False``, no LeakyReLU is applied (final layer).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_activation: bool = True,
    ) -> None:
        super().__init__()
        self.pconv = PartialConv2d(
            in_channels + skip_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        self.act: nn.Module = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        skip_x: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ── Upsample lower-resolution feature and mask ───────────────────
        x_up = F.interpolate(x, scale_factor=2, mode="nearest")
        mask_up = F.interpolate(mask, scale_factor=2, mode="nearest")

        # ── Concatenate features along the channel axis ─────────────────
        x_cat = torch.cat([x_up, skip_x], dim=1)

        # ── Combine masks via logical OR (clamp to keep binary) ─────────
        mask_or = (mask_up + skip_mask).clamp(max=1.0)

        # ── Partial convolution + (optional) activation ─────────────────
        out, out_mask = self.pconv(x_cat, mask_or)
        out = self.act(out)
        return out, out_mask


# ──────────────────────────────────────────────────────────────────────────────
#  Full PConv U-Net
# ──────────────────────────────────────────────────────────────────────────────


class PConvUNet(nn.Module):
    """Partial-Convolution U-Net for image inpainting (Liu et al., 2018).

    Takes a 3-channel RGB input ``[B, 3, H, W]`` together with a single-
    channel binary mask ``[B, 1, H, W]`` (1 = valid, 0 = hole) and predicts
    the inpainted image clamped to ``[0, 1]`` via ``torch.sigmoid``.

    Args:
        in_channels: Number of input image channels (default 3 for RGB).
        out_channels: Number of output image channels (default 3 for RGB).
        verbose: If ``True``, prints the parameter count at construction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ── Encoder (7 layers) ───────────────────────────────────────────
        # (in, out, k, s, p)
        self.enc_1 = _PConvEncoderBlock(in_channels, 64, 7, 2, 3)
        self.enc_2 = _PConvEncoderBlock(64, 128, 5, 2, 2)
        self.enc_3 = _PConvEncoderBlock(128, 256, 5, 2, 2)
        self.enc_4 = _PConvEncoderBlock(256, 512, 3, 2, 1)
        self.enc_5 = _PConvEncoderBlock(512, 512, 3, 2, 1)
        self.enc_6 = _PConvEncoderBlock(512, 512, 3, 2, 1)
        self.enc_7 = _PConvEncoderBlock(512, 512, 3, 2, 1)

        # ── Decoder (7 layers) ───────────────────────────────────────────
        # (in_from_below, skip_channels, out_channels)
        self.dec_7 = _PConvDecoderBlock(512, 512, 512)               # +E6  →  4×4
        self.dec_6 = _PConvDecoderBlock(512, 512, 512)               # +E5  →  8×8
        self.dec_5 = _PConvDecoderBlock(512, 512, 512)               # +E4  → 16×16
        self.dec_4 = _PConvDecoderBlock(512, 256, 256)               # +E3  → 32×32
        self.dec_3 = _PConvDecoderBlock(256, 128, 128)               # +E2  → 64×64
        self.dec_2 = _PConvDecoderBlock(128, 64, 64)                 # +E1  → 128×128
        self.dec_1 = _PConvDecoderBlock(                              # +input → 256×256
            64, in_channels, out_channels,
            use_activation=False,                                    # final: no activation
        )

        # ── Report parameter count ──────────────────────────────────────
        if verbose:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"PConvUNet — {total:,} total parameters ({trainable:,} trainable)")

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_image: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a full forward pass through the PConv U-Net.

        Args:
            input_image: Tensor ``(B, in_channels, H, W)`` — RGB image with
                hole pixels already zeroed (``image * mask``).
            input_mask:  Tensor ``(B, 1, H, W)`` — binary mask with
                ``1 = valid``, ``0 = hole``.

        Returns:
            Tuple ``(output_image, output_mask)`` where:

            - **output_image** is the inpainted image of shape
              ``(B, out_channels, H, W)`` with values in ``[0, 1]``.
            - **output_mask** is the mask emitted by the final PConv layer
              of shape ``(B, 1, H, W)`` — typically all-ones except for
              positions that were entirely unreachable from any valid
              input pixel.
        """
        # Cache encoder outputs (features + masks) for skip connections
        e0_x, e0_m = input_image, input_mask                  # 256×256

        e1_x, e1_m = self.enc_1(e0_x, e0_m)                   # 128×128, 64
        e2_x, e2_m = self.enc_2(e1_x, e1_m)                   # 64×64,   128
        e3_x, e3_m = self.enc_3(e2_x, e2_m)                   # 32×32,   256
        e4_x, e4_m = self.enc_4(e3_x, e3_m)                   # 16×16,   512
        e5_x, e5_m = self.enc_5(e4_x, e4_m)                   # 8×8,     512
        e6_x, e6_m = self.enc_6(e5_x, e5_m)                   # 4×4,     512
        e7_x, e7_m = self.enc_7(e6_x, e6_m)                   # 2×2,     512  (bottleneck)

        # Decoder — each block upsamples + skips with the matching encoder layer
        d_x, d_m = self.dec_7(e7_x, e7_m, e6_x, e6_m)         # 4×4,   512
        d_x, d_m = self.dec_6(d_x, d_m, e5_x, e5_m)           # 8×8,   512
        d_x, d_m = self.dec_5(d_x, d_m, e4_x, e4_m)           # 16×16, 512
        d_x, d_m = self.dec_4(d_x, d_m, e3_x, e3_m)           # 32×32, 256
        d_x, d_m = self.dec_3(d_x, d_m, e2_x, e2_m)           # 64×64, 128
        d_x, d_m = self.dec_2(d_x, d_m, e1_x, e1_m)           # 128×128, 64
        d_x, d_m = self.dec_1(d_x, d_m, e0_x, e0_m)           # 256×256, 3 (raw)

        # Sigmoid clamps the raw output to [0, 1]
        output_image = torch.sigmoid(d_x)
        return output_image, d_m


# ======================================================================
#  Smoke test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("PConvUNet — smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Build the network ────────────────────────────────────────────────
    model = PConvUNet().to(device)

    # ── Forward pass on a random batch ──────────────────────────────────
    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W, device=device)
    m = (torch.rand(B, 1, H, W, device=device) > 0.4).float()  # ~60 % valid

    out, out_mask = model(x * m, m)

    # ── Shape assertions ─────────────────────────────────────────────────
    print(f"\n[Test 1] full forward pass")
    print(f"  input image : {tuple(x.shape)}")
    print(f"  input mask  : {tuple(m.shape)}")
    print(f"  output      : {tuple(out.shape)}")
    print(f"  output mask : {tuple(out_mask.shape)}")
    assert out.shape == (B, C, H, W), f"output shape wrong: {out.shape}"
    assert out_mask.shape == (B, 1, H, W), f"mask shape wrong: {out_mask.shape}"
    print("  shapes OK ✓")

    # ── Output range (sigmoid → [0, 1]) ─────────────────────────────────
    print(f"\n[Test 2] output range")
    print(f"  min: {out.min().item():.4f}   max: {out.max().item():.4f}")
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0, \
        "sigmoid output must be in [0, 1]"
    print("  output ⊂ [0, 1] ✓")

    # ── Updated mask is binary ──────────────────────────────────────────
    print(f"\n[Test 3] final mask is binary")
    vals = out_mask.unique().tolist()
    print(f"  unique values: {vals}")
    assert set(vals).issubset({0.0, 1.0}), f"non-binary final mask: {vals}"
    print("  binary ✓")

    # ── Gradient flow through the entire network ───────────────────────
    print(f"\n[Test 4] gradient flow")
    loss = out.sum()
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_total = sum(1 for _ in model.parameters())
    print(f"  parameters with non-zero grad: {n_with_grad}/{n_total}")
    assert n_with_grad == n_total, "some parameters have no gradient"
    print("  all parameters receive gradients ✓")

    # ── Per-layer parameter breakdown ───────────────────────────────────
    print("\n[Test 5] parameter breakdown")
    total = 0
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        total += n
        print(f"  {name:8s}  {n:>12,}")
    print(f"  {'TOTAL':8s}  {total:>12,}")

    # ── Mask shrinkage — check that holes vanish through the bottleneck ─
    print(f"\n[Test 6] mask propagation through encoder")
    test_in = torch.randn(1, 3, 256, 256, device=device)
    test_m = (torch.rand(1, 1, 256, 256, device=device) > 0.4).float()
    print(f"  input        :  hole_frac = {(1.0 - test_m.mean()).item():.2%}")

    # Manually walk the encoder
    em = test_m
    ex = test_in
    for i, blk in enumerate([model.enc_1, model.enc_2, model.enc_3,
                             model.enc_4, model.enc_5, model.enc_6, model.enc_7], 1):
        ex, em = blk(ex, em)
        hole_frac = (1.0 - em.mean()).item()
        print(f"  after enc_{i} :  hole_frac = {hole_frac:.2%}   spatial = {tuple(em.shape[-2:])}")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
