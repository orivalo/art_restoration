"""Gated U-Net baseline (DeepFillv2-style, Yu et al. 2019).

A U-Net topology in which every convolution is a ``GatedConv2d`` from
``src.models.gated_conv``.  Mirrors the PConvUNet skeleton (same depth,
same channel ladder, same skip connections) so the only architectural
axis that differs from PConvUNet is the convolution operator.

Why this baseline matters
-------------------------

DeepFillv2 / gated convolution is the standard "modern strong baseline"
for free-form irregular-mask inpainting.  Comparing PConv against
gated conv answers the supervisor's expected question — *"is PConv
still competitive against newer mask-aware convolutions?"*

Architecture
------------

Encoder (7 layers, all use GatedConv2d)::

    Layer 1:  GatedConv(  4 →  64, k=7, s=2, p=3)  256 → 128   ← +1 ch for mask
    Layer 2:  GatedConv( 64 → 128, k=5, s=2, p=2)  128 →  64
    Layer 3:  GatedConv(128 → 256, k=5, s=2, p=2)   64 →  32
    Layer 4:  GatedConv(256 → 512, k=3, s=2, p=1)   32 →  16
    Layer 5:  GatedConv(512 → 512, k=3, s=2, p=1)   16 →   8
    Layer 6:  GatedConv(512 → 512, k=3, s=2, p=1)    8 →   4
    Layer 7:  GatedConv(512 → 512, k=3, s=2, p=1)    4 →   2

Decoder (7 layers, NearestUpsample → Concat(skip) → GatedConv)::

    Same channel-flow as PConvUNet's decoder, but every convolution is
    a GatedConv2d.  Final layer uses ``activation=Identity`` so the
    raw pre-sigmoid logits feed into ``torch.sigmoid`` for [0, 1] output.

Note: the typical DeepFillv2 generator uses BatchNorm-free encoder and
a coarse-to-fine two-stage refinement.  We deliberately keep the
single-stage U-Net topology to maximise architectural fairness with
PConvUNet (same depth, same channel widths, same training recipe).
The PConvUNet does include BatchNorm in the encoder; we keep that here
too — same recipe across both models.

Forward contract
~~~~~~~~~~~~~~~~

For trainer compatibility, ``forward(image, mask)`` returns the tuple
``(output, mask)``; the mask is **passed through unchanged** because
gated conv has no separate mask-state to propagate (the soft gate is
absorbed into the feature map).

References
----------
[1] Yu et al., "Free-Form Image Inpainting with Gated Convolution",
    ICCV 2019.  arXiv:1806.03589.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gated_conv import GatedConv2d


# ──────────────────────────────────────────────────────────────────────────────
#  Encoder block — GatedConv + BN
# ──────────────────────────────────────────────────────────────────────────────


class _GatedEncoderBlock(nn.Module):
    """Encoder stage: GatedConv2d → BatchNorm2d.

    The GatedConv2d's internal ELU activation is applied before the
    multiplicative gate, so we add BatchNorm2d **after** the gated
    output to normalise the gated feature distribution — symmetric to
    the PConvUNet encoder block which does ``conv → bn → relu``.
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
        self.gconv = GatedConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.gconv(x))


# ──────────────────────────────────────────────────────────────────────────────
#  Decoder block — Upsample → Concat(skip) → GatedConv
# ──────────────────────────────────────────────────────────────────────────────


class _GatedDecoderBlock(nn.Module):
    """Decoder stage with skip connection (no BatchNorm in decoder)."""

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
        # When use_activation is False (final layer), we still want the
        # gate to operate (sigmoid * conv_feat), but skip the ELU on the
        # feature path — pass Identity as the activation.
        feat_act: nn.Module = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
        self.gconv = GatedConv2d(
            in_channels + skip_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
            activation=feat_act,
        )

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2, mode="nearest")
        x_cat = torch.cat([x_up, skip_x], dim=1)
        return self.gconv(x_cat)


# ──────────────────────────────────────────────────────────────────────────────
#  Full Gated U-Net
# ──────────────────────────────────────────────────────────────────────────────


class GatedUNet(nn.Module):
    """Gated convolution U-Net for inpainting (DeepFillv2-style).

    Mask is concatenated to RGB as a 4th input channel inside ``forward``;
    the trainer continues to call ``model(image, mask)`` exactly like for
    PConvUNet and VanillaUNet.

    Args:
        in_channels: RGB channels of the input image (default 3).  Mask
            adds a 4th channel internally — do not include it here.
        out_channels: Output image channels (default 3 for RGB).
        verbose: If ``True``, prints parameter count at construction.
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

        first_in = in_channels + 1                                # +1 for mask channel

        # ── Encoder (7 layers) ───────────────────────────────────────────
        self.enc_1 = _GatedEncoderBlock(first_in, 64, 7, 2, 3)
        self.enc_2 = _GatedEncoderBlock(64, 128, 5, 2, 2)
        self.enc_3 = _GatedEncoderBlock(128, 256, 5, 2, 2)
        self.enc_4 = _GatedEncoderBlock(256, 512, 3, 2, 1)
        self.enc_5 = _GatedEncoderBlock(512, 512, 3, 2, 1)
        self.enc_6 = _GatedEncoderBlock(512, 512, 3, 2, 1)
        self.enc_7 = _GatedEncoderBlock(512, 512, 3, 2, 1)

        # ── Decoder (7 layers) ───────────────────────────────────────────
        self.dec_7 = _GatedDecoderBlock(512, 512, 512)            # +E6 → 4×4
        self.dec_6 = _GatedDecoderBlock(512, 512, 512)            # +E5 → 8×8
        self.dec_5 = _GatedDecoderBlock(512, 512, 512)            # +E4 → 16×16
        self.dec_4 = _GatedDecoderBlock(512, 256, 256)            # +E3 → 32×32
        self.dec_3 = _GatedDecoderBlock(256, 128, 128)            # +E2 → 64×64
        self.dec_2 = _GatedDecoderBlock(128, 64, 64)              # +E1 → 128×128
        self.dec_1 = _GatedDecoderBlock(                          # +input → 256×256
            64, first_in, out_channels,
            use_activation=False,
        )

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"GatedUNet — {total:,} total parameters ({trainable:,} trainable)")

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_image: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            input_image: ``(B, in_channels, H, W)`` RGB image with hole
                pixels pre-zeroed by the trainer.
            input_mask:  ``(B, 1, H, W)`` binary mask (1 = valid, 0 = hole).

        Returns:
            Tuple ``(output, input_mask)``:

            * ``output`` — ``(B, out_channels, H, W)`` in ``[0, 1]``.
            * ``input_mask`` — passed through unchanged for trainer
              compatibility.  Gated conv has no internal mask state.
        """
        e0 = torch.cat([input_image, input_mask], dim=1)          # (B, in+1, H, W)

        e1 = self.enc_1(e0)                                       # 128×128, 64
        e2 = self.enc_2(e1)                                       # 64×64,   128
        e3 = self.enc_3(e2)                                       # 32×32,   256
        e4 = self.enc_4(e3)                                       # 16×16,   512
        e5 = self.enc_5(e4)                                       # 8×8,     512
        e6 = self.enc_6(e5)                                       # 4×4,     512
        e7 = self.enc_7(e6)                                       # 2×2,     512

        d = self.dec_7(e7, e6)                                    # 4×4,    512
        d = self.dec_6(d, e5)                                     # 8×8,    512
        d = self.dec_5(d, e4)                                     # 16×16,  512
        d = self.dec_4(d, e3)                                     # 32×32,  256
        d = self.dec_3(d, e2)                                     # 64×64,  128
        d = self.dec_2(d, e1)                                     # 128×128, 64
        d = self.dec_1(d, e0)                                     # 256×256, out_channels

        output = torch.sigmoid(d)
        return output, input_mask


# ======================================================================
#  Smoke test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("GatedUNet — smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = GatedUNet().to(device)

    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W, device=device)
    m = (torch.rand(B, 1, H, W, device=device) > 0.4).float()

    out, ret_mask = model(x * m, m)

    # ── Shapes ─────────────────────────────────────────────────────────
    print(f"\n[Test 1] full forward pass")
    print(f"  input image : {tuple(x.shape)}")
    print(f"  input mask  : {tuple(m.shape)}")
    print(f"  output      : {tuple(out.shape)}")
    print(f"  ret mask    : {tuple(ret_mask.shape)}")
    assert out.shape == (B, C, H, W)
    assert ret_mask.shape == (B, 1, H, W)
    print("  shapes OK ✓")

    # ── Mask passed through ────────────────────────────────────────────
    print(f"\n[Test 2] mask pass-through")
    assert torch.equal(ret_mask, m)
    print("  mask passed through unchanged ✓")

    # ── Output range ───────────────────────────────────────────────────
    print(f"\n[Test 3] output range")
    print(f"  min: {out.min().item():.4f}   max: {out.max().item():.4f}")
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0
    print("  output ⊂ [0, 1] ✓")

    # ── Gradient flow ──────────────────────────────────────────────────
    print(f"\n[Test 4] gradient flow")
    loss = out.sum()
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_total = sum(1 for _ in model.parameters())
    print(f"  parameters with non-zero grad: {n_with_grad}/{n_total}")
    assert n_with_grad == n_total
    print("  all parameters receive gradients ✓")

    # ── Param count ────────────────────────────────────────────────────
    print(f"\n[Test 5] parameter breakdown")
    total = 0
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        total += n
        print(f"  {name:8s}  {n:>12,}")
    print(f"  {'TOTAL':8s}  {total:>12,}")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
