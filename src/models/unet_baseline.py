"""Vanilla U-Net baseline for inpainting (no partial convolution).

Drop-in topological twin of ``src.models.pconv_unet.PConvUNet`` that uses
ordinary ``nn.Conv2d`` everywhere instead of ``PartialConv2d``.  The
binary mask is supplied to the network by **concatenating it as a fourth
input channel** to the RGB image at layer 1; subsequent layers see no
mask information directly.

Why this baseline matters
-------------------------

This network differs from PConvUNet in **exactly one** axis:

* ``PartialConv2d`` (mask-aware: weights are renormalized over the valid
  area, output mask is propagated and updated) is replaced by
* ``nn.Conv2d`` (mask-agnostic: hole pixels propagate as zeros and are
  treated as if they were valid inputs by every subsequent layer).

Any quality gap measured between PConvUNet and VanillaUNet — under
identical data, identical loss, identical hyperparameters — is therefore
attributable to the partial-convolution mechanism itself, not to depth,
channel width, optimizer, or training schedule.  This is the standard
ablation in the original Liu et al. paper (ECCV 2018, §4) and every
inpainting paper that follows it.

Architecture summary (matches PConvUNet — only the conv operator changes)
-------------------------------------------------------------------------

Encoder (7 layers, all use Conv2d + BN + ReLU)::

    Layer 1:  Conv2d(  4 →  64, k=7, s=2, p=3)  256 → 128   ← +1 ch for mask
    Layer 2:  Conv2d( 64 → 128, k=5, s=2, p=2)  128 →  64
    Layer 3:  Conv2d(128 → 256, k=5, s=2, p=2)   64 →  32
    Layer 4:  Conv2d(256 → 512, k=3, s=2, p=1)   32 →  16
    Layer 5:  Conv2d(512 → 512, k=3, s=2, p=1)   16 →   8
    Layer 6:  Conv2d(512 → 512, k=3, s=2, p=1)    8 →   4
    Layer 7:  Conv2d(512 → 512, k=3, s=2, p=1)    4 →   2

Decoder (7 layers, NearestUpsample → Concat(skip) → Conv2d + LeakyReLU)::

    Layer 7:  upsample → cat(E6) → Conv(1024 → 512)   2 →   4
    Layer 6:  upsample → cat(E5) → Conv(1024 → 512)   4 →   8
    Layer 5:  upsample → cat(E4) → Conv(1024 → 512)   8 →  16
    Layer 4:  upsample → cat(E3) → Conv( 768 → 256)  16 →  32
    Layer 3:  upsample → cat(E2) → Conv( 384 → 128)  32 →  64
    Layer 2:  upsample → cat(E1) → Conv( 192 →  64)  64 → 128
    Layer 1:  upsample → cat(I+M) → Conv( 68 →   3) 128 → 256   ← +M skip

Final layer applies ``torch.sigmoid`` to clamp values to ``[0, 1]``.

Forward contract
~~~~~~~~~~~~~~~~

For trainer compatibility, ``forward(image, mask)`` returns the tuple
``(output, mask)`` — the mask is **passed through unchanged**, since
this network has no internal mask state to propagate.  The trainer's
downstream code ignores the second element for non-PConv models.

References
----------
[1] Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015.
[2] Liu et al., "Image Inpainting for Irregular Holes Using Partial
    Convolutions", ECCV 2018, §4 (vanilla U-Net ablation baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  Encoder block — Conv + BN + ReLU
# ──────────────────────────────────────────────────────────────────────────────


class _ConvEncoderBlock(nn.Module):
    """Single encoder stage: Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ──────────────────────────────────────────────────────────────────────────────
#  Decoder block — Upsample → Concat(skip) → Conv [+ LeakyReLU(0.2)]
# ──────────────────────────────────────────────────────────────────────────────


class _ConvDecoderBlock(nn.Module):
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
        self.conv = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        self.act: nn.Module = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2, mode="nearest")
        x_cat = torch.cat([x_up, skip_x], dim=1)
        return self.act(self.conv(x_cat))


# ──────────────────────────────────────────────────────────────────────────────
#  Full Vanilla U-Net
# ──────────────────────────────────────────────────────────────────────────────


class VanillaUNet(nn.Module):
    """Vanilla U-Net baseline (no partial convolution).

    Mask is concatenated to the RGB image as a 4th input channel inside
    the forward pass; the trainer continues to call ``model(image, mask)``
    just like for PConvUNet.

    Args:
        in_channels: RGB channels of the input image (default 3).  The
            mask adds a 4th channel internally — do not include it here.
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

        # ── Encoder.  Layer 1 receives RGB+mask = in_channels + 1 ─────────
        first_in = in_channels + 1                                # +1 for mask
        self.enc_1 = _ConvEncoderBlock(first_in, 64, 7, 2, 3)
        self.enc_2 = _ConvEncoderBlock(64, 128, 5, 2, 2)
        self.enc_3 = _ConvEncoderBlock(128, 256, 5, 2, 2)
        self.enc_4 = _ConvEncoderBlock(256, 512, 3, 2, 1)
        self.enc_5 = _ConvEncoderBlock(512, 512, 3, 2, 1)
        self.enc_6 = _ConvEncoderBlock(512, 512, 3, 2, 1)
        self.enc_7 = _ConvEncoderBlock(512, 512, 3, 2, 1)

        # ── Decoder.  Last layer skips with input (RGB + mask) = first_in ─
        self.dec_7 = _ConvDecoderBlock(512, 512, 512)             # +E6 → 4×4
        self.dec_6 = _ConvDecoderBlock(512, 512, 512)             # +E5 → 8×8
        self.dec_5 = _ConvDecoderBlock(512, 512, 512)             # +E4 → 16×16
        self.dec_4 = _ConvDecoderBlock(512, 256, 256)             # +E3 → 32×32
        self.dec_3 = _ConvDecoderBlock(256, 128, 128)             # +E2 → 64×64
        self.dec_2 = _ConvDecoderBlock(128, 64, 64)               # +E1 → 128×128
        self.dec_1 = _ConvDecoderBlock(                           # +input → 256×256
            64, first_in, out_channels,
            use_activation=False,
        )

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"VanillaUNet — {total:,} total parameters ({trainable:,} trainable)")

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
            input_image: ``(B, in_channels, H, W)`` RGB image (hole pixels
                pre-zeroed by the trainer).
            input_mask:  ``(B, 1, H, W)`` binary mask (1 = valid, 0 = hole).

        Returns:
            Tuple ``(output, input_mask)``:

            * ``output`` — ``(B, out_channels, H, W)`` in ``[0, 1]``.
            * ``input_mask`` — passed through unchanged for trainer
              compatibility; the network does not maintain its own mask
              propagation state.
        """
        # Concatenate RGB + mask as the input to the encoder
        e0 = torch.cat([input_image, input_mask], dim=1)          # (B, in+1, H, W)

        # Encoder
        e1 = self.enc_1(e0)                                       # 128×128, 64
        e2 = self.enc_2(e1)                                       # 64×64,   128
        e3 = self.enc_3(e2)                                       # 32×32,   256
        e4 = self.enc_4(e3)                                       # 16×16,   512
        e5 = self.enc_5(e4)                                       # 8×8,     512
        e6 = self.enc_6(e5)                                       # 4×4,     512
        e7 = self.enc_7(e6)                                       # 2×2,     512  (bottleneck)

        # Decoder
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
    print("VanillaUNet — smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = VanillaUNet().to(device)

    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W, device=device)
    m = (torch.rand(B, 1, H, W, device=device) > 0.4).float()

    out, ret_mask = model(x * m, m)

    # ── Shape assertions ───────────────────────────────────────────────────
    print(f"\n[Test 1] full forward pass")
    print(f"  input image : {tuple(x.shape)}")
    print(f"  input mask  : {tuple(m.shape)}")
    print(f"  output      : {tuple(out.shape)}")
    print(f"  ret mask    : {tuple(ret_mask.shape)}")
    assert out.shape == (B, C, H, W), f"output shape wrong: {out.shape}"
    assert ret_mask.shape == (B, 1, H, W), f"mask shape wrong: {ret_mask.shape}"
    print("  shapes OK ✓")

    # ── Returned mask is the input mask passed through ────────────────────
    print(f"\n[Test 2] mask pass-through")
    assert torch.equal(ret_mask, m), "VanillaUNet must pass mask through unchanged"
    print("  mask passed through unchanged ✓")

    # ── Output range (sigmoid) ────────────────────────────────────────────
    print(f"\n[Test 3] output range")
    print(f"  min: {out.min().item():.4f}   max: {out.max().item():.4f}")
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0
    print("  output ⊂ [0, 1] ✓")

    # ── Gradient flow ─────────────────────────────────────────────────────
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

    # ── Param count ───────────────────────────────────────────────────────
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
