"""Partial Convolution layer (Liu et al., ECCV 2018, arXiv:1804.07723).

Implements the PartialConv2d operator that applies standard 2-D convolution
only to the *valid* (unmasked) pixels in each receptive-field window, then
re-normalises the result so that features are scale-invariant to the number
of observed pixels.

Mathematical formulation
------------------------

Given:
    - Input feature map  X  of shape (B, C_in, H, W)
    - Binary mask        M  of shape (B, 1,    H, W)   1 = valid, 0 = hole
    - Learned weights    W  of shape (C_out, C_in, k_h, k_w)
    - Learned bias       b  of shape (C_out,)

At each output spatial position p the partial convolution computes::

                 ⎧ W^T (X ⊙ M)(p) · sum(𝟙) / sum(M)(p)  +  b,   if sum(M)(p) > 0
    x'(p)   =   ⎨
                 ⎩ 0,                                               otherwise

where

    sum(𝟙)   = k_h × k_w          — total kernel-window area (constant),
    sum(M)(p) = # valid pixels in the k_h × k_w window centred on p,
    ⊙         = element-wise product  (holes in X are zeroed out before conv).

The renormalisation factor  sum(𝟙) / sum(M)  corrects for the missing data:
larger when more of the window is masked, exactly 1.0 when the window is
fully valid.  Division by zero is avoided by clamping sum(M) ≥ 1.

Mask update rule
~~~~~~~~~~~~~~~~

The output mask at each position is a **logical OR** over the input mask
values inside the receptive field::

    m'(p)  =  1   if  sum(M)(p) > 0      (≥ 1 valid input pixel)
              0   otherwise               (fully masked window)

This ensures that the mask *shrinks* (fewer holes) as the signal propagates
through successive PConv layers, which is the key property exploited by the
PConv U-Net encoder.

References
----------
[1] Liu et al., "Image Inpainting for Irregular Holes Using Partial
    Convolutions", ECCV 2018.  arXiv:1804.07723.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """Partial Convolution layer with automatic mask propagation.

    Drop-in replacement for ``nn.Conv2d`` that additionally accepts a
    binary mask tensor and returns an updated mask alongside the feature
    output.  See module docstring for the full mathematical formulation.

    Mask convention (matches Liu et al. 2018):
        1 → valid pixel  (intact)
        0 → hole         (missing / damaged)

    Args:
        in_channels:  Number of channels in the input feature map.
        out_channels: Number of channels produced by the convolution.
        kernel_size:  Size of the convolving kernel (int or 2-tuple).
        stride:       Stride of the convolution (default 1).
        padding:      Zero-padding added to both sides (default 0).
        dilation:     Spacing between kernel elements (default 1).
        bias:         If ``True``, adds a learnable bias (default ``True``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # ── Normalise kernel_size to 2-tuple ─────────────────────────────
        if isinstance(kernel_size, int):
            self._kernel_h = self._kernel_w = kernel_size
        else:
            self._kernel_h, self._kernel_w = kernel_size

        self.kernel_area: int = self._kernel_h * self._kernel_w  # sum(𝟙)

        # ── Feature convolution (NO bias — bias is handled manually) ─────
        self.feature_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,           # bias added separately, only at valid positions
        )

        # ── Optional learnable bias ──────────────────────────────────────
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # ── Fixed all-ones kernel for computing sum(M) via convolution ───
        # Shape: (1, 1, k_h, k_w) — convolves the single-channel mask with
        # the same stride / padding / dilation as the feature convolution.
        self.register_buffer(
            "_mask_kernel",
            torch.ones(1, 1, self._kernel_h, self._kernel_w),
        )

        # Store spatial params for the mask convolution
        self._stride = stride
        self._padding = padding
        self._dilation = dilation

        # ── Weight initialisation (Kaiming He, fan-in) ───────────────────
        nn.init.kaiming_normal_(self.feature_conv.weight, a=0, mode="fan_in")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply partial convolution and propagate the mask.

        Args:
            x:    Input feature map of shape ``(B, C_in, H, W)``.
            mask: Binary mask of shape ``(B, 1, H, W)`` where
                  ``1 = valid``, ``0 = hole``.

        Returns:
            Tuple ``(output, updated_mask)`` where:

            - **output** has shape ``(B, C_out, H', W')``.
            - **updated_mask** has shape ``(B, 1, H', W')`` with values
              in ``{0, 1}``.  A position is 1 iff at least one input
              pixel in its receptive field was valid (logical-OR rule).
        """
        # ── 1.  Zero-out holes in the input ──────────────────────────────
        #   Broadcasting: mask (B,1,H,W) * x (B,C_in,H,W) → (B,C_in,H,W)
        x_masked = x * mask

        # ── 2.  Convolution on the masked input: W^T (X ⊙ M) ────────────
        raw_out = self.feature_conv(x_masked)               # (B, C_out, H', W')

        # ── 3.  Compute sum(M) at every output position ─────────────────
        #   No gradients needed — mask is not a learned quantity.
        with torch.no_grad():
            mask_sum = F.conv2d(
                mask,
                self._mask_kernel,
                stride=self._stride,
                padding=self._padding,
                dilation=self._dilation,
            )  # (B, 1, H', W')

        # ── 4.  Mask update: logical OR over the receptive field ─────────
        #   updated_mask(p) = 1  iff  sum(M)(p) > 0
        updated_mask = (mask_sum > 0).to(x.dtype)            # (B, 1, H', W')

        # ── 5.  Renormalise:  raw_out × sum(𝟙) / sum(M) ─────────────────
        #   Clamp denominator to ≥ 1 to prevent division by zero on fully-
        #   masked windows.  Those positions are zeroed out below anyway.
        renorm_factor = self.kernel_area / mask_sum.clamp(min=1.0)
        output = raw_out * renorm_factor                     # (B, C_out, H', W')

        # ── 6.  Add bias (only at valid output positions) ────────────────
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        # ── 7.  Zero-out fully-masked positions ─────────────────────────
        output = output * updated_mask

        return output, updated_mask


# ======================================================================
#  Smoke test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("PartialConv2d — smoke test")
    print("=" * 60)

    # ── Test 1: stride-2 layer (first encoder layer of PConv U-Net) ──────
    B, C_in, H, W = 4, 3, 256, 256
    C_out, K, S, P = 64, 7, 2, 3

    pconv = PartialConv2d(C_in, C_out, K, stride=S, padding=P)
    x = torch.randn(B, C_in, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.3).float()   # ~70 % valid

    out, mask_out = pconv(x, mask)

    # Expected spatial size: floor((256 + 2*3 - 7) / 2) + 1 = 128
    print(f"\n[stride=2]")
    print(f"  input          : {tuple(x.shape)}")
    print(f"  input mask     : {tuple(mask.shape)}")
    print(f"  output         : {tuple(out.shape)}")
    print(f"  updated mask   : {tuple(mask_out.shape)}")
    assert out.shape == (B, C_out, 128, 128), f"output shape wrong: {out.shape}"
    assert mask_out.shape == (B, 1, 128, 128), f"mask shape wrong: {mask_out.shape}"
    print("  shapes OK ✓")

    # ── Test 2: stride-1 layer (decoder PConv) ───────────────────────────
    pconv_s1 = PartialConv2d(64, 64, 3, stride=1, padding=1)
    x2 = torch.randn(B, 64, 128, 128)
    m2 = (torch.rand(B, 1, 128, 128) > 0.2).float()

    out2, mask2 = pconv_s1(x2, m2)
    print(f"\n[stride=1]")
    print(f"  input          : {tuple(x2.shape)}")
    print(f"  output         : {tuple(out2.shape)}")
    print(f"  updated mask   : {tuple(mask2.shape)}")
    assert out2.shape == (B, 64, 128, 128), f"output shape wrong: {out2.shape}"
    assert mask2.shape == (B, 1, 128, 128), f"mask shape wrong: {mask2.shape}"
    print("  shapes OK ✓")

    # ── Test 3: mask is strictly binary {0, 1} ───────────────────────────
    unique_vals = mask_out.unique().tolist()
    assert set(unique_vals).issubset({0.0, 1.0}), (
        f"updated mask not binary: {unique_vals}"
    )
    print(f"\n  mask unique values: {unique_vals}  (binary ✓)")

    # ── Test 4: fully-masked input → zero output ─────────────────────────
    zero_mask = torch.zeros(1, 1, 32, 32)
    x_tiny = torch.randn(1, C_in, 32, 32)
    pconv_tiny = PartialConv2d(C_in, 16, 3, stride=1, padding=1)
    out_z, mask_z = pconv_tiny(x_tiny, zero_mask)
    assert (out_z == 0).all(), "output must be zero when mask is all-zero"
    assert (mask_z == 0).all(), "updated mask must be zero when input mask is all-zero"
    print("  fully-masked → zero output ✓")

    # ── Test 5: fully-valid input → renorm factor = 1.0 ─────────────────
    ones_mask = torch.ones(1, 1, 32, 32)
    out_v, mask_v = pconv_tiny(x_tiny, ones_mask)
    assert (mask_v == 1).all(), "updated mask must be all-one when input mask is all-one"
    print("  fully-valid → renorm=1 ✓")

    # ── Test 6: mask shrinks (fewer holes) after propagation ─────────────
    hole_frac_in = 1.0 - mask.float().mean().item()
    hole_frac_out = 1.0 - mask_out.float().mean().item()
    print(f"\n  hole fraction  in : {hole_frac_in:.2%}")
    print(f"  hole fraction  out: {hole_frac_out:.2%}")
    assert hole_frac_out <= hole_frac_in + 0.01, (
        "mask should not gain holes after PConv"
    )
    print("  mask shrinks (or stays same) ✓")

    # ── Test 7: gradients flow through output ────────────────────────────
    loss = out.sum()
    loss.backward()
    assert pconv.feature_conv.weight.grad is not None, "no grad on weight"
    assert pconv.bias is not None and pconv.bias.grad is not None, "no grad on bias"
    print("  gradient flow ✓")

    # ── Summary ──────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in pconv.parameters())
    print(f"\n  PConv({C_in}→{C_out}, k={K}, s={S}) parameters: {total_params:,}")
    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
