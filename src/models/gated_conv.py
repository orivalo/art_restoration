"""Gated convolution layer (Yu et al., ICCV 2019).

Gated convolution replaces the hard binary mask of PConv with a **learned
per-pixel soft gate**.  Each output is the elementwise product of two
parallel convolution branches::

    feat = ELU( conv_feat(x) )
    gate = sigmoid( conv_gate(x) )
    out  = feat * gate

The gate path's sigmoid output naturally models "how confident is each
output position" — it can attend to any spatial pattern, including the
input mask channel concatenated to the RGB image, but is **not forced**
to follow the binary input mask the way PConv does.

This is the building block of DeepFillv2 (Yu et al. 2019, arXiv:1806.03589),
and a strong modern competitor to PConv on free-form irregular masks.

Parameter cost
--------------

Gated conv duplicates the convolution weights (one for the feature path,
one for the gate path), so a GatedConv2d has **roughly 2× the parameters
of an equivalent nn.Conv2d**.  When stacked into a U-Net with the same
channel ladder as PConvUNet, the resulting GatedUNet has ≈ 1.9× the
parameter count of PConvUNet.  This is documented in the project's plan
as an accepted fairness trade-off (same channel ladder = same effective
architectural capacity in width, not the same param count).

References
----------
[1] Yu et al., "Free-Form Image Inpainting with Gated Convolution",
    ICCV 2019.  arXiv:1806.03589.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedConv2d(nn.Module):
    """Gated convolution layer (Yu et al. 2019).

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Convolution kernel size.
        stride:       Convolution stride.
        padding:      Zero padding on both sides.
        bias:         If ``True``, both feat and gate convs have a bias.
        activation:   Activation applied to the feature path.  Default
            ``nn.ELU(inplace=True)``.  Pass ``None`` or ``nn.Identity()``
            for the final output layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv_feat = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.conv_gate = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.act_feat: nn.Module = activation if activation is not None else nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated convolution.

        Args:
            x: ``(B, C_in, H, W)`` input feature map.

        Returns:
            ``(B, C_out, H', W')`` gated feature map.
        """
        feat = self.act_feat(self.conv_feat(x))
        gate = torch.sigmoid(self.conv_gate(x))
        return feat * gate


# ======================================================================
#  Smoke test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("GatedConv2d — smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    layer = GatedConv2d(4, 64, kernel_size=7, stride=2, padding=3).to(device)

    x = torch.randn(2, 4, 256, 256, device=device)
    y = layer(x)

    print(f"[Test 1] forward pass")
    print(f"  input  : {tuple(x.shape)}")
    print(f"  output : {tuple(y.shape)}")
    assert y.shape == (2, 64, 128, 128), f"shape wrong: {y.shape}"
    print("  shape OK ✓")

    print(f"\n[Test 2] gradient flow")
    loss = y.sum()
    loss.backward()
    assert layer.conv_feat.weight.grad is not None
    assert layer.conv_gate.weight.grad is not None
    print("  feat path grad ✓")
    print("  gate path grad ✓")

    n = sum(p.numel() for p in layer.parameters())
    print(f"\n[Test 3] parameter count")
    print(f"  GatedConv2d(4 → 64, k=7) : {n:,} params (≈ 2× a Conv2d)")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
