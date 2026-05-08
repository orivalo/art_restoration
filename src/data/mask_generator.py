"""Synthetic mask generator for art inpainting.

Produces binary damage masks that simulate four real-world artwork degradation
types: brush strokes, cracks, paint loss, and aging stains. Masks are generated
on the fly (never written to disk) and follow the Liu et al. 2018 convention:
1 = valid pixel, 0 = hole (missing / damaged region).
"""

from __future__ import annotations

import random
from typing import Callable, Literal

import cv2
import numpy as np
from numpy.typing import NDArray


DifficultyLevel = Literal["light", "medium", "heavy"]

# Hole-area ratio ranges per difficulty level (fraction of total pixels)
DIFFICULTY_RANGES: dict[DifficultyLevel, tuple[float, float]] = {
    "light": (0.10, 0.20),
    "medium": (0.20, 0.40),
    "heavy": (0.40, 0.60),
}


class MaskGenerator:
    """Generates binary damage masks for artwork inpainting on the fly.

    Combines up to four damage-type primitives (brush strokes, cracks,
    paint loss, aging stains) to reach a target hole-area ratio determined
    by the chosen difficulty level.

    Mask convention (matches Liu et al. ECCV 2018):
        1 → valid pixel (intact)
        0 → hole (missing / damaged)

    Args:
        seed: Fixed seed for reproducible masks. Pass ``None`` (default)
            during training so each epoch sees different masks.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        h: int,
        w: int,
        difficulty: DifficultyLevel = "medium",
    ) -> NDArray[np.float32]:
        """Generate a composite damage mask for a single image.

        Iterates through the four damage-type primitives in a random order,
        accumulating holes until the target area ratio is reached.

        Args:
            h: Mask height in pixels.
            w: Mask width in pixels.
            difficulty: Hole-area difficulty — ``'light'`` (10–20 %),
                ``'medium'`` (20–40 %), or ``'heavy'`` (40–60 %).

        Returns:
            Float32 array of shape ``(h, w)`` with values in ``{0.0, 1.0}``.
        """
        lo, hi = DIFFICULTY_RANGES[difficulty]
        target_ratio = float(self._rng.uniform(lo, hi))
        target_holes = int(target_ratio * h * w)

        mask = np.ones((h, w), dtype=np.float32)

        damage_fns: list[Callable[[int, int, float], NDArray[np.float32]]] = [
            self.random_brush_strokes,
            self.simulated_cracks,
            self.simulated_paint_loss,
            self.random_aging_stains,
        ]
        self._rng.shuffle(damage_fns)  # type: ignore[arg-type]

        for fn in damage_fns:
            current_holes = int((mask == 0).sum())
            if current_holes >= target_holes:
                break
            remaining_ratio = (target_holes - current_holes) / (h * w)
            # Over-drive density slightly so we reach the target in fewer passes
            density = float(np.clip(remaining_ratio * 2.5, 0.05, 1.0))
            damage = fn(h, w, density)
            mask = np.minimum(mask, damage)

        current_holes = int((mask == 0).sum())
        if current_holes != target_holes:
            flat_mask = mask.reshape(-1)
            if current_holes < target_holes:
                valid_indices = np.flatnonzero(flat_mask == 1.0)
                num_to_flip = min(target_holes - current_holes, valid_indices.size)
                if num_to_flip > 0:
                    chosen = self._rng.choice(valid_indices, size=num_to_flip, replace=False)
                    flat_mask[chosen] = 0.0
            else:
                hole_indices = np.flatnonzero(flat_mask == 0.0)
                num_to_flip = min(current_holes - target_holes, hole_indices.size)
                if num_to_flip > 0:
                    chosen = self._rng.choice(hole_indices, size=num_to_flip, replace=False)
                    flat_mask[chosen] = 1.0

        return mask.astype(np.float32)

    def random_brush_strokes(
        self,
        h: int,
        w: int,
        density: float = 0.5,
    ) -> NDArray[np.float32]:
        """Simulate irregular brush-stroke damage via quadratic Bézier curves.

        Args:
            h: Mask height in pixels.
            w: Mask width in pixels.
            density: Controls stroke count and width; clamped to ``[0, 1]``.

        Returns:
            Float32 array ``(h, w)`` — ``0`` where stroke damage occurred.
        """
        mask = np.ones((h, w), dtype=np.uint8)
        num_strokes = max(1, int(density * 12))
        short = min(h, w)

        for _ in range(num_strokes):
            num_pts = int(self._rng.integers(4, 10))
            pts = [
                (int(self._rng.uniform(0, w)), int(self._rng.uniform(0, h)))
                for _ in range(num_pts)
            ]
            thickness = int(self._rng.uniform(
                max(3, short * 0.02),
                max(10, short * 0.12 * density),
            ))
            for i in range(len(pts) - 1):
                cx = int((pts[i][0] + pts[i + 1][0]) / 2
                         + self._rng.uniform(-w * 0.1, w * 0.1))
                cy = int((pts[i][1] + pts[i + 1][1]) / 2
                         + self._rng.uniform(-h * 0.1, h * 0.1))
                curve = self._bezier_points(pts[i], (cx, cy), pts[i + 1])
                for j in range(len(curve) - 1):
                    cv2.line(mask, curve[j], curve[j + 1], 0, thickness)

        return mask.astype(np.float32)

    def simulated_cracks(
        self,
        h: int,
        w: int,
        density: float = 0.5,
    ) -> NDArray[np.float32]:
        """Simulate thin, branching crack networks.

        Cracks are drawn recursively as thin branching polylines to mimic
        the fractal-like appearance of dried-paint or varnish cracking.

        Args:
            h: Mask height in pixels.
            w: Mask width in pixels.
            density: Controls crack count and branching depth; clamped to
                ``[0, 1]``.

        Returns:
            Float32 array ``(h, w)`` — ``0`` along crack paths.
        """
        mask = np.ones((h, w), dtype=np.uint8)
        num_cracks = max(1, int(density * 8))
        max_depth = max(2, int(density * 4 + 2))

        for _ in range(num_cracks):
            x = int(self._rng.uniform(0, w))
            y = int(self._rng.uniform(0, h))
            angle = float(self._rng.uniform(0, 2 * np.pi))
            self._draw_crack(mask, x, y, angle, h, w, depth=0, max_depth=max_depth)

        return mask.astype(np.float32)

    def simulated_paint_loss(
        self,
        h: int,
        w: int,
        density: float = 0.5,
    ) -> NDArray[np.float32]:
        """Simulate large paint-flaking or delamination regions.

        Irregular blobs are drawn as distorted ellipses using Gaussian noise
        on the boundary to achieve an organic, non-geometric appearance.

        Args:
            h: Mask height in pixels.
            w: Mask width in pixels.
            density: Controls blob count and maximum size; clamped to ``[0, 1]``.

        Returns:
            Float32 array ``(h, w)`` — ``0`` inside paint-loss regions.
        """
        mask = np.ones((h, w), dtype=np.float32)
        num_blobs = max(1, int(density * 6))
        short = min(h, w)

        for _ in range(num_blobs):
            cx = int(self._rng.uniform(0, w))
            cy = int(self._rng.uniform(0, h))
            # Guard against degenerate uniform(low, high) when density is so
            # small that 0.25 * short * density + 1 < 0.05 * short.
            lo_axis = max(2, int(short * 0.05))
            hi_axis = max(lo_axis + 1, int(short * 0.25 * density) + 1)
            ax = int(self._rng.uniform(lo_axis, hi_axis))
            ay = int(self._rng.uniform(lo_axis, hi_axis))
            rot = int(self._rng.uniform(0, 180))

            blob = np.ones((h, w), dtype=np.uint8)
            cv2.ellipse(blob, (cx, cy), (ax, ay), rot, 0, 360, 0, -1)

            # Perturb boundary with smoothed noise for organic look
            noise = self._rng.standard_normal((h, w)).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (21, 21), 0)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

            blob_f = np.where(blob == 0, noise * 0.6, 1.0).astype(np.float32)
            mask = np.minimum(mask, np.where(blob_f > 0.5, 1.0, 0.0))

        return mask.astype(np.float32)

    def random_aging_stains(
        self,
        h: int,
        w: int,
        density: float = 0.5,
    ) -> NDArray[np.float32]:
        """Simulate soft-edged aging stains via Gaussian intensity blobs.

        Overlapping Gaussian blobs create a stain intensity map; pixels
        above a random threshold are marked as holes, producing soft,
        gradient-bounded damage regions.

        Args:
            h: Mask height in pixels.
            w: Mask width in pixels.
            density: Controls stain count and spread; clamped to ``[0, 1]``.

        Returns:
            Float32 array ``(h, w)`` — ``0`` inside stain regions.
        """
        stain_map = np.zeros((h, w), dtype=np.float32)
        num_stains = max(1, int(density * 8))

        ys, xs = np.ogrid[:h, :w]
        # Guard against degenerate uniform(low, high) when density is so
        # small that 0.20 * dim * density + 1 < 0.04 * dim.
        lo_sx = max(2.0, w * 0.04)
        hi_sx = max(lo_sx + 1.0, w * 0.20 * density + 1.0)
        lo_sy = max(2.0, h * 0.04)
        hi_sy = max(lo_sy + 1.0, h * 0.20 * density + 1.0)
        for _ in range(num_stains):
            cx = float(self._rng.uniform(0, w))
            cy = float(self._rng.uniform(0, h))
            sx = float(self._rng.uniform(lo_sx, hi_sx))
            sy = float(self._rng.uniform(lo_sy, hi_sy))
            intensity = float(self._rng.uniform(0.6, 1.0))
            blob = intensity * np.exp(
                -((xs - cx) ** 2 / (2 * sx ** 2) + (ys - cy) ** 2 / (2 * sy ** 2))
            )
            stain_map = np.maximum(stain_map, blob)

        threshold = float(self._rng.uniform(0.3, 0.6))
        return np.where(stain_map > threshold, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bezier_points(
        self,
        p0: tuple[int, int],
        p1: tuple[int, int],
        p2: tuple[int, int],
        num_points: int = 20,
    ) -> list[tuple[int, int]]:
        """Sample points along a quadratic Bézier curve.

        Args:
            p0: Start point ``(x, y)``.
            p1: Control point ``(x, y)``.
            p2: End point ``(x, y)``.
            num_points: Number of evenly spaced samples along the curve.

        Returns:
            List of integer ``(x, y)`` tuples suitable for ``cv2.line``.
        """
        t_vals = np.linspace(0.0, 1.0, num_points)
        return [
            (
                int((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]),
                int((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]),
            )
            for t in t_vals
        ]

    def _draw_crack(
        self,
        mask: NDArray[np.uint8],
        x: int,
        y: int,
        angle: float,
        h: int,
        w: int,
        depth: int,
        max_depth: int,
        length: int | None = None,
    ) -> None:
        """Recursively draw a branching crack segment in-place.

        Args:
            mask: Uint8 mask array modified in place (0 = hole).
            x: Current x position.
            y: Current y position.
            angle: Current direction in radians.
            h: Mask height.
            w: Mask width.
            depth: Current recursion depth.
            max_depth: Maximum branching depth before recursion stops.
            length: Total crack length in pixels; auto-computed if ``None``.
        """
        if depth >= max_depth:
            return
        if length is None:
            length = int(self._rng.uniform(min(h, w) * 0.05, min(h, w) * 0.20))

        thickness = max(1, 2 - depth)
        num_segments = int(self._rng.integers(3, 8))
        seg_len = max(1, length // num_segments)

        for _ in range(num_segments):
            angle += float(self._rng.uniform(-0.4, 0.4))
            nx = int(np.clip(x + seg_len * np.cos(angle), 0, w - 1))
            ny = int(np.clip(y + seg_len * np.sin(angle), 0, h - 1))
            cv2.line(mask, (x, y), (nx, ny), 0, thickness)
            x, y = nx, ny

        # Stochastic branching
        if depth < max_depth - 1 and self._rng.random() < 0.6:
            direction = float(self._rng.choice([-1, 1]))  # type: ignore[arg-type]
            branch_angle = angle + direction * float(self._rng.uniform(0.3, 1.0))
            self._draw_crack(
                mask, x, y, branch_angle, h, w,
                depth + 1, max_depth, int(length * 0.6),
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path("outputs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = MaskGenerator(seed=42)
    difficulties: list[DifficultyLevel] = ["light", "medium", "heavy"]
    method_names = ["brush_strokes", "cracks", "paint_loss", "aging_stains"]
    method_fns = [
        gen.random_brush_strokes,
        gen.simulated_cracks,
        gen.simulated_paint_loss,
        gen.random_aging_stains,
    ]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for row, diff in enumerate(difficulties):
        for col, (name, fn) in enumerate(zip(method_names, method_fns)):
            mask = fn(256, 256, density=0.4)
            hole_pct = 100.0 * (mask == 0).mean()
            axes[row, col].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{diff} / {name}\n{hole_pct:.1f}% hole")
            axes[row, col].axis("off")

    for diff in difficulties:
        m = gen.generate(256, 256, difficulty=diff)
        lo, hi = DIFFICULTY_RANGES[diff]
        hole_frac = (m == 0).mean()
        assert lo - 0.05 <= hole_frac <= hi + 0.10, (
            f"generate({diff}) hole fraction {hole_frac:.3f} out of range [{lo},{hi}]"
        )
        print(f"generate('{diff}'): hole area = {hole_frac:.1%}  ✓")

    plt.tight_layout()
    save_path = out_dir / "mask_sanity.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Smoke test passed. Grid saved → {save_path}")
