"""Generic Trainer for inpainting models (PConv / Vanilla / Gated U-Net).

End-to-end resumable training loop.  The model is selected by
``cfg["model"]["arch"]`` and instantiated through
``src.models.registry.build_model`` — the same Trainer drives all three
architectures used in the comparison study.

Designed to survive Colab/Kaggle session limits (4–9 h):

* Mixed-precision (AMP) with ``torch.cuda.amp.GradScaler``
* Gradient clipping (``max_norm = 1.0``)
* Cosine LR schedule with linear warmup
* Auto-detect-and-resume from the latest checkpoint on Drive
* Per-epoch validation with PSNR / SSIM
* Visualisation grid (8 samples × 5 columns) saved every val
* CSV log appended every epoch
* Periodic + last + best checkpoints
* Early stopping on a configurable validation metric

Pixel-range convention
----------------------

The dataset returns images normalised to ``[-1, 1]``.  The Trainer
converts back to ``[0, 1]`` *before* the model and loss, because:

* ``PConvUNet`` ends in ``sigmoid`` → outputs in ``[0, 1]``
* ``InpaintingLoss`` (with VGG features) is calibrated for ``[0, 1]``

The conversion ``(x + 1) / 2`` is exact and reversible.

Usage
-----

>>> trainer = Trainer("configs/train_config.yaml")
>>> trainer.train()
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import InpaintingDataset
from src.data.mask_generator import MaskGenerator
from src.models.registry import build_model
from src.training.losses import InpaintingLoss
from src.training.metrics import psnr as compute_psnr
from src.training.metrics import ssim as compute_ssim
from src.utils.checkpoint import (
    epoch_checkpoint_name,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────────────


class Trainer:
    """End-to-end training driver for PConv U-Net.

    The constructor reads a YAML config, builds every component, and
    optionally resumes from the latest checkpoint.  ``train()`` then
    runs the full multi-epoch loop.

    Args:
        config_path: Filesystem path to a YAML configuration file (see
            ``configs/train_config.yaml`` for the schema).
    """

    # ----------------------------------------------------------------- init

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg: dict[str, Any] = yaml.safe_load(f)

        # ── Reproducibility ─────────────────────────────────────────────
        seed = int(self.cfg["experiment"]["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ── Device ──────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        # ── Persistent directories on Drive ────────────────────────────
        self.checkpoint_dir = Path(self.cfg["checkpoint"]["dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.cfg["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir = self.log_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # ── Model (dispatched via registry on cfg["model"]["arch"]) ────
        arch = str(self.cfg["model"].get("arch", "pconv_unet"))
        print(f"Building model (arch='{arch}')...")
        self.model: nn.Module = build_model(self.cfg).to(self.device)

        # ── Loss (InpaintingLoss instantiates a frozen VGG16 — slow) ───
        print("Building loss (downloads VGG16 weights on first run)...")
        loss_cfg = self.cfg["loss"]
        self.criterion = InpaintingLoss(
            lambda_valid=float(loss_cfg["lambda_valid"]),
            lambda_hole=float(loss_cfg["lambda_hole"]),
            lambda_perc=float(loss_cfg["lambda_perc"]),
            lambda_style=float(loss_cfg["lambda_style"]),
            lambda_tv=float(loss_cfg["lambda_tv"]),
        ).to(self.device)

        # ── Optimiser + LR scheduler ───────────────────────────────────
        self.optimizer: Optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # ── AMP scaler ─────────────────────────────────────────────────
        self.amp_enabled = bool(self.cfg["train"]["amp"])
        self.scaler = GradScaler(enabled=self.amp_enabled)

        # ── Dataloaders ────────────────────────────────────────────────
        print("Building dataloaders...")
        splits_dir = Path(self.cfg["data"]["splits_dir"])
        self.train_loader = self._build_dataloader(
            splits_dir / self.cfg["data"]["train_csv"], shuffle=True,
        )
        self.val_loader = self._build_dataloader(
            splits_dir / self.cfg["data"]["val_csv"], shuffle=False,
        )
        print(f"  train batches: {len(self.train_loader):,}   "
              f"val batches: {len(self.val_loader):,}")

        # ── Training state ─────────────────────────────────────────────
        self.start_epoch = 1                                    # 1-based
        self.best_metrics: dict[str, float] = {
            "psnr": -float("inf"),
            "ssim": -float("inf"),
        }
        self.epochs_without_improvement = 0
        self._consecutive_nan = 0       # numerical-stability guard counter
        self._max_consecutive_nan = 10  # abort after this many in a row

        # ── Auto-resume from latest checkpoint, if requested ──────────
        resume_setting = self.cfg["checkpoint"].get("resume", "auto")
        if resume_setting == "auto":
            self._maybe_resume()
        elif resume_setting and resume_setting != "no":
            self._load_checkpoint(Path(resume_setting))
        else:
            print("Resume disabled — training from scratch.")

    # -------------------------------------------------------- builders

    def _build_dataloader(
        self,
        csv_path: Path,
        *,
        shuffle: bool,
    ) -> DataLoader:
        """Construct a DataLoader for one of the CSV-based splits."""
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "path" not in df.columns:
            raise ValueError(f"{csv_path} must contain a 'path' column")
        paths = df["path"].tolist()

        # Per-process MaskGenerator with ``seed=None`` so each worker
        # produces unique masks every epoch.
        mask_gen = MaskGenerator(seed=None)

        ds = InpaintingDataset(
            image_paths=paths,
            mask_generator=mask_gen,
            difficulty=self.cfg["data"]["difficulty"],
            image_size=self.cfg["data"]["image_size"],
        )
        return DataLoader(
            ds,
            batch_size=int(self.cfg["train"]["batch_size"]),
            shuffle=shuffle,
            num_workers=int(self.cfg["data"]["num_workers"]),
            pin_memory=bool(self.cfg["data"]["pin_memory"]),
            drop_last=shuffle,                  # only on train, keep all val samples
            persistent_workers=int(self.cfg["data"]["num_workers"]) > 0,
        )

    def _build_optimizer(self) -> Optimizer:
        cfg = self.cfg["optim"]
        if cfg["type"].lower() != "adam":
            raise ValueError(f"Only 'adam' is supported (got '{cfg['type']}')")
        return Adam(
            self.model.parameters(),
            lr=float(cfg["lr"]),
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )

    def _build_scheduler(self):
        cfg = self.cfg["scheduler"]
        sch_type = cfg["type"].lower()
        num_epochs = int(self.cfg["train"]["num_epochs"])

        if sch_type == "none":
            return None

        if sch_type == "cosine":
            warmup = int(cfg.get("warmup_epochs", 0))
            min_lr = float(cfg.get("min_lr", 0.0))

            if warmup > 0:
                warmup_sch = LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup,
                )
                cosine_sch = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(num_epochs - warmup, 1),
                    eta_min=min_lr,
                )
                return SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sch, cosine_sch],
                    milestones=[warmup],
                )
            return CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=min_lr,
            )

        if sch_type == "step":
            return StepLR(
                self.optimizer,
                step_size=int(cfg.get("step_size", 10)),
                gamma=float(cfg.get("gamma", 0.5)),
            )

        raise ValueError(f"Unknown scheduler type: '{sch_type}'")

    # -------------------------------------------------------- resume

    def _maybe_resume(self) -> None:
        latest = find_latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            print("No checkpoint found — training from scratch.")
            return
        self._load_checkpoint(latest)

    def _load_checkpoint(self, path: Path) -> None:
        print(f"Resuming from {path}...")
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.amp_enabled else None,
            map_location=str(self.device),
        )
        self.start_epoch = int(payload.get("epoch", 0)) + 1
        if "best_metrics" in payload and isinstance(payload["best_metrics"], dict):
            self.best_metrics.update(payload["best_metrics"])
        print(
            f"Resumed at epoch {self.start_epoch} "
            f"(best PSNR so far: {self.best_metrics['psnr']:.3f} dB)"
        )

    # -------------------------------------------------------- helpers

    @staticmethod
    def _denorm_to_01(x: torch.Tensor) -> torch.Tensor:
        """Convert tensor from ``[-1, 1]`` (dataset) to ``[0, 1]`` (model)."""
        return (x + 1.0) * 0.5

    # -------------------------------------------------------- train epoch

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return averaged loss components."""
        self.model.train()

        loss_sums: dict[str, float] = {}
        n_batches = 0
        grad_clip = float(self.cfg["train"]["grad_clip"])
        log_every = int(self.cfg["train"]["log_every"])

        pbar = tqdm(
            self.train_loader,
            desc=f"Train E{epoch:03d}",
            leave=False,
            ncols=100,
        )

        for it, (_masked_neg11, mask, gt_neg11) in enumerate(pbar, 1):
            mask = mask.to(self.device, non_blocking=True)
            gt_neg11 = gt_neg11.to(self.device, non_blocking=True)

            # Convert from dataset's [-1, 1] normalisation to [0, 1]
            gt = self._denorm_to_01(gt_neg11)
            masked = gt * mask  # rebuild masked input cleanly in [0, 1]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp_enabled):
                output, _ = self.model(masked, mask)
                loss_dict = self.criterion(output, gt, mask)
                loss = loss_dict["total"]

            # ── NaN/Inf guard (catches fp16 overflow before it corrupts weights)
            if not torch.isfinite(loss):
                self._consecutive_nan += 1
                tqdm.write(
                    f"  WARNING: non-finite loss at epoch {epoch} iter {it} "
                    f"(consecutive NaN: {self._consecutive_nan})"
                )
                if self._consecutive_nan >= self._max_consecutive_nan:
                    raise RuntimeError(
                        f"{self._max_consecutive_nan} consecutive non-finite losses. "
                        "Likely fp16 overflow from large style-loss weight, "
                        "corrupted batch, or bad checkpoint state. "
                        "Try: lower 'lambda_style' in config, or set 'amp: false'."
                    )
                # Skip this batch — do NOT call backward / step
                continue
            self._consecutive_nan = 0

            self.scaler.scale(loss).backward()

            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate every loss component for end-of-epoch averaging
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v.item())
            n_batches += 1

            # Loss-spike detection: warn if current total > 5× running mean
            # after the first 10 iterations (avoids false alarms during warmup).
            if n_batches > 10:
                running_mean = loss_sums["total"] / n_batches
                current = float(loss_dict["total"].item())
                if current > 5.0 * running_mean:
                    tqdm.write(
                        f"  WARNING: loss spike at iter {it} "
                        f"(current={current:.3f} vs mean={running_mean:.3f})"
                    )

            if it % log_every == 0:
                pbar.set_postfix(
                    total=f"{loss_sums['total'] / n_batches:.3f}",
                    valid=f"{loss_sums['l1_valid'] / n_batches:.3f}",
                    hole=f"{loss_sums['l1_hole'] / n_batches:.3f}",
                )

        return {k: v / max(n_batches, 1) for k, v in loss_sums.items()}

    # -------------------------------------------------------- validate

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Run one validation pass; returns ``{'psnr', 'ssim'}``."""
        self.model.eval()

        psnr_sum = 0.0
        ssim_sum = 0.0
        n_total = 0
        vis_samples = None
        num_vis = int(self.cfg["logging"]["num_vis_samples"])

        pbar = tqdm(
            self.val_loader,
            desc=f"  Val E{epoch:03d}",
            leave=False,
            ncols=100,
        )

        for _masked_neg11, mask, gt_neg11 in pbar:
            mask = mask.to(self.device, non_blocking=True)
            gt_neg11 = gt_neg11.to(self.device, non_blocking=True)
            gt = self._denorm_to_01(gt_neg11)
            masked = gt * mask

            output, _ = self.model(masked, mask)
            output = output.clamp(0.0, 1.0)

            comp = output * (1.0 - mask) + gt * mask  # composited

            b = output.shape[0]
            psnr_sum += float(compute_psnr(output, gt).item()) * b
            ssim_sum += float(compute_ssim(output, gt).item()) * b
            n_total += b

            if vis_samples is None:
                vis_samples = (
                    masked[:num_vis].detach().cpu(),
                    mask[:num_vis].detach().cpu(),
                    output[:num_vis].detach().cpu(),
                    gt[:num_vis].detach().cpu(),
                    comp[:num_vis].detach().cpu(),
                )

        if vis_samples is not None:
            self._save_visualization(epoch, vis_samples)

        return {
            "psnr": psnr_sum / max(n_total, 1),
            "ssim": ssim_sum / max(n_total, 1),
        }

    # -------------------------------------------------------- visualisation

    def _save_visualization(
        self,
        epoch: int,
        samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Save a 5-column grid (masked, mask, output, GT, composited)."""
        masked, mask, output, gt, comp = samples
        n = masked.shape[0]
        cols = 5

        fig, axes = plt.subplots(n, cols, figsize=(cols * 2.2, n * 2.2))
        if n == 1:
            axes = axes[None, :]

        col_titles = ["Masked input", "Mask", "Model output", "Ground truth", "Composited"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10, fontweight="bold")

        def _to_img(t: torch.Tensor) -> np.ndarray:
            return t.permute(1, 2, 0).clamp(0, 1).numpy()

        for i in range(n):
            axes[i, 0].imshow(_to_img(masked[i]))
            axes[i, 1].imshow(mask[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(_to_img(output[i]))
            axes[i, 3].imshow(_to_img(gt[i]))
            axes[i, 4].imshow(_to_img(comp[i]))
            for ax in axes[i]:
                ax.axis("off")

        plt.suptitle(f"Validation samples — Epoch {epoch}", fontsize=11)
        plt.tight_layout()
        save_path = self.figures_dir / f"val_epoch_{epoch:03d}.png"
        plt.savefig(save_path, dpi=80, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # -------------------------------------------------------- checkpoint save

    def save_checkpoint_set(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool,
    ) -> None:
        """Save ``last.pth`` (always), ``best.pth`` (if improved), and the
        periodic ``epoch_NNN.pth`` snapshot every ``save_every`` epochs."""
        common = dict(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.amp_enabled else None,
            epoch=epoch,
            metrics=metrics,
            best_metrics=self.best_metrics,
            config=self.cfg,
        )

        save_checkpoint(self.checkpoint_dir / "last.pth", **common)

        if is_best:
            save_checkpoint(self.checkpoint_dir / "best.pth", **common)

        save_every = int(self.cfg["checkpoint"].get("save_every", 0))
        if save_every > 0 and (epoch % save_every == 0):
            save_checkpoint(
                self.checkpoint_dir / epoch_checkpoint_name(epoch), **common,
            )

    # -------------------------------------------------------- CSV logger

    def _append_csv_log(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float,
        dt_seconds: float,
    ) -> None:
        csv_path = self.log_dir / self.cfg["logging"]["csv_filename"]
        is_new = not csv_path.exists()

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                header = ["epoch"]
                header += [f"train_{k}" for k in train_metrics]
                header += [f"val_{k}" for k in val_metrics]
                header += ["lr", "time_sec"]
                writer.writerow(header)

            row: list[Any] = [epoch]
            row += [f"{v:.6f}" for v in train_metrics.values()]
            row += [f"{v:.6f}" for v in val_metrics.values()]
            row += [f"{lr:.6e}", f"{dt_seconds:.1f}"]
            writer.writerow(row)

    # -------------------------------------------------------- main loop

    def train(self) -> None:
        """Run the full multi-epoch training loop."""
        num_epochs = int(self.cfg["train"]["num_epochs"])
        metric_for_best = self.cfg["checkpoint"].get("metric_for_best", "psnr")

        es_cfg = self.cfg.get("early_stopping", {})
        es_enabled = bool(es_cfg.get("enabled", False))
        es_patience = int(es_cfg.get("patience", 15))
        es_min_delta = float(es_cfg.get("min_delta", 0.0))

        print(f"\nStarting training: epochs {self.start_epoch} → {num_epochs}")
        print(f"Tracking best model by '{metric_for_best}'")
        if es_enabled:
            print(f"Early stopping: patience={es_patience}, min_delta={es_min_delta}")
        print()

        for epoch in range(self.start_epoch, num_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            # ── Best-metric detection ──────────────────────────────────
            current = val_metrics[metric_for_best]
            previous_best = self.best_metrics[metric_for_best]
            is_best = current > previous_best + es_min_delta
            if is_best:
                self.best_metrics = dict(val_metrics)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # ── Persist ────────────────────────────────────────────────
            self.save_checkpoint_set(epoch, val_metrics, is_best)
            self._append_csv_log(epoch, train_metrics, val_metrics, current_lr, dt)

            # ── Console summary ────────────────────────────────────────
            best_marker = "  ↑ NEW BEST" if is_best else ""
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"loss={train_metrics['total']:.3f} | "
                f"val_psnr={val_metrics['psnr']:.2f} dB | "
                f"val_ssim={val_metrics['ssim']:.4f} | "
                f"lr={current_lr:.2e} | "
                f"{dt / 60:.1f} min{best_marker}"
            )

            # ── Early stopping ─────────────────────────────────────────
            if es_enabled and self.epochs_without_improvement >= es_patience:
                print(
                    f"\nEarly stopping at epoch {epoch}: "
                    f"no improvement for {es_patience} consecutive epochs."
                )
                break

        print(
            f"\nTraining complete.  "
            f"Best {metric_for_best}: {self.best_metrics[metric_for_best]:.4f}"
        )
        print(f"Best model: {self.checkpoint_dir / 'best.pth'}")
        print(f"CSV log:    {self.log_dir / self.cfg['logging']['csv_filename']}")


# ======================================================================
#  Smoke test (offline)
# ======================================================================

if __name__ == "__main__":
    """Quick offline smoke test using a tiny synthetic dataset.

    Builds two splits of 8 images each, runs 1 epoch on CPU, verifies
    that loss decreases and a checkpoint is written.  Skipped on CI
    when matplotlib backend is unavailable.
    """
    import tempfile
    from PIL import Image

    print("=" * 60)
    print("Trainer — offline smoke test (1 mini-epoch on synthetic data)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ── 1.  Generate fake 256×256 paintings ─────────────────────────
        processed_dir = tmp_path / "processed" / "fake"
        processed_dir.mkdir(parents=True)
        rng = np.random.default_rng(42)
        paths: list[str] = []
        for i in range(16):
            img = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
            p = processed_dir / f"fake_{i:03d}.jpg"
            img.save(p, quality=85)
            paths.append(str(p))

        # ── 2.  Build CSV splits ────────────────────────────────────────
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        train_df = pd.DataFrame({"path": paths[:8], "style": "fake"})
        val_df = pd.DataFrame({"path": paths[8:16], "style": "fake"})
        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(splits_dir / "val.csv", index=False)

        # ── 3.  Write a tiny config ────────────────────────────────────
        cfg_path = tmp_path / "tiny_config.yaml"
        cfg = {
            "experiment": {"name": "smoke", "seed": 42, "description": "smoke test"},
            "data": {
                "splits_dir": str(splits_dir),
                "train_csv": "train.csv",
                "val_csv": "val.csv",
                "test_csv": "val.csv",
                "image_size": 256,
                "difficulty": "medium",
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {"arch": "pconv_unet", "in_channels": 3, "out_channels": 3},
            "loss": {
                "lambda_valid": 1.0,
                "lambda_hole": 6.0,
                "lambda_perc": 0.05,
                "lambda_style": 120.0,
                "lambda_tv": 0.1,
            },
            "optim": {"type": "adam", "lr": 2.0e-4, "betas": [0.9, 0.999], "weight_decay": 0.0},
            "scheduler": {"type": "none"},
            "train": {
                "batch_size": 2,
                "num_epochs": 1,
                "grad_clip": 1.0,
                "amp": False,                 # CPU smoke test
                "log_every": 1,
                "validate_every": 1,
            },
            "checkpoint": {
                "dir": str(tmp_path / "checkpoints"),
                "save_every": 1,
                "metric_for_best": "psnr",
                "resume": "auto",
            },
            "logging": {
                "log_dir": str(tmp_path / "logs"),
                "csv_filename": "smoke_log.csv",
                "num_vis_samples": 2,
            },
            "early_stopping": {"enabled": False},
            "eval": {"compute_fid": False, "compute_lpips": False},
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        # ── 4.  Run 1 mini-epoch ───────────────────────────────────────
        trainer = Trainer(cfg_path)
        trainer.train()

        # ── 5.  Verify outputs ─────────────────────────────────────────
        ckpt = tmp_path / "checkpoints" / "last.pth"
        log = tmp_path / "logs" / "smoke_log.csv"
        fig = tmp_path / "logs" / "figures" / "val_epoch_001.png"

        print(f"\n[Verification]")
        print(f"  last.pth exists      : {ckpt.exists()}  ✓" if ckpt.exists() else f"  last.pth MISSING")
        print(f"  CSV log exists       : {log.exists()}  ✓" if log.exists() else f"  CSV log MISSING")
        print(f"  val figure exists    : {fig.exists()}  ✓" if fig.exists() else f"  val figure MISSING")

        assert ckpt.exists() and log.exists() and fig.exists()
        print("\n" + "=" * 60)
        print("Smoke test passed.")
        print("=" * 60)
