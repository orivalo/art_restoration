"""Checkpoint save / load utilities for resumable training.

Designed for Colab/Kaggle where sessions die after 4–9 hours.  Every
checkpoint contains the full state needed to resume training without
loss: model weights, optimizer state, LR scheduler state, AMP grad-scaler
state, current epoch, best metrics, and the training config snapshot.

Filename convention inside the checkpoint directory:

* ``last.pth``         — overwritten every epoch, always reflects most
  recent state.  Loaded automatically by the trainer on restart.
* ``best.pth``         — overwritten whenever a new best validation
  metric is seen.  Used for final evaluation and the demo app.
* ``epoch_007.pth``    — three-digit-padded periodic snapshots (every N
  epochs) so individual milestones survive a corrupted ``last.pth``.

All checkpoints are saved with ``torch.save`` (Pickle protocol 2,
backwards-compatible) onto the path supplied — typically a Google Drive
mount such as ``/content/drive/MyDrive/art_restoration/checkpoints/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


_EPOCH_FILE_RE = re.compile(r"epoch_(\d+)\.pth$")


# ──────────────────────────────────────────────────────────────────────────────
#  Save
# ──────────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    metrics: Optional[dict[str, float]] = None,
    best_metrics: Optional[dict[str, float]] = None,
    config: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Serialise the full training state to ``path``.

    Args:
        path: Destination ``.pth`` file.  Parent directory will be
            created if missing.
        model: The model whose ``state_dict()`` is saved.  If wrapped in
            ``DataParallel`` / ``DistributedDataParallel`` the
            ``.module`` is unwrapped automatically.
        optimizer: Optimizer whose ``state_dict()`` is saved.
        epoch: Current epoch number (1-based recommended).
        scheduler: Optional LR scheduler state to save.
        scaler: Optional ``torch.cuda.amp.GradScaler`` state to save.
        metrics: Optional dict of metrics for the *current* epoch.
        best_metrics: Optional dict of best metrics observed so far.
        config: Optional snapshot of the training config (YAML loaded
            into a dict).  Stored for reproducibility / sanity checks
            on resume.
        extra: Free-form dict for any additional state.

    Returns:
        ``Path`` of the saved checkpoint file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DataParallel / DDP to keep the state dict portable
    state_dict_model = (
        model.module.state_dict()              # type: ignore[attr-defined]
        if hasattr(model, "module")
        else model.state_dict()
    )

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model": state_dict_model,
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    if best_metrics is not None:
        payload["best_metrics"] = best_metrics
    if config is not None:
        payload["config"] = config
    if extra is not None:
        payload["extra"] = extra

    # Save atomically: write to .tmp, then rename — protects against
    # session crashes mid-write that would leave a corrupted .pth.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
#  Load
# ──────────────────────────────────────────────────────────────────────────────


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Optional[str | torch.device] = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore training state from a checkpoint file.

    The model is updated in place via ``load_state_dict``.  Optimizer /
    scheduler / scaler are also restored in place if provided.

    Args:
        path: Path to the ``.pth`` checkpoint.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional LR scheduler to restore.
        scaler: Optional ``GradScaler`` to restore.
        map_location: Passed through to ``torch.load`` (e.g. ``"cuda"``
            or ``"cpu"``).  Defaults to current device.
        strict: Forwarded to ``model.load_state_dict``.

    Returns:
        The full payload dict (all keys that were present at save time)
        — useful for retrieving ``epoch``, ``best_metrics``, ``config``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location)

    # Tolerate DataParallel / DDP wrappers on either side
    target_model = model.module if hasattr(model, "module") else model  # type: ignore[attr-defined]
    target_model.load_state_dict(payload["model"], strict=strict)

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])

    return payload


# ──────────────────────────────────────────────────────────────────────────────
#  Discovery helpers
# ──────────────────────────────────────────────────────────────────────────────


def find_latest_checkpoint(directory: str | Path) -> Optional[Path]:
    """Locate the most recent checkpoint inside ``directory``.

    Resolution order:
        1. ``last.pth`` if it exists  (always the freshest snapshot).
        2. The ``epoch_NNN.pth`` with the largest ``NNN``.
        3. ``None`` if no checkpoints are found.

    Args:
        directory: Folder to scan.  Non-existence is treated as empty.

    Returns:
        ``Path`` of the most recent checkpoint, or ``None``.
    """
    directory = Path(directory)
    if not directory.exists():
        return None

    last = directory / "last.pth"
    if last.exists():
        return last

    epoch_files: list[tuple[int, Path]] = []
    for f in directory.iterdir():
        m = _EPOCH_FILE_RE.match(f.name)
        if m:
            epoch_files.append((int(m.group(1)), f))

    if not epoch_files:
        return None

    epoch_files.sort(key=lambda x: x[0])
    return epoch_files[-1][1]


def epoch_checkpoint_name(epoch: int) -> str:
    """Standard filename for the periodic per-epoch snapshot."""
    return f"epoch_{epoch:03d}.pth"


# ======================================================================
#  Smoke test
# ======================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Checkpoint utilities — smoke test")
    print("=" * 60)

    # ── Build a tiny network + optimizer for round-trip testing ─────────
    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model_a = _Tiny()
    optim_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    sched_a = torch.optim.lr_scheduler.StepLR(optim_a, step_size=10, gamma=0.5)

    # Take one optimizer step so optimizer state is non-empty
    out = model_a(torch.randn(2, 8)).sum()
    out.backward()
    optim_a.step()
    sched_a.step()

    # ── Test 1: save / load round-trip ──────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "checkpoints"
        path1 = ckpt_dir / "last.pth"

        print("\n[Test 1] save_checkpoint")
        save_checkpoint(
            path1,
            model=model_a, optimizer=optim_a, scheduler=sched_a,
            epoch=7,
            metrics={"psnr": 28.4, "ssim": 0.91},
            best_metrics={"psnr": 28.4, "ssim": 0.91},
            config={"lr": 1e-3, "batch_size": 4},
        )
        assert path1.exists(), "checkpoint file missing"
        size_kb = path1.stat().st_size / 1024
        print(f"  saved → {path1.name}  ({size_kb:.1f} KB)")
        print("  PASSED")

        # ── Test 2: load into fresh model + optimizer ───────────────────
        print("\n[Test 2] load_checkpoint")
        model_b = _Tiny()
        optim_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
        sched_b = torch.optim.lr_scheduler.StepLR(optim_b, step_size=10, gamma=0.5)

        payload = load_checkpoint(
            path1,
            model=model_b, optimizer=optim_b, scheduler=sched_b,
            map_location="cpu",
        )
        # Compare a parameter
        w_a = model_a.fc.weight.detach()
        w_b = model_b.fc.weight.detach()
        assert torch.allclose(w_a, w_b), "model weights mismatch after load"
        assert payload["epoch"] == 7
        assert payload["metrics"]["psnr"] == 28.4
        assert payload["config"]["batch_size"] == 4
        print(f"  epoch        : {payload['epoch']}")
        print(f"  best_metrics : {payload['best_metrics']}")
        print(f"  config keys  : {list(payload['config'].keys())}")
        print("  PASSED")

        # ── Test 3: find_latest_checkpoint with last.pth present ────────
        print("\n[Test 3] find_latest_checkpoint (with last.pth)")
        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None and latest.name == "last.pth"
        print(f"  latest = {latest.name}")
        print("  PASSED")

        # ── Test 4: find_latest_checkpoint falling back to epoch files ──
        print("\n[Test 4] find_latest_checkpoint (epoch fallback)")
        path1.unlink()                            # remove last.pth
        for ep in [5, 10, 7]:                     # save in scrambled order
            save_checkpoint(
                ckpt_dir / epoch_checkpoint_name(ep),
                model=model_a, optimizer=optim_a, epoch=ep,
            )
        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None and latest.name == "epoch_010.pth", \
            f"expected epoch_010.pth, got {latest}"
        print(f"  latest = {latest.name}  (highest epoch wins)")
        print("  PASSED")

        # ── Test 5: empty directory returns None ────────────────────────
        print("\n[Test 5] empty directory returns None")
        empty_dir = Path(tmp) / "empty"
        empty_dir.mkdir()
        assert find_latest_checkpoint(empty_dir) is None
        assert find_latest_checkpoint("/nonexistent/path/xyz") is None
        print("  PASSED")

        # ── Test 6: non-existent file raises FileNotFoundError ──────────
        print("\n[Test 6] missing file raises FileNotFoundError")
        try:
            load_checkpoint("/no/such/path.pth", model=model_b)
        except FileNotFoundError:
            print("  raised FileNotFoundError ✓")
            print("  PASSED")
        else:
            raise AssertionError("expected FileNotFoundError")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
