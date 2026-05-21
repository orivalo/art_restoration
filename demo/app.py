"""Interactive Gradio demo — Art Restoration via Deep Learning Inpainting.

Compares three trained inpainting models side-by-side (PConv U-Net,
Vanilla U-Net, Gated U-Net) on user-supplied or example paintings.

Run::

    python demo/app.py
"""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.load_models import DISPLAY_NAMES, MODEL_NAMES, load_all_models  # noqa: E402
from src.data.mask_generator import MaskGenerator  # noqa: E402
from src.training.metrics import psnr as psnr_fn  # noqa: E402
from src.training.metrics import ssim as ssim_fn  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

print(f"[demo] Loading models on {DEVICE}...")
MODELS = load_all_models(DEVICE)
print(f"[demo] Loaded: {list(MODELS.keys())}")

MASKER = MaskGenerator(seed=12345)

BENCH_CSV = PROJECT_ROOT / "outputs" / "outputs" / "eval" / "tables" / "overall_metrics.csv"
BENCH_DF = pd.read_csv(BENCH_CSV) if BENCH_CSV.exists() else pd.DataFrame(
    columns=["model", "psnr", "ssim", "lpips"]
)

MODEL_META = {
    "gated_unet": {
        "display": "Gated U-Net",
        "subtitle": "DeepFillv2-style soft attention",
        "paper": "Yu et al., ICCV 2019",
        "params": "~ 50 M",
        "tag": "Best on art",
        "color": "#D97757",
        "desc": "Replaces every convolution with a gated convolution: "
                "each spatial position learns a continuous soft gate "
                "that decides how much the feature contributes. The "
                "gate is preserved end-to-end, unlike PConv's binary "
                "mask which saturates to all-ones after a few layers.",
    },
    "unet_baseline": {
        "display": "Vanilla U-Net",
        "subtitle": "Mask concatenated as 4th channel",
        "paper": "Ronneberger et al., MICCAI 2015",
        "params": "~ 26 M",
        "tag": "Strong baseline",
        "color": "#5B7C99",
        "desc": "The simplest possible mask-conditioning strategy: "
                "the binary mask is concatenated to the RGB image as a "
                "fourth input channel. Surprisingly competitive against "
                "more sophisticated mask-aware operators.",
    },
    "pconv_unet": {
        "display": "PConv U-Net",
        "subtitle": "Partial convolution, hard mask",
        "paper": "Liu et al., ECCV 2018",
        "params": "~ 26 M",
        "tag": "The classic",
        "color": "#8B7E66",
        "desc": "Replaces every convolution with a partial convolution: "
                "outputs are renormalised by the valid-area ratio, and "
                "an updated binary mask is propagated to the next "
                "layer. The most-cited irregular-mask inpainting "
                "architecture in the literature.",
    },
}


# ──────────────────────────────────────────────────────────────────────
# Image / mask preprocessing
# ──────────────────────────────────────────────────────────────────────

def _to_pil(arr: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def preprocess_image(image: np.ndarray | Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    img = _to_pil(image)
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.array(img)


def image_to_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    arr = rgb_u8.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().clamp(0, 1).squeeze(0).cpu().numpy()
    return (arr.transpose(1, 2, 0) * 255).round().astype(np.uint8)


def mask_to_tensor(mask_u8_or_bool: np.ndarray) -> torch.Tensor:
    if mask_u8_or_bool.ndim == 3:
        mask_u8_or_bool = mask_u8_or_bool[..., 0]
    holes = mask_u8_or_bool > 0
    valid = (~holes).astype(np.float32)
    return torch.from_numpy(valid).unsqueeze(0).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_one_model(model: torch.nn.Module,
                  image_t: torch.Tensor,
                  mask_t: torch.Tensor) -> torch.Tensor:
    image_t = image_t.to(DEVICE)
    mask_t = mask_t.to(DEVICE)
    image_corrupted = image_t * mask_t
    pred, _ = model(image_corrupted, mask_t)
    pred = pred.clamp(0, 1)
    composite = image_t * mask_t + pred * (1 - mask_t)
    return composite.cpu()


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    p = pred.to(DEVICE)
    g = gt.to(DEVICE)
    return {
        "psnr": float(psnr_fn(p, g).item()),
        "ssim": float(ssim_fn(p, g).item()),
    }


# ──────────────────────────────────────────────────────────────────────
# UI rendering helpers (pure HTML / CSS — fully under our control)
# ──────────────────────────────────────────────────────────────────────

def _benchmark_html() -> str:
    if BENCH_DF.empty:
        return ""
    df = BENCH_DF.copy()
    df["Model"] = df["model"].map(DISPLAY_NAMES).fillna(df["model"])
    df = df[["Model", "psnr", "ssim", "lpips"]]
    df = df.sort_values("psnr", ascending=False).reset_index(drop=True)
    rows = []
    medals = ["gold", "silver", "bronze"]
    for i, r in df.iterrows():
        medal_cls = medals[i] if i < 3 else ""
        winner_cls = "winner" if i == 0 else ""
        rows.append(f"""
            <tr class="ar-row {winner_cls}">
              <td class="ar-model"><span class="ar-medal {medal_cls}">{i+1}</span><span>{r['Model']}</span></td>
              <td class="ar-num">{r['psnr']:.2f}</td>
              <td class="ar-num">{r['ssim']:.3f}</td>
              <td class="ar-num">{r['lpips']:.3f}</td>
            </tr>""")
    return f"""
    <div class="ar-card">
      <span class="ar-eyebrow">Held-out test set</span>
      <h3 class="ar-h3">Benchmark on 3,977 paintings &times; 12 mask conditions</h3>
      <p class="ar-sub">Averaged over 47,724 (image, mask) pairs. Higher is better for PSNR / SSIM; lower is better for LPIPS.</p>
      <div class="ar-table-wrap">
        <table class="ar-table">
          <thead>
            <tr><th>Model</th><th class="ar-num">PSNR &uarr;</th><th class="ar-num">SSIM &uarr;</th><th class="ar-num">LPIPS &darr;</th></tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      <div class="ar-hint">
        <b>Metric guide.</b>
        <span class="ar-pill">PSNR / SSIM</span> measure pixel fidelity and structural similarity &mdash; higher = better.
        <span class="ar-pill">LPIPS</span> measures human-perceptual distance &mdash; lower = better.
      </div>
    </div>
    """


def _model_cards_html() -> str:
    cards = []
    for key in ["gated_unet", "unet_baseline", "pconv_unet"]:
        m = MODEL_META[key]
        cards.append(f"""
        <div class="ar-mcard" style="--accent: {m['color']}">
          <span class="ar-mtag">{m['tag']}</span>
          <h4 class="ar-mname">{m['display']}</h4>
          <p class="ar-msub">{m['subtitle']}</p>
          <p class="ar-mdesc">{m['desc']}</p>
          <div class="ar-mmeta">
            <span><b>Reference</b> {m['paper']}</span>
            <span><b>Parameters</b> {m['params']}</span>
          </div>
        </div>""")
    return f'<div class="ar-mcards">{"".join(cards)}</div>'


def _np_to_data_url(arr: np.ndarray) -> str:
    """Encode an RGB / grayscale numpy array as a base64 data URL (PNG)."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.shape[-1] == 4:
        pil = Image.fromarray(arr[..., :3], mode="RGB")
    else:
        pil = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _img_zoom_html(arr: np.ndarray | None, *,
                   placeholder_text: str = "Awaiting input...") -> str:
    """Render an output image with a Gradio-style zoom toolbar (+, -, reset, fullscreen)."""
    if arr is None:
        return (
            f'<div class="ar-img-zoom is-placeholder">'
            f'  <div class="ar-img-empty">{placeholder_text}</div>'
            f'</div>'
        )
    url = _np_to_data_url(arr)
    return (
        f'<div class="ar-img-zoom" data-zoom="1" data-panx="0" data-pany="0">'
        f'  <div class="ar-img-frame">'
        f'    <img class="ar-img-thumb" src="{url}" alt="" draggable="false" />'
        f'  </div>'
        f'  <div class="ar-img-tools">'
        f'    <button type="button" title="Zoom in"     onclick="window.arZoom(this, 1.25)">&plus;</button>'
        f'    <button type="button" title="Zoom out"    onclick="window.arZoom(this, 0.8)">&minus;</button>'
        f'    <button type="button" title="Reset zoom"  onclick="window.arZoomReset(this)">&#x21BB;</button>'
        f'    <span class="ar-zoom-pct" data-role="pct">100%</span>'
        f'    <button type="button" title="Fullscreen" class="ar-img-fs" onclick="window.arFullscreen(this)">&#x26F6;</button>'
        f'  </div>'
        f'  <div class="ar-img-overlay">'
        f'    <img src="{url}" alt="" />'
        f'    <button class="ar-img-close" type="button" aria-label="Close"'
        f'            onclick="event.stopPropagation();'
        f'            this.closest(\'.ar-img-zoom\').classList.remove(\'open\')">'
        f'      &times;'
        f'    </button>'
        f'    <div class="ar-img-hint">Click outside the image to close</div>'
        f'  </div>'
        f'</div>'
    )


def _labeled_img_zoom_html(arr: np.ndarray | None, label: str,
                           *, placeholder_text: str = "—") -> str:
    """Image with a clear label above + click-to-zoom."""
    return (
        f'<div class="ar-labeled-img-block">'
        f'  <div class="ar-labeled-img-lbl">{label}</div>'
        f'  {_img_zoom_html(arr, placeholder_text=placeholder_text)}'
        f'</div>'
    )


def _make_damaged_preview(image_rgb: np.ndarray, mask_t: torch.Tensor) -> np.ndarray:
    out = image_rgb.copy()
    m = mask_t.squeeze().numpy()
    holes = m < 0.5
    out[holes] = (out[holes] * 0.25 + np.array([255, 0, 255]) * 0.75).astype(np.uint8)
    return out


def _result_header(name: str, m: dict[str, float] | None, is_best: bool) -> str:
    meta = MODEL_META[name]
    rank = {"gated_unet": "1", "unet_baseline": "2", "pconv_unet": "3"}[name]
    if m is None:
        psnr_str, ssim_str = "—", "—"
        winner_cls = "pending"
        badge = ""
    else:
        psnr_str = f"{m['psnr']:.2f}"
        ssim_str = f"{m['ssim']:.3f}"
        winner_cls = "winner" if is_best else ""
        badge = '<span class="ar-rbadge">Best</span>' if is_best else ""
    return f"""
    <div class="ar-rhead {winner_cls}" style="--accent: {meta['color']}">
      <div class="ar-rhead-top">
        <span class="ar-rrank">{rank}</span>
        <div class="ar-rtitles">
          <div class="ar-rname">{meta['display']} {badge}</div>
          <div class="ar-rsub">{meta['subtitle']}</div>
        </div>
      </div>
      <div class="ar-rmetrics">
        <div class="ar-mp">
          <span class="ar-mp-lbl">PSNR &uarr;</span>
          <span class="ar-mp-val">{psnr_str}</span>
        </div>
        <div class="ar-mp">
          <span class="ar-mp-lbl">SSIM &uarr;</span>
          <span class="ar-mp-val">{ssim_str}</span>
        </div>
      </div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────
# Pipelines
# ──────────────────────────────────────────────────────────────────────

def _run_pipeline(rgb_full: np.ndarray, mask_u8: np.ndarray) -> Tuple:
    image_t = image_to_tensor(rgb_full)
    mask_t = mask_to_tensor(mask_u8)
    damaged_preview = _make_damaged_preview(rgb_full, mask_t)

    outputs: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, model in MODELS.items():
        outputs[name] = run_one_model(model, image_t, mask_t)
        metrics[name] = compute_metrics(outputs[name], image_t)

    best = max(metrics, key=lambda k: metrics[k]["psnr"])
    head_g = _result_header("gated_unet",    metrics["gated_unet"],    best == "gated_unet")
    head_v = _result_header("unet_baseline", metrics["unet_baseline"], best == "unet_baseline")
    head_p = _result_header("pconv_unet",    metrics["pconv_unet"],    best == "pconv_unet")

    img_g_html = _img_zoom_html(tensor_to_image(outputs["gated_unet"]))
    img_v_html = _img_zoom_html(tensor_to_image(outputs["unet_baseline"]))
    img_p_html = _img_zoom_html(tensor_to_image(outputs["pconv_unet"]))

    orig_html = _labeled_img_zoom_html(rgb_full,        "Original (256 &times; 256)")
    dmg_html  = _labeled_img_zoom_html(damaged_preview, "Damaged input (magenta = hole)")
    mask_html = _labeled_img_zoom_html(mask_u8,         "Generated mask (white = hole)")

    hole_pct = float((mask_t < 0.5).float().mean().item()) * 100
    status = (f"<div class='ar-status ok'><b>Restored.</b> "
              f"{hole_pct:.1f}% of the canvas was marked as damaged. "
              f"Top result on this input: <b>{MODEL_META[best]['display']}</b>.</div>")
    return (orig_html, dmg_html,
            head_g, img_g_html, head_v, img_v_html, head_p, img_p_html,
            mask_html, status)


def _placeholder_outputs(msg: str) -> Tuple:
    return (_labeled_img_zoom_html(None, "Original (256 &times; 256)"),
            _labeled_img_zoom_html(None, "Damaged input (magenta = hole)"),
            _result_header("gated_unet",    None, False), _img_zoom_html(None),
            _result_header("unet_baseline", None, False), _img_zoom_html(None),
            _result_header("pconv_unet",    None, False), _img_zoom_html(None),
            _labeled_img_zoom_html(None, "Generated mask (white = hole)"),
            f"<div class='ar-status warn'>{msg}</div>")


def run_inpaint_brush(editor_value: dict | None) -> Tuple:
    if editor_value is None:
        return _placeholder_outputs("Please upload an image and paint over the damaged area.")
    bg = editor_value.get("background", None)
    if bg is None:
        return _placeholder_outputs("Please upload an image first.")

    rgb_full = preprocess_image(bg)
    mask_u8 = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    if "composite" in editor_value and editor_value["composite"] is not None:
        comp = preprocess_image(editor_value["composite"])
        diff = np.any(np.abs(comp.astype(np.int16) - rgb_full.astype(np.int16)) > 8, axis=-1)
        mask_u8 = (diff.astype(np.uint8) * 255)

    if mask_u8.sum() == 0:
        return _placeholder_outputs("No painted area detected. Paint on the image with the brush, then click Restore.")

    return _run_pipeline(rgb_full, mask_u8)


def run_inpaint_synthetic(image: np.ndarray | None,
                          damage_type: str,
                          difficulty: str) -> Tuple:
    if image is None:
        return _placeholder_outputs("Please upload an image.")

    rgb_full = preprocess_image(image)
    fn_map = {
        "brush":      MASKER.random_brush_strokes,
        "crack":      MASKER.simulated_cracks,
        "paint_loss": MASKER.simulated_paint_loss,
        "stain":      MASKER.random_aging_stains,
    }
    density_map = {"light": 0.15, "medium": 0.30, "heavy": 0.50}
    mask_float = fn_map[damage_type](IMAGE_SIZE, IMAGE_SIZE, density_map[difficulty])
    mask_u8 = ((mask_float < 0.5).astype(np.uint8) * 255)
    return _run_pipeline(rgb_full, mask_u8)


# ──────────────────────────────────────────────────────────────────────
# Examples
# ──────────────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLE_PATHS: list[str] = []
if EXAMPLES_DIR.exists():
    for p in sorted(EXAMPLES_DIR.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            EXAMPLE_PATHS.append(str(p))


# ──────────────────────────────────────────────────────────────────────
# CSS — namespaced `ar-*` selectors only.  We do NOT override Gradio's
# internal class names.  Buttons are styled via the orange theme +
# unique elem_id targeting (the safest, most reliable approach).
# ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ───── Page-wide background ───── */
html, body, gradio-app {
    background: #FAF9F5 !important;
}
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 28px 32px 64px !important;
    background: #FAF9F5 !important;
}

/* ───── Restore button — target by unique id ───── */
#ar-restore-btn-1, #ar-restore-btn-2,
#ar-restore-btn-1 button, #ar-restore-btn-2 button {
    background: #D97757 !important;
    background-image: linear-gradient(135deg, #D97757 0%, #B85B3D 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #9A4A30 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 16px 22px !important;
    min-height: 54px !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(184, 91, 61, 0.30) !important;
    cursor: pointer !important;
    text-shadow: 0 1px 1px rgba(0,0,0,0.12) !important;
    margin-top: 14px !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}
#ar-restore-btn-1:hover, #ar-restore-btn-2:hover,
#ar-restore-btn-1 button:hover, #ar-restore-btn-2 button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(184, 91, 61, 0.40) !important;
}

/* ───── Hero ───── */
.ar-hero {
    background: linear-gradient(145deg, #FFFFFF 0%, #FAF6EE 100%);
    border: 1px solid #E6E2D6;
    border-radius: 20px;
    padding: 36px 44px;
    margin-bottom: 24px;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 30px rgba(0,0,0,0.03);
}
.ar-hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #D97757, #E5A98E);
}
.ar-eyebrow {
    display: inline-block;
    text-transform: uppercase; letter-spacing: 0.14em;
    font-size: 11px; font-weight: 700;
    color: #9A4A30; background: #FBEFE5;
    padding: 5px 11px; border-radius: 999px;
    margin-bottom: 14px;
}
.ar-hero h1 {
    font-size: 34px; font-weight: 800; margin: 0 0 12px 0;
    color: #1F1E1A; letter-spacing: -0.02em; line-height: 1.15;
}
.ar-hero p {
    font-size: 15.5px; color: #44423D; max-width: 820px;
    line-height: 1.65; margin: 0;
}
.ar-hl {
    color: #1F1E1A; font-weight: 700;
    background: linear-gradient(180deg, transparent 62%, #F4DACD 62%);
    padding: 0 3px;
}
.ar-stats { display: flex; gap: 28px; margin-top: 26px; flex-wrap: wrap; }
.ar-stat { border-left: 3px solid #D97757; padding-left: 13px; }
.ar-stat b { display: block; font-size: 20px; color: #1F1E1A; font-weight: 700; }
.ar-stat span {
    font-size: 11px; color: #6B6860;
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
}

/* ───── Cards ───── */
.ar-card, .ar-insight, .ar-pipeline {
    background: #FFFFFF;
    border: 1px solid #E6E2D6;
    border-radius: 16px;
    padding: 22px 26px;
    margin: 14px 0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.025);
}
.ar-insight {
    background: linear-gradient(135deg, #FFFFFF 0%, #FAF6EE 100%);
    border-left: 4px solid #D97757;
}
.ar-h3 {
    margin: 4px 0 6px 0; font-size: 20px; font-weight: 700;
    color: #1F1E1A; letter-spacing: -0.015em;
}
.ar-sub { margin: 4px 0 14px 0; color: #6B6860; font-size: 13.5px; }
.ar-insight p, .ar-pipeline p, .ar-pipeline ol {
    color: #44423D; font-size: 14.5px; line-height: 1.65; margin: 8px 0 0 0;
}
.ar-pipeline ol { padding-left: 22px; }
.ar-pipeline ol li { color: #44423D; margin: 4px 0; }
.ar-pipeline ol li b { color: #1F1E1A; }

/* ───── Benchmark table ───── */
.ar-table-wrap {
    border-radius: 12px; border: 1px solid #E6E2D6;
    overflow: hidden; margin: 14px 0;
}
.ar-table {
    width: 100%; border-collapse: collapse; font-size: 14px;
    background: #FFFFFF;
}
.ar-table thead th {
    background: #FAF9F5;
    color: #6B6860;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-size: 11.5px; font-weight: 700;
    padding: 12px 14px; text-align: left;
    border-bottom: 1px solid #E6E2D6;
}
.ar-table tbody td {
    padding: 13px 14px;
    color: #1F1E1A !important;
    border-bottom: 1px solid #F2F0EA;
}
.ar-table tbody td * { color: #1F1E1A !important; }
.ar-table tbody tr:last-child td { border-bottom: none; }
.ar-num { text-align: right; font-variant-numeric: tabular-nums; color: #1F1E1A !important; }
.ar-row.winner td {
    background: linear-gradient(90deg, #FBEFE5 0%, #FFFFFF 70%);
    font-weight: 700;
}
.ar-row.winner td,
.ar-row.winner td * { color: #1F1E1A !important; font-weight: 700; }
.ar-model {
    display: flex; align-items: center; gap: 10px;
    color: #1F1E1A !important; font-weight: 600;
}
.ar-model > span:not(.ar-medal) {
    color: #1F1E1A !important;
    font-weight: 700;
    font-size: 14px;
}
.ar-medal {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    font-size: 12px; font-weight: 800;
    background: #E6E2D6; color: #44423D;
    flex-shrink: 0;
}
.ar-medal.gold   { background: #F4DACD; color: #9A4A30; box-shadow: 0 0 0 2px #FBEFE5; }
.ar-medal.silver { background: #E0E4EA; color: #4A5563; }
.ar-medal.bronze { background: #E8DCC8; color: #6B5530; }
.ar-hint {
    font-size: 13px; color: #44423D;
    background: #F2F0EA; padding: 12px 16px;
    border-radius: 10px; margin-top: 14px;
    line-height: 1.6; border-left: 3px solid #C8A35E;
}
.ar-hint b { color: #1F1E1A; }
.ar-pill {
    background: #E6E2D6; padding: 1px 7px;
    border-radius: 5px; font-weight: 700; color: #1F1E1A;
}

/* ───── Model cards ───── */
.ar-mcards {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px;
    margin: 16px 0 28px 0;
}
.ar-mcard {
    background: #FFFFFF;
    border: 1px solid #E6E2D6;
    border-top: 4px solid var(--accent);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.02);
    transition: transform .2s ease, box-shadow .2s ease;
}
.ar-mcard:hover { transform: translateY(-3px); box-shadow: 0 10px 26px rgba(0,0,0,0.06); }
.ar-mtag {
    display: inline-block; font-size: 11px; font-weight: 700;
    color: #9A4A30; background: #FBEFE5;
    padding: 4px 10px; border-radius: 999px; margin-bottom: 12px;
}
.ar-mname { margin: 4px 0 2px 0; font-size: 19px; font-weight: 700; color: #1F1E1A; }
.ar-msub { margin: 0; font-size: 13px; color: #6B6860; font-weight: 500; }
.ar-mdesc { font-size: 13.5px; color: #44423D; line-height: 1.65; margin: 14px 0 16px 0; }
.ar-mmeta {
    display: flex; flex-direction: column; gap: 5px;
    border-top: 1px solid #F2F0EA; padding-top: 12px;
}
.ar-mmeta span { font-size: 12.5px; color: #6B6860; }
.ar-mmeta b { display: inline-block; min-width: 84px; color: #1F1E1A; font-weight: 700; }

/* ───── Section heading ───── */
.ar-section {
    margin: 36px 0 14px 0;
    font-size: 12.5px; text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 700; color: #44423D;
}

/* ───── Reconstructions header ───── */
.ar-rec-head {
    margin: 28px 0 12px 0;
    display: flex; align-items: baseline; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
}
.ar-rec-head h3 { margin: 0; font-size: 22px; font-weight: 700; color: #1F1E1A; }
.ar-rec-hint {
    font-size: 13px; color: #44423D;
    background: #F2F0EA; padding: 6px 12px;
    border-radius: 999px; border: 1px solid #E6E2D6;
}
.ar-rec-hint b { color: #B85B3D; font-weight: 700; }

/* ───── Per-result header (above each output image) ───── */
.ar-rhead {
    background: #FFFFFF;
    border: 1px solid #E6E2D6;
    border-top: 4px solid var(--accent);
    border-radius: 14px;
    padding: 14px 16px 12px 16px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.ar-rhead.winner {
    background: linear-gradient(135deg, #FBEFE5 0%, #FFFFFF 100%);
    border: 1px solid #D97757;
    border-top: 4px solid #D97757;
    box-shadow: 0 0 0 3px #F4DACD, 0 6px 18px rgba(217,119,87,0.18);
}
.ar-rhead.pending { opacity: 0.62; }
.ar-rhead-top { display: flex; align-items: center; gap: 12px; }
.ar-rrank {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 50%;
    font-size: 14px; font-weight: 800; line-height: 1;
    background: #FBEFE5; color: #9A4A30; flex-shrink: 0;
}
.ar-rhead.winner .ar-rrank { background: #D97757; color: #FFFFFF; }
.ar-rtitles { flex: 1; min-width: 0; }
.ar-rname {
    font-size: 16px; font-weight: 700; color: #1F1E1A;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.ar-rsub { font-size: 12.5px; color: #6B6860; margin-top: 3px; font-weight: 500; }
.ar-rbadge {
    background: #D97757; color: #FFFFFF;
    font-size: 10.5px; padding: 2px 9px; border-radius: 999px;
    letter-spacing: 0.06em; font-weight: 700; text-transform: uppercase;
}
.ar-rmetrics {
    display: flex; gap: 10px; margin-top: 12px;
    padding-top: 12px; border-top: 1px dashed #E6E2D6;
}
.ar-mp {
    flex: 1; background: #F2F0EA; border-radius: 10px;
    padding: 9px 10px; text-align: center;
}
.ar-rhead.winner .ar-mp { background: #FBEFE5; }
.ar-mp-lbl {
    display: block; font-size: 10.5px; color: #6B6860;
    text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 700; margin-bottom: 3px;
}
.ar-mp-val {
    display: block; font-size: 22px; font-weight: 800;
    color: #1F1E1A; letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums; line-height: 1.1;
}
.ar-rhead.winner .ar-mp-val { color: #9A4A30; }

/* ───── Status pills ───── */
.ar-status {
    padding: 12px 16px; border-radius: 12px;
    font-size: 14px; margin: 12px 0; font-weight: 500;
}
.ar-status.ok {
    background: #E8F2EA; color: #1F4A28;
    border: 1px solid #C4DECB;
}
.ar-status.ok b { color: #14391E; font-weight: 700; }
.ar-status.warn {
    background: #FCEFE2; color: #6B3A1A;
    border: 1px solid #F2D5BC;
}

/* ───── Footer ───── */
.ar-footer {
    margin-top: 44px; padding: 28px 32px;
    background: #FFFFFF;
    border: 1px solid #E6E2D6; border-radius: 16px;
    font-size: 14.5px; line-height: 1.7;
    color: #1F1E1A;
}
.ar-footer h4 {
    margin: 0 0 10px 0; font-size: 16px; font-weight: 700;
    color: #1F1E1A;
}
.ar-footer h4:not(:first-child) { margin-top: 22px; }
.ar-footer p { color: #44423D; margin: 4px 0; }
.ar-footer ul { padding-left: 22px; margin: 6px 0 0 0; }
.ar-footer ul li { color: #44423D; margin: 5px 0; }
.ar-footer em { color: #1F1E1A; font-style: italic; font-weight: 500; }

/* ───── Output image with zoom toolbar ───── */
.ar-img-zoom {
    position: relative;
    width: 100%;
    border: 1px solid #E6E2D6;
    border-radius: 12px;
    background: #FFFFFF;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    aspect-ratio: 1 / 1;
    overflow: hidden;
}
.ar-img-zoom .ar-img-frame {
    position: relative;
    width: 100%; height: 100%;
    overflow: hidden;
    border-radius: 12px;
}
.ar-img-zoom .ar-img-thumb {
    display: block;
    width: 100%; height: 100%;
    object-fit: cover;
    transform-origin: center center;
    transition: transform .15s ease-out;
    user-select: none;
    -webkit-user-drag: none;
}
.ar-img-zoom.is-zoomed .ar-img-thumb { cursor: grab; transition: none; }
.ar-img-zoom.is-dragging .ar-img-thumb { cursor: grabbing; transition: none; }

/* Toolbar (top-right of each image) */
.ar-img-zoom .ar-img-tools {
    position: absolute;
    top: 8px; right: 8px;
    z-index: 5;
    display: flex; align-items: center; gap: 2px;
    background: rgba(255,255,255,0.96);
    border: 1px solid #E6E2D6;
    border-radius: 10px;
    padding: 3px 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    backdrop-filter: blur(6px);
}
.ar-img-zoom .ar-img-tools button {
    width: 28px; height: 28px;
    background: transparent;
    border: none;
    border-radius: 6px;
    font-size: 15px; font-weight: 700;
    color: #44423D;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    padding: 0; line-height: 1;
    transition: background .12s ease, color .12s ease;
}
.ar-img-zoom .ar-img-tools button:hover {
    background: #FBEFE5; color: #B85B3D;
}
.ar-img-zoom .ar-img-tools button:active {
    background: #F4DACD;
}
.ar-img-zoom .ar-img-fs { font-size: 13px; }
.ar-zoom-pct {
    font-size: 11px; font-weight: 700;
    color: #6B6860;
    min-width: 36px;
    padding: 0 4px;
    text-align: center;
    font-variant-numeric: tabular-nums;
    border-left: 1px solid #E6E2D6;
    border-right: 1px solid #E6E2D6;
    margin: 0 2px;
    line-height: 22px;
}

.ar-img-zoom.is-placeholder {
    background: #FAF9F5;
    border-style: dashed;
    display: flex; align-items: center; justify-content: center;
    box-shadow: none;
}
.ar-img-empty {
    color: #8C8879; font-size: 13px; font-style: italic;
}

/* ───── Lightbox overlay (full-screen view) ───── */
.ar-img-zoom .ar-img-overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(20, 19, 17, 0.94);
    z-index: 99999;
    align-items: center; justify-content: center;
    padding: 60px 40px;
    cursor: zoom-out;
    animation: ar-fadein 0.15s ease;
}
.ar-img-zoom.open .ar-img-overlay { display: flex; }
.ar-img-zoom .ar-img-overlay img {
    max-width: 92vw;
    max-height: 88vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.7);
    background: #FFFFFF;
}
.ar-img-zoom .ar-img-close {
    position: fixed;
    top: 22px; right: 28px;
    width: 48px; height: 48px;
    background: rgba(255,255,255,0.12);
    color: #FFFFFF;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 50%;
    font-size: 30px; line-height: 1;
    cursor: pointer;
    transition: background .15s ease;
    display: flex; align-items: center; justify-content: center;
    padding: 0;
}
.ar-img-zoom .ar-img-close:hover { background: rgba(255,255,255,0.25); }
.ar-img-zoom .ar-img-hint {
    position: fixed;
    bottom: 26px; left: 0; right: 0;
    text-align: center;
    color: rgba(255,255,255,0.65);
    font-size: 12.5px;
    letter-spacing: 0.04em;
    pointer-events: none;
}
@keyframes ar-fadein {
    from { opacity: 0; } to { opacity: 1; }
}

/* ───── Labelled image block (orig / damaged / mask) ───── */
.ar-labeled-img-block { margin-bottom: 12px; }
.ar-labeled-img-lbl {
    font-size: 13px; font-weight: 700;
    color: #1F1E1A; margin-bottom: 6px;
    letter-spacing: -0.01em;
}

/* ───── Responsive ───── */
@media (max-width: 1100px) {
    .ar-mcards { grid-template-columns: 1fr; }
}
"""


# ──────────────────────────────────────────────────────────────────────
# Theme — primary coral, warm neutrals
# ──────────────────────────────────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#FBEFE5", c100="#F4DACD", c200="#EDC3AE", c300="#E5A98E",
        c400="#DD8E6E", c500="#D97757", c600="#B85B3D", c700="#9A4A30",
        c800="#7C3B26", c900="#5E2C1C", c950="#3E1D12",
    ),
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#FAF9F5",
    body_text_color="#1F1E1A",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F2F0EA",
    border_color_primary="#E6E2D6",
    block_background_fill="#FFFFFF",
    block_border_color="#E6E2D6",
    block_radius="16px",
    button_primary_background_fill="#D97757",
    button_primary_background_fill_hover="#B85B3D",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_hover="#FFFFFF",
)


# ──────────────────────────────────────────────────────────────────────
# Static HTML blocks
# ──────────────────────────────────────────────────────────────────────

HERO_HTML = f"""
<div class="ar-hero">
  <span class="ar-eyebrow">Bachelor's Diploma &middot; Art Restoration AI</span>
  <h1>Restoring damaged paintings with deep learning</h1>
  <p>This demo runs three trained neural networks side-by-side &mdash;
  <span class="ar-hl">PConv U-Net</span>, <span class="ar-hl">Vanilla U-Net</span> and <span class="ar-hl">Gated U-Net</span>
  &mdash; on any painting you upload. Paint over the damaged area or pick a synthetic damage preset,
  and compare the three reconstructions in real time against per-image quality metrics.</p>
  <div class="ar-stats">
    <div class="ar-stat"><b>39,213</b><span>WikiArt paintings</span></div>
    <div class="ar-stat"><b>5 styles</b><span>Renaissance &rarr; Realism</span></div>
    <div class="ar-stat"><b>3 models</b><span>Identical training recipe</span></div>
    <div class="ar-stat"><b>47,724</b><span>Test pairs evaluated</span></div>
    <div class="ar-stat"><b>{DEVICE.type.upper()}</b><span>Inference device</span></div>
  </div>
</div>
"""

INSIGHT_HTML = """
<div class="ar-insight">
  <span class="ar-eyebrow">Key finding</span>
  <h3 class="ar-h3">The most-cited inpainting architecture is not the best on art.</h3>
  <p>Across <span class="ar-hl">all difficulty levels</span> and <span class="ar-hl">all damage types</span>,
  the modern <span class="ar-hl">Gated U-Net</span> wins, and even a plain <span class="ar-hl">Vanilla U-Net</span>
  with the mask concatenated as a fourth input channel beats <span class="ar-hl">PConv U-Net</span>
  &mdash; the most-cited irregular-mask architecture in the literature &mdash; by ~1.7 dB PSNR.
  The result holds under a paired Wilcoxon signed-rank test with Bonferroni correction.</p>
</div>
"""

PIPELINE_HTML = """
<div class="ar-pipeline">
  <span class="ar-eyebrow">How it works</span>
  <h3 class="ar-h3">Inference pipeline</h3>
  <ol>
    <li><b>Preprocess</b> &mdash; aspect-preserving resize + center-crop to 256 &times; 256.</li>
    <li><b>Build mask</b> &mdash; binary mask where 1 = valid, 0 = hole.</li>
    <li><b>Inference</b> &mdash; run all three networks in parallel.</li>
    <li><b>Composite</b> &mdash; hole pixels from network, valid from input.</li>
    <li><b>Score</b> &mdash; per-image PSNR &amp; SSIM against the original.</li>
  </ol>
</div>
"""

FOOTER_HTML = """
<div class="ar-footer">
  <h4>About this project</h4>
  <p>Bachelor's diploma project &mdash; <em>Restoration of Historic Artworks Using Deep Learning-Based
  Digital Inpainting</em>. All three networks were trained from scratch on the WikiArt corpus
  (Renaissance, Baroque, Impressionism, Post-Impressionism, Realism) with an identical recipe:
  Adam (lr 2 &times; 10<sup>-4</sup>), one-epoch warmup followed by cosine decay, batch size 4,
  12 epochs at 256 &times; 256, mixed-precision training. The loss is the five-term composite of
  Liu et al. 2018: L1 valid + L1 hole + VGG perceptual + Gram-matrix style + total-variation,
  with weights 1 / 6 / 0.05 / 50 / 0.1.</p>
  <h4>References</h4>
  <ul>
    <li>Liu et al., <em>Image Inpainting for Irregular Holes Using Partial Convolutions</em>, ECCV 2018.</li>
    <li>Yu et al., <em>Free-Form Image Inpainting with Gated Convolution</em>, ICCV 2019.</li>
    <li>Ronneberger et al., <em>U-Net: Convolutional Networks for Biomedical Image Segmentation</em>, MICCAI 2015.</li>
    <li>Zhang et al., <em>The Unreasonable Effectiveness of Deep Features as a Perceptual Metric</em>, CVPR 2018.</li>
  </ul>
</div>
"""


# ──────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────

_ZOOM_JS = """
() => {
    function applyTransform(wrap) {
        const img = wrap.querySelector('.ar-img-thumb');
        if (!img) return;
        const z = parseFloat(wrap.dataset.zoom || '1');
        const x = parseFloat(wrap.dataset.panx || '0');
        const y = parseFloat(wrap.dataset.pany || '0');
        img.style.transform = 'translate(' + x + 'px,' + y + 'px) scale(' + z + ')';
        const pct = wrap.querySelector('[data-role=pct]');
        if (pct) pct.textContent = Math.round(z * 100) + '%';
        wrap.classList.toggle('is-zoomed', z > 1.01);
    }
    window.arZoom = function(btn, factor) {
        const wrap = btn.closest('.ar-img-zoom');
        if (!wrap) return;
        let z = parseFloat(wrap.dataset.zoom || '1');
        z = Math.max(0.5, Math.min(8, z * factor));
        wrap.dataset.zoom = z;
        if (z <= 1.01) { wrap.dataset.panx = '0'; wrap.dataset.pany = '0'; }
        applyTransform(wrap);
    };
    window.arZoomReset = function(btn) {
        const wrap = btn.closest('.ar-img-zoom');
        if (!wrap) return;
        wrap.dataset.zoom = '1';
        wrap.dataset.panx = '0';
        wrap.dataset.pany = '0';
        applyTransform(wrap);
    };
    window.arFullscreen = function(btn) {
        const wrap = btn.closest('.ar-img-zoom');
        if (wrap) wrap.classList.toggle('open');
    };
    // Drag-to-pan when image is zoomed in
    if (!window.__arDragInit) {
        window.__arDragInit = true;
        let active = null, startX = 0, startY = 0, px0 = 0, py0 = 0;
        document.addEventListener('mousedown', function(e) {
            const img = e.target.closest('.ar-img-thumb');
            if (!img) return;
            const wrap = img.closest('.ar-img-zoom');
            if (!wrap || parseFloat(wrap.dataset.zoom || '1') <= 1.01) return;
            if (wrap.classList.contains('open')) return;
            e.preventDefault();
            active = wrap;
            startX = e.clientX; startY = e.clientY;
            px0 = parseFloat(wrap.dataset.panx || '0');
            py0 = parseFloat(wrap.dataset.pany || '0');
            wrap.classList.add('is-dragging');
        });
        document.addEventListener('mousemove', function(e) {
            if (!active) return;
            active.dataset.panx = (px0 + e.clientX - startX).toString();
            active.dataset.pany = (py0 + e.clientY - startY).toString();
            applyTransform(active);
        });
        document.addEventListener('mouseup', function() {
            if (active) { active.classList.remove('is-dragging'); active = null; }
        });
        // ESC closes any open fullscreen
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                document.querySelectorAll('.ar-img-zoom.open').forEach(w => w.classList.remove('open'));
            }
        });
    }
}
"""


with gr.Blocks(title="Art Restoration Demo", theme=theme, css=CUSTOM_CSS) as demo:

    demo.load(fn=None, inputs=None, outputs=None, js=_ZOOM_JS)

    gr.HTML(HERO_HTML)
    gr.HTML(INSIGHT_HTML)

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(_benchmark_html())
        with gr.Column(scale=2):
            gr.HTML(PIPELINE_HTML)

    gr.HTML('<div class="ar-section">The three competing architectures</div>')
    gr.HTML(_model_cards_html())

    gr.HTML('<div class="ar-section">Try it yourself</div>')

    with gr.Tabs():

        # ── Tab 1: brush-paint a custom damage mask ──────────────────
        with gr.Tab("Paint custom damage"):
            with gr.Row():
                with gr.Column(scale=1):
                    editor = gr.ImageEditor(
                        label="Upload a painting, then paint over the damage",
                        type="numpy",
                        brush=gr.Brush(default_size=2,
                                       colors=["#D97757", "#1F1E1A", "#FFFFFF"]),
                        height=440,
                    )
                    btn_brush = gr.Button("Restore with all 3 models",
                                          variant="primary",
                                          elem_id="ar-restore-btn-1")
                    status_b = gr.HTML()

                with gr.Column(scale=1):
                    orig_b = gr.HTML(_labeled_img_zoom_html(None, "Original (256 &times; 256)"))
                    damaged_b = gr.HTML(_labeled_img_zoom_html(None, "Damaged input (magenta = hole)"))

            gr.HTML(
                '<div class="ar-rec-head">'
                '<h3>Reconstructions</h3>'
                '<span class="ar-rec-hint">All three networks ran on the <b>same</b> '
                '(image, mask) pair &mdash; comparison is fair.</span>'
                '</div>'
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    head_b_g = gr.HTML(_result_header("gated_unet", None, False))
                    img_b_g = gr.HTML(_img_zoom_html(None))
                with gr.Column(scale=1):
                    head_b_v = gr.HTML(_result_header("unet_baseline", None, False))
                    img_b_v = gr.HTML(_img_zoom_html(None))
                with gr.Column(scale=1):
                    head_b_p = gr.HTML(_result_header("pconv_unet", None, False))
                    img_b_p = gr.HTML(_img_zoom_html(None))

            mask_b = gr.HTML(_labeled_img_zoom_html(None, "Generated mask (white = hole)"))

            if EXAMPLE_PATHS:
                gr.Examples(
                    examples=[[p] for p in EXAMPLE_PATHS],
                    inputs=[editor],
                    label="Example paintings (click to load)",
                )

            btn_brush.click(
                run_inpaint_brush,
                inputs=[editor],
                outputs=[orig_b, damaged_b,
                         head_b_g, img_b_g,
                         head_b_v, img_b_v,
                         head_b_p, img_b_p,
                         mask_b, status_b],
            )

        # ── Tab 2: synthetic preset damage ───────────────────────────
        with gr.Tab("Synthetic damage preset"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_s = gr.Image(label="Upload a painting", type="numpy", height=280)
                    with gr.Row():
                        damage_s = gr.Radio(
                            choices=[("Brush strokes", "brush"),
                                     ("Cracks (craquelure)", "crack"),
                                     ("Paint loss (flaking)", "paint_loss"),
                                     ("Aging stains", "stain")],
                            value="crack", label="Damage type",
                        )
                        difficulty_s = gr.Radio(
                            choices=[("Light", "light"),
                                     ("Medium", "medium"),
                                     ("Heavy", "heavy")],
                            value="medium", label="Severity",
                        )
                    btn_s = gr.Button("Generate damage & restore",
                                      variant="primary",
                                      elem_id="ar-restore-btn-2")
                    status_s = gr.HTML()
                with gr.Column(scale=1):
                    orig_s = gr.HTML(_labeled_img_zoom_html(None, "Original (256 &times; 256)"))
                    damaged_s = gr.HTML(_labeled_img_zoom_html(None, "Damaged input (magenta = hole)"))

            gr.HTML(
                '<div class="ar-rec-head">'
                '<h3>Reconstructions</h3>'
                '<span class="ar-rec-hint">All three networks ran on the <b>same</b> '
                '(image, mask) pair &mdash; comparison is fair.</span>'
                '</div>'
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    head_s_g = gr.HTML(_result_header("gated_unet", None, False))
                    img_s_g = gr.HTML(_img_zoom_html(None))
                with gr.Column(scale=1):
                    head_s_v = gr.HTML(_result_header("unet_baseline", None, False))
                    img_s_v = gr.HTML(_img_zoom_html(None))
                with gr.Column(scale=1):
                    head_s_p = gr.HTML(_result_header("pconv_unet", None, False))
                    img_s_p = gr.HTML(_img_zoom_html(None))

            mask_s = gr.HTML(_labeled_img_zoom_html(None, "Generated mask (white = hole)"))

            if EXAMPLE_PATHS:
                gr.Examples(
                    examples=[[p, "crack", "medium"] for p in EXAMPLE_PATHS],
                    inputs=[img_s, damage_s, difficulty_s],
                    label="Example paintings (click to load)",
                )

            btn_s.click(
                run_inpaint_synthetic,
                inputs=[img_s, damage_s, difficulty_s],
                outputs=[orig_s, damaged_s,
                         head_s_g, img_s_g,
                         head_s_v, img_s_v,
                         head_s_p, img_s_p,
                         mask_s, status_s],
            )

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )
