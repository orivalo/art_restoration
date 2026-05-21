# Art Restoration — Interactive Gradio Demo

Compare three deep-learning inpainting models on real paintings.  Upload
any image, paint over the damaged area (or pick a synthetic damage
preset), and the demo runs PConv U-Net, Vanilla U-Net and Gated U-Net
side-by-side, reporting per-image PSNR and SSIM.

## Run locally

```bash
# from the project root
pip install -r demo/requirements.txt
python demo/app.py
```

The app opens at <http://127.0.0.1:7860>.

## Expected checkpoint layout

The demo expects three `best.pth` files under:

```
outputs/checkpoints/pconv_unet/best.pth
outputs/checkpoints/unet_baseline/best.pth
outputs/checkpoints/gated_unet/best.pth
```

If a different layout is needed, edit the `CHECKPOINTS` dict at the top
of `demo/load_models.py`.

## Examples

Put 4–6 sample paintings in `demo/examples/` (`.jpg` or `.png`).  They
appear as one-click examples under each tab.

## Notes

* All images are resized + centre-cropped to 256×256 before inference.
* The brush colour does not matter — any painted pixel is treated as
  a hole.
* Outputs are composited: hole pixels come from the network's
  prediction, valid pixels are passed through from the input.
* The "Benchmark numbers" panel reads from
  `outputs/outputs/eval/tables/overall_metrics.csv`.

## Stretch: HuggingFace Spaces

Create a new Space with the SDK "Gradio", upload the contents of `demo/`
plus the trained checkpoints (use Git LFS for the `.pth` files), and
the app will run publicly.  Update the checkpoint paths in
`demo/load_models.py` to point at the Space's local file layout.
