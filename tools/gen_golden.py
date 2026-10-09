#!/usr/bin/env python3
"""Regenerate tests/testdata golden fixtures with onnxruntime.

Usage (run from the repo root, needs: onnxruntime, numpy, Pillow):

    python tools/gen_golden.py --models tests/testdata/models

For each tests/testdata/{tag}_src.png (BGR uint8 image), writes:
    {tag}_src.bgr      raw source bytes (tight HWC BGR)
    {tag}_input.f32    padded NCHW float input (GC-Net input)
    {tag}_shadow.f32   GC-Net output (shading map)
    {tag}_dr_in.f32    6-channel DR-Net input [src; corrected]
    {tag}_output.f32   DR-Net output (padded size)
    {tag}_final.u8     cropped BGR uint8 result
    manifest.json      dims + shapes per tag

Padding: top/left replicate to multiples of 32 (matches GCDRNet infer.py).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
    from PIL import Image
except ImportError:
    sys.exit("pip install onnxruntime pillow numpy")

PAD = 32


def load_bgr(png: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
    return rgb[..., ::-1].copy()  # RGB -> BGR


def pad_top_left(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """x: [h,w,3] -> padded to multiple of PAD, replicate, pad on top/left."""
    h, w = x.shape[:2]
    ph, pw = (-h) % PAD, (-w) % PAD
    return np.pad(x, ((ph, 0), (pw, 0), (0, 0)), mode="edge"), ph, pw


def f32(path: Path, a: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(a, dtype=np.float32).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="dir with gcdrnet_{gcnet,drnet}_sim.onnx")
    ap.add_argument("--data", default="tests/testdata", help="fixture dir")
    a = ap.parse_args()
    md, dd = Path(a.models), Path(a.data)

    gc = ort.InferenceSession(str(md / "gcdrnet_gcnet_sim.onnx"),
                              providers=["CPUExecutionProvider"])
    dr = ort.InferenceSession(str(md / "gcdrnet_drnet_sim.onnx"),
                              providers=["CPUExecutionProvider"])

    manifest: dict = {}
    for png in sorted(dd.glob("*_src.png")):
        tag = png.stem[:-4]
        bgr = load_bgr(png)
        oh, ow = bgr.shape[:2]
        (dd / f"{tag}_src.bgr").write_bytes(bgr.tobytes())

        padded, ph, pw = pad_top_left(bgr)
        h, w = padded.shape[:2]
        x = padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # [1,3,h,w]
        f32(dd / f"{tag}_input.f32", x)

        shadow = gc.run(None, {gc.get_inputs()[0].name: x})[0]
        f32(dd / f"{tag}_shadow.f32", shadow)

        corrected = np.clip(x / np.maximum(shadow, 1e-6), 0.0, 1.0)
        dr_in = np.concatenate([x, corrected], axis=1)
        f32(dd / f"{tag}_dr_in.f32", dr_in)

        out = dr.run(None, {dr.get_inputs()[0].name: dr_in})[0]
        f32(dd / f"{tag}_output.f32", out)

        u8 = np.clip(out[0].transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        u8 = u8[ph:, pw:]  # crop top/left pad back to (oh, ow)
        (dd / f"{tag}_final.u8").write_bytes(u8.tobytes())

        manifest[tag] = {
            "orig_h": oh, "orig_w": ow, "ph": ph, "pw": pw, "h": h, "w": w,
            "shapes": {k: list(v.shape) for k, v in
                       [("input", x), ("shadow", shadow), ("dr_in", dr_in), ("output", out)]},
            "final_u8_shape": list(u8.shape),
        }
        print(f"{tag}: {ow}x{oh} -> {w}x{h} pad=({ph},{pw})")

    (dd / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
