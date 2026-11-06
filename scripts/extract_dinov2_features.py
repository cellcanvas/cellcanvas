"""Extract frozen public DINOv2 ViT-B/14 slice features on an OME-Zarr tomogram grid.

Run in a torch + Zarr 2 environment. No private dependencies or model fine-tuning.
The image is grayscale replicated to RGB; patch features are bilinearly resampled
onto each original XY slice. This is not a native 3D model or a quality benchmark.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from numcodecs import Blosc


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tomogram", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new output to preserve prior features.")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This bounded workstation extraction requires the authorized CUDA device."
        )
    volume = np.asarray(
        zarr.open_group(str(args.tomogram), mode="r")["0"][:], dtype=np.float32
    )
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Expected finite ZYX volume")
    low, high = np.percentile(volume, [1, 99])
    if high <= low:
        raise ValueError("Input has no intensity range")
    model = torch.hub.load(
        str(args.model_source), "dinov2_vitb14", source="local", pretrained=False
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().cuda()
    depth, height, width = volume.shape
    features = zarr.open_array(
        str(args.out),
        mode="w",
        shape=(768, depth, height, width),
        chunks=(768, 1, min(64, height), min(64, width)),
        dtype="f2",
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    start = time.monotonic()
    stats = []
    with torch.inference_mode():
        for z in range(0, depth, 2):
            x = torch.from_numpy(volume[z : z + 2]).cuda()[:, None]
            x = ((x - low) / (high - low)).clamp(0, 1).repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = model.forward_features((x - mean) / std)["x_norm_patchtokens"]
            dense = tokens.float().transpose(1, 2).reshape(-1, 768, 32, 32)
            dense = F.interpolate(
                dense, size=(height, width), mode="bilinear", align_corners=False
            )
            out = dense.cpu().numpy().astype(np.float16)
            if not np.isfinite(out).all():
                raise RuntimeError(f"Nonfinite features at z={z}")
            features[:, z : z + len(out)] = out.transpose(1, 0, 2, 3)
            stats.append(float(out.astype(np.float32).std()))
            print(
                json.dumps(
                    {
                        "slices": min(z + 2, depth),
                        "of": depth,
                        "seconds": round(time.monotonic() - start, 2),
                    }
                ),
                flush=True,
            )
    if min(stats) <= 0:
        raise RuntimeError("Constant features")
    sources = {
        str(p.relative_to(args.model_source)): digest(p)
        for p in sorted(args.model_source.rglob("*.py"))
    }
    manifest = {
        "model": "DINOv2 ViT-B/14 (facebookresearch/dinov2, dinov2_vitb14)",
        "license": "Apache-2.0",
        "public": True,
        "source": str(args.tomogram.resolve()),
        "grid": {
            "shape": list(volume.shape),
            "axes": ["d", "z", "y", "x"],
            "resampling": "Each XY slice resized to 448x448; 32x32 patch grid bilinearly resampled to original XY grid, align_corners=False; Z unchanged",
        },
        "preprocessing": {
            "percentiles": [1, 99],
            "clip_values": [float(low), float(high)],
            "gray_to_rgb": "repeat",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": digest(args.checkpoint),
            "official_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        },
        "implementation": {
            "source_tree": str(args.model_source),
            "python_source_hashes": sources,
            "extractor_sha256": digest(__file__),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "validation": {
            "finite": True,
            "min_slice_feature_std": min(stats),
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "seconds": time.monotonic() - start,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "scope": "Frozen per-slice public features for interactive classification; no native-3D or segmentation accuracy claim.",
    }
    path = args.out.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2))
    print(str(path), flush=True)


if __name__ == "__main__":
    main()
