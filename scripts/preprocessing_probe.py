from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image


CSV = Path(r"C:\SlickTrace\outputs\train.csv")
OUT = Path(r"C:\SlickTrace\outputs\preprocessing_probe")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# One sample from each class
samples = [
    df[df["label"] == "oil"].iloc[0],
    df[df["label"] == "lookalike"].iloc[0],
    df[df["label"] == "no_oil"].iloc[0],
]

SIZES = [256, 384, 512]


def load_tiff(path):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile.copy()

    return image, profile


def normalize_per_band(image):
    result = np.zeros_like(image, dtype=np.float32)

    for b in range(image.shape[0]):
        band = image[b]
        lo = np.percentile(band, 1)
        hi = np.percentile(band, 99)

        if hi <= lo:
            result[b] = 0.0
        else:
            result[b] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)

    return result


print("=" * 80)
print("SLICKTRACE AI — PREPROCESSING PROBE")
print("=" * 80)

for row in samples:
    print()
    print("-" * 80)
    print("CLASS:", row["label"])
    print("IMAGE:", row["image_path"])
    print("MASK :", row["mask_path"])

    image, profile = load_tiff(row["image_path"])
    mask, _ = load_tiff(row["mask_path"])

    print("Original image shape:", image.shape)
    print("Original image dtype:", image.dtype)
    print("Original mask shape :", mask.shape)
    print("Original mask dtype :", mask.dtype)

    normalized = normalize_per_band(image)

    print(
        "Normalized range:",
        float(normalized.min()),
        "to",
        float(normalized.max()),
    )

    # Convert CHW -> HWC for resizing/export
    rgb_like = np.transpose(normalized, (1, 2, 0))

    for size in SIZES:
        resized = Image.fromarray(
            np.uint8(rgb_like * 255.0)
        ).resize((size, size), Image.Resampling.BILINEAR)

        mask_2d = mask[0]
        mask_resized = Image.fromarray(
            np.uint8(mask_2d)
        ).resize((size, size), Image.Resampling.NEAREST)

        resized.save(
            OUT / f"{row['label']}_{size}.png"
        )

        mask_resized.save(
            OUT / f"{row['label']}_{size}_mask.png"
        )

        print(
            f"  {size}x{size}: "
            f"image={resized.size}, "
            f"mask={mask_resized.size}"
        )

print()
print("=" * 80)
print("PROBE COMPLETE")
print("Output:", OUT)
print("=" * 80)