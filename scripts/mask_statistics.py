from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


CSV = Path(r"C:\SlickTrace\outputs\train.csv")

df = pd.read_csv(CSV)

oil = df[df["label"] == "oil"]

print("=" * 80)
print("SLICKTRACE AI — OIL MASK STATISTICS")
print("=" * 80)
print("Oil training samples:", len(oil))
print()

positive_ratios = []

for i, row in enumerate(oil.itertuples(index=False), start=1):
    with rasterio.open(row.mask_path) as src:
        mask = src.read(1)

    positive = mask > 0
    ratio = positive.mean() * 100
    positive_ratios.append(ratio)

    if i % 100 == 0:
        print(f"Processed {i}/{len(oil)}")

positive_ratios = np.array(positive_ratios)

print()
print("-" * 80)
print("MASK COVERAGE")
print("-" * 80)

print(f"Minimum oil area   : {positive_ratios.min():.4f}%")
print(f"Maximum oil area   : {positive_ratios.max():.4f}%")
print(f"Mean oil area      : {positive_ratios.mean():.4f}%")
print(f"Median oil area    : {np.median(positive_ratios):.4f}%")
print(f"25th percentile    : {np.percentile(positive_ratios, 25):.4f}%")
print(f"75th percentile    : {np.percentile(positive_ratios, 75):.4f}%")

print()
print("Images with <0.1% oil :", np.sum(positive_ratios < 0.1))
print("Images with <0.5% oil :", np.sum(positive_ratios < 0.5))
print("Images with <1.0% oil :", np.sum(positive_ratios < 1.0))
print("Images with >10% oil  :", np.sum(positive_ratios > 10.0))

print()
print("MASK STATISTICS COMPLETE")