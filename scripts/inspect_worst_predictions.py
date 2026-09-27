import os
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt

RESULTS = r"C:\SlickTrace\outputs\test_predictions\prediction_results.csv"
OUTPUT = r"C:\SlickTrace\outputs\test_predictions\worst_cases.png"

df = pd.read_csv(RESULTS).nsmallest(10, "dice")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for ax, (_, row) in zip(axes.flat, df.iterrows()):
    with rasterio.open(row["image_path"]) as src:
        image = src.read()

    with rasterio.open(row["prediction_path"]) as src:
        prediction = src.read(1)

    # Display VV band after robust normalization
    band = image[0].astype(np.float32)
    lo, hi = np.percentile(band, [1, 99])
    band = np.clip((band - lo) / (hi - lo), 0, 1)

    ax.imshow(band, cmap="gray")
    ax.imshow(
        np.ma.masked_where(prediction == 0, prediction),
        cmap="Reds",
        alpha=0.45,
    )

    name = os.path.basename(row["image_path"])
    category = row["image_path"].split("\\")[-2]

    ax.set_title(
        f"{category}/{name}\n"
        f"Dice={row['dice']:.2f} | Pixels={row['predicted_pixels']}"
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", OUTPUT)