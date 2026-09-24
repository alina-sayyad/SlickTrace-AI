import csv
from pathlib import Path

DATA_ROOT = Path(r"A:\SIH26143_DATA\OFFICIAL_DATASET")
OUT = Path(r"C:\SlickTrace\outputs\dataset_manifest.csv")

OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []


def add_pairs(image_dir, mask_dir, label):
    images = {p.stem: p for p in image_dir.glob("*.tif")}
    masks = {p.stem: p for p in mask_dir.glob("*.tif")}

    if set(images) != set(masks):
        missing_images = sorted(set(masks) - set(images))
        missing_masks = sorted(set(images) - set(masks))
        raise RuntimeError(
            f"{label}: image/mask mismatch. "
            f"Missing images={missing_images[:5]}, "
            f"missing masks={missing_masks[:5]}"
        )

    for key in sorted(images):
        rows.append(
            {
                "id": key,
                "label": label,
                "image_path": str(images[key]),
                "mask_path": str(masks[key]),
            }
        )


add_pairs(
    DATA_ROOT / "PART_I_OIL" / "Oil",
    DATA_ROOT / "PART_I_OIL" / "Mask_oil",
    "oil",
)

add_pairs(
    DATA_ROOT / "PART_II_LOOKALIKE_NOOIL" / "Lookalike",
    DATA_ROOT / "PART_II_LOOKALIKE_NOOIL" / "Mask_lookalike",
    "lookalike",
)

add_pairs(
    DATA_ROOT / "PART_II_LOOKALIKE_NOOIL" / "No_oil",
    DATA_ROOT / "PART_II_LOOKALIKE_NOOIL" / "Mask_no_oil",
    "no_oil",
)


with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "label", "image_path", "mask_path"],
    )
    writer.writeheader()
    writer.writerows(rows)


print(f"Manifest: {OUT}")
print(f"Total samples: {len(rows)}")

for label in ["oil", "lookalike", "no_oil"]:
    print(label, sum(r["label"] == label for r in rows))