from pathlib import Path
import shutil
import pandas as pd


PROJECT_ROOT = Path(r"C:\SlickTrace")

VAL_CSV = PROJECT_ROOT / "outputs" / "val.csv"

SUBSET_ROOT = Path(r"C:\SlickTrace\VALIDATION_SUBSET")

MANIFEST_OUT = PROJECT_ROOT / "outputs" / "val_subset_manifest.csv"


def main():
    print("=" * 70)
    print("SLICKTRACE AI - CREATE PHYSICAL VALIDATION SUBSET")
    print("=" * 70)

    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Validation CSV not found: {VAL_CSV}")

    df = pd.read_csv(VAL_CSV)

    required_columns = {
        "id",
        "label",
        "image_path",
        "mask_path",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not (df["split"] == "val").all():
        raise ValueError(
            "val.csv contains rows whose split is not 'val'."
        )

    print(f"Validation samples in CSV: {len(df)}")

    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())

    print("\nSubset destination:")
    print(SUBSET_ROOT)

    SUBSET_ROOT.mkdir(parents=True, exist_ok=True)

    output_rows = []

    for index, row in df.iterrows():

        label = str(row["label"]).strip()

        if label not in {"oil", "lookalike", "no_oil"}:
            raise ValueError(
                f"Unexpected label '{label}' at CSV row {index}"
            )

        source_image = Path(str(row["image_path"]))
        source_mask = Path(str(row["mask_path"]))

        if not source_image.exists():
            raise FileNotFoundError(
                f"Image does not exist:\n{source_image}"
            )

        if not source_mask.exists():
            raise FileNotFoundError(
                f"Mask does not exist:\n{source_mask}"
            )

        image_dir = SUBSET_ROOT / label / "images"
        mask_dir = SUBSET_ROOT / label / "masks"

        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        destination_image = image_dir / source_image.name
        destination_mask = mask_dir / source_mask.name

        if not destination_image.exists():
            shutil.copy2(source_image, destination_image)

        if not destination_mask.exists():
            shutil.copy2(source_mask, destination_mask)

        output_rows.append(
            {
                "id": row["id"],
                "label": label,
                "image_path": str(destination_image),
                "mask_path": str(destination_mask),
                "split": "val",
            }
        )

        completed = index + 1

        if (
            completed == 1
            or completed % 25 == 0
            or completed == len(df)
        ):
            print(
                f"[{completed:4d}/{len(df)}] "
                f"copied: {label}/{source_image.name}"
            )

    manifest = pd.DataFrame(output_rows)

    manifest.to_csv(MANIFEST_OUT, index=False)

    print("\n" + "=" * 70)
    print("VALIDATION SUBSET CREATION COMPLETE")
    print("=" * 70)

    print(f"Samples: {len(manifest)}")
    print(f"Images:  {len(manifest)}")
    print(f"Masks:   {len(manifest)}")

    print("\nClass distribution:")
    print(manifest["label"].value_counts().to_string())

    print("\nSubset root:")
    print(SUBSET_ROOT)

    print("\nManifest:")
    print(MANIFEST_OUT)

    print("\nVerification:")

    missing_images = []
    missing_masks = []

    for _, row in manifest.iterrows():

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        if not image_path.exists():
            missing_images.append(str(image_path))

        if not mask_path.exists():
            missing_masks.append(str(mask_path))

    print(f"Missing images: {len(missing_images)}")
    print(f"Missing masks:  {len(missing_masks)}")

    if missing_images or missing_masks:
        raise RuntimeError(
            "Validation subset verification failed."
        )

    print("\nSUCCESS: Physical validation subset is ready.")


if __name__ == "__main__":
    main()