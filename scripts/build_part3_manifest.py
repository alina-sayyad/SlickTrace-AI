import os
import pandas as pd

ROOT = r"A:\SIH26143_DATA\OFFICIAL_DATASET\PART_III_TEST"
rows = []

classes = {
    "Oil": "Oil",
    "No oil": "No oil",
    "Lookalike": "Lookalike",
}

for label, folder in classes.items():
    image_dir = os.path.join(ROOT, "Images", folder)
    mask_dir = os.path.join(ROOT, "Mask", folder)

    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith(".tif"):
            continue

        image_path = os.path.join(image_dir, filename)

        stem = os.path.splitext(filename)[0]
        mask_filename = stem + "_segmentation.tif"
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            print("Missing mask:", mask_path)
            continue

        rows.append({
            "id": stem,
            "label": label.lower().replace(" ", "_"),
            "image_path": image_path,
            "mask_path": mask_path,
            "split": "part_iii",
        })

df = pd.DataFrame(rows)

output = r"C:\SlickTrace\outputs\part_iii.csv"
df.to_csv(output, index=False)

print("=" * 70)
print("PART III MANIFEST")
print("=" * 70)
print("Created:", output)
print("Total:", len(df))

if len(df) > 0:
    print()
    print(df["label"].value_counts().to_string())
    print()
    print(df.head(3).to_string(index=False))
