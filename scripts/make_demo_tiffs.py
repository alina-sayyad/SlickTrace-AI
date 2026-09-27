from pathlib import Path
import shutil
import pandas as pd

# ============================================================
# SLICKTRACE AI — JUDGE DEMO TIFF GALLERY
# Selects the best-performing REAL OIL examples from the
# completed prediction run.
# ============================================================

PROJECT_ROOT = Path(r"C:\SlickTrace")

RESULTS_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "test_predictions_055"
    / "prediction_results.csv"
)

DEMO_ROOT = PROJECT_ROOT / "demo_tiffs"
TOP_N = 12

if not RESULTS_CSV.exists():
    raise FileNotFoundError(f"Prediction results not found:\n{RESULTS_CSV}")

df = pd.read_csv(RESULTS_CSV)

required = [
    "image_path",
    "mask_path",
    "prediction_path",
    "dice",
    "iou",
    "precision",
    "recall",
    "predicted_pixels",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# ------------------------------------------------------------
# IMPORTANT:
# Select only actual OIL examples.
# Do NOT select perfect no-oil examples just because Dice=1.
# ------------------------------------------------------------

oil = df[
    df["image_path"].astype(str).str.contains(
        r"PART_I_OIL", case=False, regex=True
    )
].copy()

for col in ["dice", "iou", "precision", "recall", "predicted_pixels"]:
    oil[col] = pd.to_numeric(oil[col], errors="coerce")

oil = oil.dropna(subset=["dice", "iou"])

# Highest Dice first, then IoU, then precision.
oil = oil.sort_values(
    ["dice", "iou", "precision"],
    ascending=[False, False, False],
)

# ------------------------------------------------------------
# Rebuild demo folder
# ------------------------------------------------------------

if DEMO_ROOT.exists():
    shutil.rmtree(DEMO_ROOT)

DEMO_ROOT.mkdir(parents=True, exist_ok=True)

selected = []
rank = 1

for _, row in oil.iterrows():
    if len(selected) >= TOP_N:
        break

    image_src = Path(str(row["image_path"]))
    mask_src = Path(str(row["mask_path"]))
    prediction_src = Path(str(row["prediction_path"]))

    if not image_src.exists():
        print(f"SKIP image missing: {image_src}")
        continue

    if not mask_src.exists():
        print(f"SKIP mask missing: {mask_src}")
        continue

    if not prediction_src.exists():
        print(f"SKIP prediction missing: {prediction_src}")
        continue

    scene_dir = DEMO_ROOT / f"{rank:02d}_oil_scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_src, scene_dir / "input_sar.tif")
    shutil.copy2(mask_src, scene_dir / "ground_truth_mask.tif")
    shutil.copy2(prediction_src, scene_dir / "model_prediction.tif")

    selected.append({
        "demo_rank": rank,
        "original_index": int(row["index"]),
        "original_image": str(image_src),
        "input_sar": str(scene_dir / "input_sar.tif"),
        "ground_truth_mask": str(scene_dir / "ground_truth_mask.tif"),
        "model_prediction": str(scene_dir / "model_prediction.tif"),
        "dice": float(row["dice"]),
        "iou": float(row["iou"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "predicted_pixels": int(row["predicted_pixels"]),
    })

    rank += 1

if not selected:
    raise RuntimeError(
        "No usable oil examples were found. "
        "The A: dataset paths may no longer be available."
    )

metrics = pd.DataFrame(selected)
metrics.to_csv(DEMO_ROOT / "demo_metrics.csv", index=False)

# ------------------------------------------------------------
# README
# ------------------------------------------------------------

readme = f"""SLICKTRACE AI — JUDGE DEMO TIFF GALLERY

Purpose
-------
A compact collection of {len(metrics)} high-performing REAL OIL examples
from the completed prediction run.

Selection
---------
Source run:
C:\\SlickTrace\\outputs\\test_predictions_055\\prediction_results.csv

Only PART_I_OIL examples were considered.
Examples were sorted by:
1. Dice score
2. IoU
3. Precision

No-oil examples were intentionally excluded, even when their Dice score
was 1.0, because this gallery is intended to demonstrate actual
oil-spill segmentation.

Each scene contains:
- input_sar.tif          Original SAR input
- ground_truth_mask.tif  Reference mask
- model_prediction.tif  Model output

Metrics
-------
See demo_metrics.csv for Dice, IoU, Precision, Recall and predicted pixels.

Judge presentation
-------------------
Recommended visual flow:

SAR INPUT
   ->
MODEL PREDICTION
   ->
GROUND TRUTH COMPARISON
   ->
SPILL CHARACTERIZATION
   ->
PROBABLE ORIGIN
   ->
AIS CANDIDATE EVIDENCE

Important
---------
These examples demonstrate successful model cases.
They should not be presented as representative of every possible
oil-spill scene.

SlickTrace AI is an evidence-based decision-support system.
A detected spill and a ranked vessel candidate do not establish
responsibility or guilt.
"""

(DEMO_ROOT / "README.txt").write_text(readme, encoding="utf-8")

print("=" * 75)
print("SLICKTRACE AI — JUDGE DEMO GALLERY CREATED")
print("=" * 75)
print(f"Source results : {RESULTS_CSV}")
print(f"Oil examples   : {len(oil)}")
print(f"Selected       : {len(metrics)}")
print(f"Demo folder    : {DEMO_ROOT}")
print()

for _, r in metrics.iterrows():
    print(
        f"#{int(r.demo_rank):02d}  "
        f"Dice={r.dice:.4f}  "
        f"IoU={r.iou:.4f}  "
        f"Precision={r.precision:.4f}  "
        f"Recall={r.recall:.4f}  "
        f"pixels={int(r.predicted_pixels)}"
    )

print()
print("Open with:")
print(f'explorer "{DEMO_ROOT}"')