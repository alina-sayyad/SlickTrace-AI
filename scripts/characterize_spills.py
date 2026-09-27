import os
import cv2
import numpy as np
import pandas as pd
import rasterio

PRED_DIR = r"C:\SlickTrace\outputs\test_predictions_055"
RESULTS = os.path.join(PRED_DIR, "prediction_results.csv")
OUTPUT = os.path.join(PRED_DIR, "spill_characterization.csv")

MIN_AREA_PIXELS = 100

df = pd.read_csv(RESULTS)
rows = []

for _, row in df.iterrows():
    mask_path = row["prediction_path"]

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    component_count = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < MIN_AREA_PIXELS:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        cx, cy = centroids[label]

        component_mask = (labels == label).astype(np.uint8)

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygon = max(contours, key=cv2.contourArea)

        polygon_points = [
            [int(point[0][0]), int(point[0][1])]
            for point in polygon
        ]

        component_count += 1

        rows.append({
            "index": row["index"],
            "image_path": row["image_path"],
            "prediction_path": mask_path,
            "component_id": component_count,
            "area_pixels": area,
            "centroid_x_px": round(float(cx), 2),
            "centroid_y_px": round(float(cy), 2),
            "bbox_x_px": x,
            "bbox_y_px": y,
            "bbox_width_px": w,
            "bbox_height_px": h,
            "polygon_pixels": str(polygon_points),
        })

result = pd.DataFrame(rows)

if len(result) > 0:
    result = result.sort_values(
        ["index", "area_pixels"],
        ascending=[True, False]
    )

result.to_csv(OUTPUT, index=False)

print("=" * 70)
print("SLICKTRACE AI SPILL CHARACTERIZATION")
print("=" * 70)
print("Input predictions :", len(df))
print("Detected regions  :", len(result))
print("Minimum area      :", MIN_AREA_PIXELS, "pixels")
print("Output            :", OUTPUT)

if len(result) > 0:
    print()
    print("Largest predicted regions:")
    print(
        result[
            [
                "index",
                "component_id",
                "area_pixels",
                "centroid_x_px",
                "centroid_y_px",
                "bbox_width_px",
                "bbox_height_px",
            ]
        ].head(15).to_string(index=False)
    )
else:
    print("No regions above the minimum area were detected.")
