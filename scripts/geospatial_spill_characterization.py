import os
import cv2
import numpy as np
import pandas as pd
import rasterio

PRED_DIR = r"C:\SlickTrace\outputs\part_iii_predictions"
RESULTS = os.path.join(PRED_DIR, "prediction_results.csv")
OUTPUT = os.path.join(PRED_DIR, "spill_events_geospatial.csv")

MIN_AREA_PIXELS = 500

df = pd.read_csv(RESULTS)
rows = []

for _, row in df.iterrows():

    image_path = row["image_path"]
    prediction_path = row["prediction_path"]

    with rasterio.open(image_path) as src:
        mask = rasterio.open(prediction_path).read(1)
        left, bottom, right, top = src.bounds
        width = src.width
        height = src.height
        crs = src.crs

    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    lon_per_pixel = (right - left) / width
    lat_per_pixel = (top - bottom) / height

    pixel_area_km2 = (
        abs(lon_per_pixel) *
        abs(lat_per_pixel) *
        111.32 *
        111.32 *
        np.cos(np.radians((top + bottom) / 2))
    )

    component_id = 0

    for label in range(1, num_labels):

        area_pixels = int(stats[label, cv2.CC_STAT_AREA])

        if area_pixels < MIN_AREA_PIXELS:
            continue

        component_id += 1

        cx, cy = centroids[label]

        longitude = left + (cx / width) * (right - left)
        latitude = top - (cy / height) * (top - bottom)

        bbox_x = int(stats[label, cv2.CC_STAT_LEFT])
        bbox_y = int(stats[label, cv2.CC_STAT_TOP])
        bbox_w = int(stats[label, cv2.CC_STAT_WIDTH])
        bbox_h = int(stats[label, cv2.CC_STAT_HEIGHT])

        area_km2 = area_pixels * pixel_area_km2

        rows.append({
            "index": row["index"],
            "label": os.path.basename(os.path.dirname(image_path)),
            "image_path": image_path,
            "prediction_path": prediction_path,
            "component_id": component_id,
            "area_pixels": area_pixels,
            "area_km2_approx": area_km2,
            "centroid_x_px": cx,
            "centroid_y_px": cy,
            "longitude": longitude,
            "latitude": latitude,
            "bbox_x_px": bbox_x,
            "bbox_y_px": bbox_y,
            "bbox_width_px": bbox_w,
            "bbox_height_px": bbox_h,
            "crs": str(crs),
        })

result = pd.DataFrame(rows)

result.to_csv(OUTPUT, index=False)

print("=" * 70)
print("SLICKTRACE AI GEOSPATIAL SPILL CHARACTERIZATION")
print("=" * 70)
print("Input images       :", len(df))
print("Detected events    :", len(result))
print("Minimum area       :", MIN_AREA_PIXELS, "pixels")
print("Output             :", OUTPUT)

if len(result):
    print()
    print(
        result[
            [
                "index",
                "label",
                "area_km2_approx",
                "latitude",
                "longitude",
            ]
        ]
        .sort_values("area_km2_approx", ascending=False)
        .head(15)
        .to_string(index=False)
    )
