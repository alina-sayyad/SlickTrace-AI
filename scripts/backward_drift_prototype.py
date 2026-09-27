from pathlib import Path
import math
import pandas as pd


# ============================================================
# SLICKTRACE AI — BACKWARD DRIFT PROTOTYPE
# ============================================================
# IMPORTANT:
# This is a DEMONSTRATION drift model.
# It does NOT use real historical ocean-current data because
# the Part III dataset contains no acquisition timestamp and
# no environmental forcing data.
#
# The model starts from the detected spill centroid and
# integrates a configurable current vector BACKWARD in time.
# ============================================================


# -----------------------------
# INPUT: DETECTED SPILL
# -----------------------------

SPILL_LAT = 28.942782
SPILL_LON = -88.834958

# -----------------------------
# PROTOTYPE ENVIRONMENT
# -----------------------------
# These values are configurable demonstration values.
#
# Current direction:
# Direction the water is MOVING toward, clockwise from North.
#
# Example:
# 90 degrees = east
# 180 degrees = south

CURRENT_SPEED_MPS = 0.30
CURRENT_DIRECTION_DEG = 135.0

# Backward simulation duration
HOURS_BACK = 12

# Number of output points
STEPS = 25

# -----------------------------
# OUTPUT
# -----------------------------

OUTPUT_DIR = Path(r"C:\SlickTrace\outputs\backward_drift_prototype")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "backward_drift_track.csv"


def destination_point(lat, lon, distance_m, bearing_deg):
    """
    Move a geographic point by distance_m along bearing_deg.

    Bearing:
        0   = north
        90  = east
        180 = south
        270 = west
    """

    earth_radius = 6_371_000.0

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)

    angular_distance = distance_m / earth_radius

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance)
        - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def main():

    print("=" * 65)
    print("SLICKTRACE AI — BACKWARD DRIFT PROTOTYPE")
    print("=" * 65)

    print(f"\nDetected spill:")
    print(f"  Latitude : {SPILL_LAT:.6f}")
    print(f"  Longitude: {SPILL_LON:.6f}")

    print("\nPrototype environmental forcing:")
    print(f"  Current speed     : {CURRENT_SPEED_MPS:.2f} m/s")
    print(f"  Current direction : {CURRENT_DIRECTION_DEG:.1f}°")
    print(f"  Time window       : {HOURS_BACK} hours")

    print("\nWARNING:")
    print("  This is NOT a historical hindcast.")
    print("  Environmental forcing is synthetic/configurable.")
    print("  The dataset contains no acquisition timestamp.")

    # --------------------------------------------------------
    # We move BACKWARD opposite to the current direction.
    # --------------------------------------------------------

    backward_bearing = (CURRENT_DIRECTION_DEG + 180.0) % 360.0

    total_seconds = HOURS_BACK * 3600.0
    time_step_seconds = total_seconds / (STEPS - 1)

    rows = []

    for i in range(STEPS):

        elapsed_seconds = i * time_step_seconds
        elapsed_hours = elapsed_seconds / 3600.0

        distance_m = CURRENT_SPEED_MPS * elapsed_seconds

        lat, lon = destination_point(
            SPILL_LAT,
            SPILL_LON,
            distance_m,
            backward_bearing,
        )

        rows.append(
            {
                "step": i,
                "hours_before_detection": elapsed_hours,
                "latitude": lat,
                "longitude": lon,
                "distance_from_spill_km": distance_m / 1000.0,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nBackward drift track:")
    print(df.to_string(index=False))

    print("\n" + "=" * 65)
    print("RESULT")
    print("=" * 65)

    final = df.iloc[-1]

    print(
        f"\nAt approximately {HOURS_BACK} hours before detection:"
    )
    print(f"  Latitude : {final['latitude']:.6f}")
    print(f"  Longitude: {final['longitude']:.6f}")
    print(
        f"  Distance : {final['distance_from_spill_km']:.2f} km"
    )

    print(f"\nSaved:")
    print(f"  {OUTPUT_CSV}")

    print("\nPipeline status:")
    print("  [PASS] Detected spill")
    print("  [PASS] Georeferenced centroid")
    print("  [PASS] Backward drift prototype")
    print("  [NEXT] Origin uncertainty region")
    print("  [NEXT] AIS candidate filtering")
    print("  [NEXT] Vessel evidence ranking")


if __name__ == "__main__":
    main()