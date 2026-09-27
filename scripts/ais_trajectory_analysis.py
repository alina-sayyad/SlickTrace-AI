from pathlib import Path
import csv
import io
import math
from collections import defaultdict

import zstandard as zstd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AIS_INPUT = (
    PROJECT_ROOT
    / "data"
    / "ais"
    / "raw"
    / "ais-2025-01-08.csv.zst"
)

OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ais_trajectory_analysis_2025-01-08.csv"
)

SPILL_LAT = 28.942782
SPILL_LON = -88.834958

SEARCH_RADIUS_KM = 50.0

EARTH_RADIUS_KM = 6371.0

# Analyze the trajectory around the closest AIS observation.
WINDOW_MINUTES = 60


def haversine_km(lat1, lon1, lat2, lon2):

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def parse_time(value):
    from datetime import datetime

    return datetime.strptime(
        value,
        "%Y-%m-%d %H:%M:%S",
    )


def get_distance(row):

    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (TypeError, ValueError):
        return None

    return haversine_km(
        SPILL_LAT,
        SPILL_LON,
        lat,
        lon,
    )


def classify_local_trajectory(before_distance, after_distance):

    if before_distance is None or after_distance is None:
        return "insufficient_data"

    change = before_distance - after_distance

    # Small changes are treated as approximately stationary.
    if abs(change) < 1.0:
        return "near_stationary"

    if change > 0:
        return "approaching"

    return "departing"


def analyze_vessel(rows):

    rows = sorted(
        rows,
        key=lambda x: x["base_date_time"],
    )

    observations = []

    for row in rows:

        distance = get_distance(row)

        if distance is None:
            continue

        row_copy = dict(row)
        row_copy["_distance_km"] = distance

        observations.append(row_copy)

    if not observations:
        return None

    closest_index = min(
        range(len(observations)),
        key=lambda i: observations[i]["_distance_km"],
    )

    closest = observations[closest_index]

    closest_time = parse_time(
        closest["base_date_time"]
    )

    before = []
    after = []

    for row in observations:

        timestamp = parse_time(
            row["base_date_time"]
        )

        delta_minutes = (
            timestamp - closest_time
        ).total_seconds() / 60.0

        if (
            -WINDOW_MINUTES
            <= delta_minutes
            < 0
        ):
            before.append(row)

        elif (
            0
            < delta_minutes
            <= WINDOW_MINUTES
        ):
            after.append(row)

    before_distance = None
    after_distance = None

    if before:
        before_distance = before[0]["_distance_km"]

    if after:
        after_distance = after[-1]["_distance_km"]

    local_direction = classify_local_trajectory(
        before_distance,
        after_distance,
    )

    # Determine whether the vessel moved closer before
    # the minimum point and farther away afterward.
    approach_pattern = "unknown"

    if before and after:

        distances_before = [
            x["_distance_km"]
            for x in before
        ]

        distances_after = [
            x["_distance_km"]
            for x in after
        ]

        before_trend = (
            distances_before[-1]
            < distances_before[0]
        )

        after_trend = (
            distances_after[-1]
            > distances_after[0]
        )

        if before_trend and after_trend:
            approach_pattern = (
                "approached_then_departed"
            )

        elif before_trend:
            approach_pattern = "approaching"

        elif after_trend:
            approach_pattern = "departing"

        else:
            approach_pattern = "mixed"

    return {
        "mmsi": closest["mmsi"],
        "vessel_name": closest["vessel_name"],
        "imo": closest["imo"],
        "call_sign": closest["call_sign"],
        "vessel_type": closest["vessel_type"],

        "min_distance_km": round(
            closest["_distance_km"],
            3,
        ),

        "min_distance_time": (
            closest["base_date_time"]
        ),

        "min_distance_latitude": (
            closest["latitude"]
        ),

        "min_distance_longitude": (
            closest["longitude"]
        ),

        "before_distance_km": (
            round(before_distance, 3)
            if before_distance is not None
            else ""
        ),

        "after_distance_km": (
            round(after_distance, 3)
            if after_distance is not None
            else ""
        ),

        "local_direction": local_direction,

        "approach_pattern": approach_pattern,

        "observations_before": len(before),

        "observations_after": len(after),

        "total_observations": len(observations),
    }


def main():

    if not AIS_INPUT.exists():
        raise FileNotFoundError(
            f"AIS file not found: {AIS_INPUT}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # First identify all MMSIs that entered the
    # 50 km candidate region.
    vessel_rows = defaultdict(list)

    with open(
        AIS_INPUT,
        "rb",
    ) as raw_file:

        decompressor = zstd.ZstdDecompressor()

        with decompressor.stream_reader(
            raw_file
        ) as reader:

            text_stream = io.TextIOWrapper(
                reader
            )

            csv_reader = csv.DictReader(
                text_stream
            )

            for row in csv_reader:

                distance = get_distance(row)

                if distance is None:
                    continue

                if distance <= SEARCH_RADIUS_KM:

                    vessel_rows[
                        row["mmsi"]
                    ].append(row)

    results = []

    for rows in vessel_rows.values():

        result = analyze_vessel(rows)

        if result is not None:
            results.append(result)

    results.sort(
        key=lambda x: x["min_distance_km"]
    )

    fieldnames = [
        "mmsi",
        "vessel_name",
        "imo",
        "call_sign",
        "vessel_type",
        "min_distance_km",
        "min_distance_time",
        "min_distance_latitude",
        "min_distance_longitude",
        "before_distance_km",
        "after_distance_km",
        "local_direction",
        "approach_pattern",
        "observations_before",
        "observations_after",
        "total_observations",
    ]

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("AIS TRAJECTORY ANALYSIS COMPLETE")
    print("--------------------------------")
    print(f"Candidate vessels: {len(results)}")
    print(f"Analysis window: +/- {WINDOW_MINUTES} minutes")
    print(f"Output: {OUTPUT}")
    print()

    print(
        "MMSI | Vessel | Min km | "
        "Before | After | Pattern"
    )
    print("-" * 100)

    for result in results[:30]:

        print(
            f"{result['mmsi']} | "
            f"{result['vessel_name']} | "
            f"{result['min_distance_km']:.2f} | "
            f"{result['before_distance_km']} | "
            f"{result['after_distance_km']} | "
            f"{result['approach_pattern']}"
        )


if __name__ == "__main__":
    main()