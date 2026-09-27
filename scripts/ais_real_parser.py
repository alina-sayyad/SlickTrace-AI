from pathlib import Path
import csv
import io
import math

import zstandard as zstd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AIS_INPUT = (
    PROJECT_ROOT
    / "data"
    / "ais"
    / "raw"
    / "ais-2025-01-08.csv.zst"
)

AIS_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ais_candidates_2025-01-08.csv"
)

SPILL_LAT = 28.942782
SPILL_LON = -88.834958

SEARCH_RADIUS_KM = 50.0
EARTH_RADIUS_KM = 6371.0


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


def classify_trajectory(first_distance, last_distance):

    change = first_distance - last_distance

    if abs(change) < 5:
        return "stationary/near-stationary"

    if change > 0:
        return "approaching"

    return "departing"


def spatial_score(distance_km):

    # 100 at 0 km, decreasing linearly to 0 at 50 km.
    score = 100 * (1 - distance_km / SEARCH_RADIUS_KM)

    return max(0.0, min(100.0, score))


def trajectory_score(direction):

    if direction == "approaching":
        return 100.0

    if direction == "stationary/near-stationary":
        return 50.0

    if direction == "departing":
        return 25.0

    return 0.0


def coverage_score(points):

    # More observations provide stronger trajectory evidence.
    # Cap at 500 observations.
    return min(100.0, points / 500.0 * 100.0)


def main():

    if not AIS_INPUT.exists():
        raise FileNotFoundError(
            f"AIS file not found: {AIS_INPUT}"
        )

    AIS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = {}

    with open(AIS_INPUT, "rb") as raw_file:

        decompressor = zstd.ZstdDecompressor()

        with decompressor.stream_reader(raw_file) as reader:

            text_stream = io.TextIOWrapper(reader)

            csv_reader = csv.DictReader(text_stream)

            for row in csv_reader:

                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                except (TypeError, ValueError):
                    continue

                distance = haversine_km(
                    SPILL_LAT,
                    SPILL_LON,
                    lat,
                    lon,
                )

                if distance > SEARCH_RADIUS_KM:
                    continue

                mmsi = row["mmsi"]

                if mmsi not in candidates:

                    candidates[mmsi] = {
                        "mmsi": mmsi,
                        "vessel_name": row["vessel_name"],
                        "imo": row["imo"],
                        "call_sign": row["call_sign"],
                        "vessel_type": row["vessel_type"],
                        "ais_points": 0,

                        "min_distance_km": distance,
                        "min_distance_time": row["base_date_time"],
                        "min_distance_latitude": lat,
                        "min_distance_longitude": lon,

                        "first_timestamp": row["base_date_time"],
                        "first_latitude": lat,
                        "first_longitude": lon,
                        "first_distance_km": distance,

                        "last_timestamp": row["base_date_time"],
                        "last_latitude": lat,
                        "last_longitude": lon,
                        "last_distance_km": distance,
                    }

                candidate = candidates[mmsi]

                candidate["ais_points"] += 1

                if row["base_date_time"] < candidate["first_timestamp"]:

                    candidate["first_timestamp"] = row["base_date_time"]
                    candidate["first_latitude"] = lat
                    candidate["first_longitude"] = lon
                    candidate["first_distance_km"] = distance

                if row["base_date_time"] > candidate["last_timestamp"]:

                    candidate["last_timestamp"] = row["base_date_time"]
                    candidate["last_latitude"] = lat
                    candidate["last_longitude"] = lon
                    candidate["last_distance_km"] = distance

                if distance < candidate["min_distance_km"]:

                    candidate["min_distance_km"] = distance
                    candidate["min_distance_time"] = row["base_date_time"]
                    candidate["min_distance_latitude"] = lat
                    candidate["min_distance_longitude"] = lon

    results = []

    for candidate in candidates.values():

        distance_change = (
            candidate["first_distance_km"]
            - candidate["last_distance_km"]
        )

        direction = classify_trajectory(
            candidate["first_distance_km"],
            candidate["last_distance_km"],
        )

        s_score = spatial_score(
            candidate["min_distance_km"]
        )

        t_score = trajectory_score(
            direction
        )

        c_score = coverage_score(
            candidate["ais_points"]
        )

        evidence_score = (
            0.60 * s_score
            + 0.25 * t_score
            + 0.15 * c_score
        )

        candidate["distance_change_km"] = distance_change
        candidate["trajectory_direction"] = direction

        candidate["spatial_score"] = round(s_score, 2)
        candidate["trajectory_score"] = round(t_score, 2)
        candidate["coverage_score"] = round(c_score, 2)

        candidate["temporal_compatibility"] = "unavailable"

        candidate["evidence_score"] = round(
            evidence_score,
            2,
        )

        results.append(candidate)

    results.sort(
        key=lambda x: x["evidence_score"],
        reverse=True,
    )

    fieldnames = [
        "mmsi",
        "vessel_name",
        "imo",
        "call_sign",
        "vessel_type",
        "ais_points",

        "min_distance_km",
        "min_distance_time",
        "min_distance_latitude",
        "min_distance_longitude",

        "first_timestamp",
        "first_latitude",
        "first_longitude",
        "first_distance_km",

        "last_timestamp",
        "last_latitude",
        "last_longitude",
        "last_distance_km",

        "distance_change_km",
        "trajectory_direction",

        "spatial_score",
        "trajectory_score",
        "coverage_score",

        "temporal_compatibility",
        "evidence_score",
    ]

    with open(
        AIS_OUTPUT,
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
    print("REAL AIS EVIDENCE PROCESSING COMPLETE")
    print("--------------------------------------")
    print(f"Candidate vessels: {len(results)}")
    print(f"Output: {AIS_OUTPUT}")
    print()
    print(
        "MMSI | Vessel | Distance | "
        "Trajectory | Evidence Score"
    )
    print("-" * 90)

    for candidate in results[:20]:

        print(
            f"{candidate['mmsi']} | "
            f"{candidate['vessel_name']} | "
            f"{candidate['min_distance_km']:.2f} km | "
            f"{candidate['trajectory_direction']} | "
            f"{candidate['evidence_score']:.2f}"
        )

    print()
    print(
        "NOTE: Temporal compatibility is unavailable "
        "because the satellite acquisition timestamp "
        "is not available for this Part III scene."
    )


if __name__ == "__main__":
    main()