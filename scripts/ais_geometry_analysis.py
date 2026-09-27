from pathlib import Path
import csv
import io
import math
from collections import defaultdict
from datetime import datetime

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
    / "ais_geometry_analysis_2025-01-08.csv"
)

SPILL_LAT = 28.942782
SPILL_LON = -88.834958

SEARCH_RADIUS_KM = 50.0

WINDOW_MINUTES = 60

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


def bearing_degrees(lat1, lon1, lat2, lon2):

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2)

    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(dlon)
    )

    bearing = math.degrees(
        math.atan2(x, y)
    )

    return (bearing + 360) % 360


def angular_difference(a, b):

    difference = abs(a - b)

    return min(
        difference,
        360 - difference,
    )


def parse_time(value):

    return datetime.strptime(
        value,
        "%Y-%m-%d %H:%M:%S",
    )


def row_distance(row):

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


def row_bearing_to_spill(row):

    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (TypeError, ValueError):
        return None

    return bearing_degrees(
        lat,
        lon,
        SPILL_LAT,
        SPILL_LON,
    )


def analyze_vessel(rows):

    observations = []

    for row in rows:

        distance = row_distance(row)

        if distance is None:
            continue

        row = dict(row)

        row["_distance_km"] = distance
        row["_spill_bearing"] = row_bearing_to_spill(row)

        try:
            row["_time"] = parse_time(
                row["base_date_time"]
            )
        except ValueError:
            continue

        observations.append(row)

    if not observations:
        return None

    observations.sort(
        key=lambda x: x["_time"]
    )

    closest = min(
        observations,
        key=lambda x: x["_distance_km"],
    )

    closest_index = observations.index(
        closest
    )

    closest_time = closest["_time"]

    before = [
        row
        for row in observations
        if 0
        < (
            closest_time - row["_time"]
        ).total_seconds()
        <= WINDOW_MINUTES * 60
    ]

    after = [
        row
        for row in observations
        if 0
        < (
            row["_time"] - closest_time
        ).total_seconds()
        <= WINDOW_MINUTES * 60
    ]

    before.sort(
        key=lambda x: x["_time"]
    )

    after.sort(
        key=lambda x: x["_time"]
    )

    closest_lat = float(
        closest["latitude"]
    )

    closest_lon = float(
        closest["longitude"]
    )

    # COG at closest observation.
    cog = None

    try:
        cog = float(closest["cog"])
        if not math.isfinite(cog):
            cog = None
    except (TypeError, ValueError):
        pass

    # Bearing from vessel toward spill.
    spill_bearing = closest[
        "_spill_bearing"
    ]

    heading_difference = None

    if cog is not None and spill_bearing is not None:

        heading_difference = angular_difference(
            cog,
            spill_bearing,
        )

    # Estimate movement using the closest observations
    # immediately before and after the minimum-distance point.
    movement_bearing = None

    if before and after:

        previous = before[-1]
        following = after[0]

        movement_bearing = bearing_degrees(
            float(previous["latitude"]),
            float(previous["longitude"]),
            float(following["latitude"]),
            float(following["longitude"]),
        )

    movement_to_spill_difference = None

    if (
        movement_bearing is not None
        and spill_bearing is not None
    ):

        movement_to_spill_difference = (
            angular_difference(
                movement_bearing,
                spill_bearing,
            )
        )

    # Determine local geometry.
    geometry = "insufficient_data"

    if (
        heading_difference is not None
        and movement_to_spill_difference is not None
    ):

        if heading_difference <= 45:

            if movement_to_spill_difference <= 45:
                geometry = "motion_toward_spill"

            else:
                geometry = "heading_toward_spill"

        elif heading_difference <= 90:

            geometry = "partially_aligned"

        else:

            geometry = "not_aligned_with_spill"

    elif heading_difference is not None:

        if heading_difference <= 45:
            geometry = "heading_toward_spill"

        elif heading_difference <= 90:
            geometry = "partially_aligned"

        else:
            geometry = "not_aligned_with_spill"

    # Speed at closest observation.
    sog = None

    try:
        sog = float(closest["sog"])
    except (TypeError, ValueError):
        pass

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

        "min_distance_time":
            closest["base_date_time"],

        "min_distance_latitude":
            closest["latitude"],

        "min_distance_longitude":
            closest["longitude"],

        "sog_at_min_kmh": (
            round(sog, 2)
            if sog is not None
            else ""
        ),

        "cog_at_min_degrees": (
            round(cog, 2)
            if cog is not None
            else ""
        ),

        "bearing_to_spill_degrees": (
            round(spill_bearing, 2)
            if spill_bearing is not None
            else ""
        ),

        "heading_difference_degrees": (
            round(heading_difference, 2)
            if heading_difference is not None
            else ""
        ),

        "movement_bearing_degrees": (
            round(movement_bearing, 2)
            if movement_bearing is not None
            else ""
        ),

        "movement_to_spill_difference_degrees": (
            round(
                movement_to_spill_difference,
                2,
            )
            if movement_to_spill_difference
            is not None
            else ""
        ),

        "geometry_assessment": geometry,

        "observations_before": len(before),

        "observations_after": len(after),

        "total_observations": len(
            observations
        ),
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

    vessels = defaultdict(list)

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

                distance = row_distance(row)

                if distance is None:
                    continue

                if distance <= SEARCH_RADIUS_KM:

                    vessels[
                        row["mmsi"]
                    ].append(row)

    results = []

    for rows in vessels.values():

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

        "sog_at_min_kmh",
        "cog_at_min_degrees",

        "bearing_to_spill_degrees",
        "heading_difference_degrees",

        "movement_bearing_degrees",
        "movement_to_spill_difference_degrees",

        "geometry_assessment",

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
    print("AIS GEOMETRY ANALYSIS COMPLETE")
    print("--------------------------------")
    print(
        f"Candidate vessels: {len(results)}"
    )
    print(
        f"Analysis window: +/- {WINDOW_MINUTES} minutes"
    )
    print(f"Output: {OUTPUT}")
    print()

    print(
        "MMSI | Vessel | Min km | "
        "COG | Bearing | Difference | Geometry"
    )

    print("-" * 120)

    for result in results[:30]:

        print(
            f"{result['mmsi']} | "
            f"{result['vessel_name']} | "
            f"{result['min_distance_km']:.2f} | "
            f"{result['cog_at_min_degrees']} | "
            f"{result['bearing_to_spill_degrees']} | "
            f"{result['heading_difference_degrees']} | "
            f"{result['geometry_assessment']}"
        )


if __name__ == "__main__":
    main()