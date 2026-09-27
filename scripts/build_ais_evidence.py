from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = (
    PROJECT_ROOT
    / "outputs"
    / "ais_candidates_2025-01-08.csv"
)

TRAJECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "ais_trajectory_analysis_2025-01-08.csv"
)

GEOMETRY = (
    PROJECT_ROOT
    / "outputs"
    / "ais_geometry_analysis_2025-01-08.csv"
)

OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ais_evidence_2025-01-08.csv"
)


def main():

    print("Loading AIS evidence files...")

    candidates = pd.read_csv(
        CANDIDATES,
        dtype={"mmsi": str},
    )

    trajectory = pd.read_csv(
        TRAJECTORY,
        dtype={"mmsi": str},
    )

    geometry = pd.read_csv(
        GEOMETRY,
        dtype={"mmsi": str},
    )

    print(f"Spatial candidates: {len(candidates)}")
    print(f"Trajectory records: {len(trajectory)}")
    print(f"Geometry records: {len(geometry)}")

    # Keep only the trajectory/geometry fields that add
    # new evidence to the base candidate table.

    trajectory_columns = [
        "mmsi",
        "before_distance_km",
        "after_distance_km",
        "local_direction",
        "approach_pattern",
        "observations_before",
        "observations_after",
        "total_observations",
    ]

    geometry_columns = [
        "mmsi",
        "sog_at_min_kmh",
        "cog_at_min_degrees",
        "bearing_to_spill_degrees",
        "heading_difference_degrees",
        "movement_bearing_degrees",
        "movement_to_spill_difference_degrees",
        "geometry_assessment",
    ]

    trajectory = trajectory[
        trajectory_columns
    ]

    geometry = geometry[
        geometry_columns
    ]

    # Merge spatial candidate evidence
    # with trajectory evidence.

    evidence = candidates.merge(
        trajectory,
        on="mmsi",
        how="left",
        suffixes=("", "_trajectory"),
    )

    # Merge geometry evidence.

    evidence = evidence.merge(
        geometry,
        on="mmsi",
        how="left",
    )

    # Explicitly document the current limitation.
    evidence["temporal_compatibility"] = (
        "unavailable"
    )

    evidence["analysis_date"] = (
        "2025-01-08"
    )

    evidence["spill_latitude"] = (
        28.942782
    )

    evidence["spill_longitude"] = (
        -88.834958
    )

    evidence["spill_area_km2"] = (
        57.677181
    )

    # Evidence categories.
    evidence["spatial_evidence"] = (
        evidence["min_distance_km"]
        <= 10
    )

    evidence["trajectory_evidence"] = (
        evidence["approach_pattern"]
        == "approached_then_departed"
    )

    evidence["geometry_evidence"] = (
        evidence["geometry_assessment"].isin(
            [
                "motion_toward_spill",
                "heading_toward_spill",
            ]
        )
    )

    # No final responsibility score is generated here.
    #
    # This file is an evidence table, not a guilt/
    # responsibility determination.

    evidence["attribution_status"] = (
        "candidate_only"
    )

    # Sort by spatial proximity for easy inspection.

    evidence = evidence.sort_values(
        "min_distance_km"
    )

    evidence.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("AIS EVIDENCE TABLE COMPLETE")
    print("----------------------------")
    print(f"Rows: {len(evidence)}")
    print(f"Columns: {len(evidence.columns)}")
    print(f"Output: {OUTPUT}")
    print()

    print(
        "Top 20 spatial candidates:"
    )

    display_columns = [
        "mmsi",
        "vessel_name",
        "min_distance_km",
        "approach_pattern",
        "geometry_assessment",
        "temporal_compatibility",
        "attribution_status",
    ]

    print(
        evidence[
            display_columns
        ].head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()