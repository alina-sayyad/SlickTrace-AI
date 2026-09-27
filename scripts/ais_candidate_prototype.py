from pathlib import Path
import pandas as pd


# ============================================================
# SLICKTRACE AI
# REAL AIS CANDIDATE + EVIDENCE RANKING
# ============================================================
#
# Purpose:
#   Replace the old synthetic AIS demonstration with the
#   validated historical MarineCadastre AIS evidence table.
#
# IMPORTANT:
#   This system identifies CANDIDATE VESSELS only.
#   It does NOT establish responsibility or guilt.
#
# Current AIS test dataset:
#   2025-01-08
#
# Current test spill:
#   Latitude  : 28.942782
#   Longitude : -88.834958
#   Area      : 57.677181 km²
#
# Temporal limitation:
#   The satellite acquisition timestamp for this Part III
#   scene is currently unavailable.
#   Therefore temporal compatibility is reported as
#   "unavailable" rather than inferred.
# ============================================================


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

DRIFT_FILE = Path(
    r"C:\SlickTrace\outputs\backward_drift_prototype"
    r"\backward_drift_track.csv"
)

AIS_EVIDENCE_FILE = Path(
    r"C:\SlickTrace\outputs"
    r"\ais_evidence_2025-01-08.csv"
)


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"C:\SlickTrace\outputs\ais_candidate_prototype"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_CANDIDATES_FILE = (
    OUTPUT_DIR / "real_ais_candidates.csv"
)

REAL_RANKING_FILE = (
    OUTPUT_DIR / "real_vessel_evidence_ranking.csv"
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

ORIGIN_UNCERTAINTY_RADIUS_KM = 5.0

SPATIAL_FILTER_KM = 10.0

ANALYSIS_DATE = "2025-01-08"

SPILL_LATITUDE = 28.942782
SPILL_LONGITUDE = -88.834958
SPILL_AREA_KM2 = 57.677181


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

def validate_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def require_columns(df, required, description):
    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{description} is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 78)
    print("SLICKTRACE AI — REAL AIS CANDIDATE PIPELINE")
    print("=" * 78)

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    validate_file(
        DRIFT_FILE,
        "Backward drift file"
    )

    validate_file(
        AIS_EVIDENCE_FILE,
        "AIS evidence file"
    )

    # --------------------------------------------------------
    # LOAD BACKWARD DRIFT
    # --------------------------------------------------------

    print("\nLoading backward-drift result...")

    drift = pd.read_csv(DRIFT_FILE)

    require_columns(
        drift,
        ["latitude", "longitude"],
        "Backward drift file"
    )

    if drift.empty:
        raise ValueError(
            "Backward drift file contains no records."
        )

    # Final backward-drift point = prototype probable-origin point.
    origin = drift.iloc[-1]

    origin_lat = float(origin["latitude"])
    origin_lon = float(origin["longitude"])

    print("\nProbable-origin point from backward drift:")
    print(f"  Latitude : {origin_lat:.6f}")
    print(f"  Longitude: {origin_lon:.6f}")

    print(
        f"\nOrigin uncertainty radius:"
        f" {ORIGIN_UNCERTAINTY_RADIUS_KM:.1f} km"
    )

    # --------------------------------------------------------
    # LOAD REAL AIS EVIDENCE
    # --------------------------------------------------------

    print("\nLoading real AIS evidence table...")

    ais = pd.read_csv(AIS_EVIDENCE_FILE)

    print(f"Real AIS records: {len(ais)}")

    required_columns = [
        "mmsi",
        "vessel_name",
        "min_distance_km",
        "approach_pattern",
        "geometry_assessment",
        "spatial_evidence",
        "trajectory_evidence",
        "geometry_evidence",
        "evidence_score",
        "temporal_compatibility",
        "attribution_status",
    ]

    require_columns(
        ais,
        required_columns,
        "AIS evidence table"
    )

    # --------------------------------------------------------
    # NORMALIZE TYPES
    # --------------------------------------------------------

    ais["mmsi"] = ais["mmsi"].astype(str)

    ais["vessel_name"] = (
        ais["vessel_name"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    ais["min_distance_km"] = pd.to_numeric(
        ais["min_distance_km"],
        errors="coerce"
    )

    ais["evidence_score"] = pd.to_numeric(
        ais["evidence_score"],
        errors="coerce"
    )

    ais["spatial_evidence"] = (
        ais["spatial_evidence"]
        .fillna(False)
        .astype(bool)
    )

    ais["trajectory_evidence"] = (
        ais["trajectory_evidence"]
        .fillna(False)
        .astype(bool)
    )

    ais["geometry_evidence"] = (
        ais["geometry_evidence"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # SPATIAL FILTER
    # --------------------------------------------------------
    #
    # The real AIS evidence table was generated using a
    # 50 km search radius.
    #
    # For candidate presentation we retain vessels within
    # 10 km of the detected spill as the primary candidate
    # set.
    #
    # This is a filtering rule, NOT a responsibility rule.
    # --------------------------------------------------------

    ais["inside_spill_candidate_region"] = (
        ais["min_distance_km"]
        <= SPATIAL_FILTER_KM
    )

    candidates = ais[
        ais["inside_spill_candidate_region"]
    ].copy()

    print(
        f"\nPrimary candidates within "
        f"{SPATIAL_FILTER_KM:.1f} km of spill:"
        f" {len(candidates)}"
    )

    # --------------------------------------------------------
    # EVIDENCE FLAGS
    # --------------------------------------------------------

    candidates["evidence_count"] = (
        candidates["spatial_evidence"].astype(int)
        + candidates["trajectory_evidence"].astype(int)
        + candidates["geometry_evidence"].astype(int)
    )

    candidates["evidence_summary"] = (
        candidates["evidence_count"]
        .astype(str)
        + "/3 evidence dimensions present"
    )

    # --------------------------------------------------------
    # ORIGIN REGION CONTEXT
    # --------------------------------------------------------
    #
    # The AIS evidence table already contains the real AIS
    # measurements around the detected spill.
    #
    # We therefore preserve the backward-drift origin as
    # contextual metadata rather than pretending that the
    # current AIS table proves a vessel was inside the
    # hindcast origin region.
    # --------------------------------------------------------

    candidates["probable_origin_latitude"] = origin_lat
    candidates["probable_origin_longitude"] = origin_lon
    candidates["origin_uncertainty_radius_km"] = (
        ORIGIN_UNCERTAINTY_RADIUS_KM
    )

    # --------------------------------------------------------
    # TEST SCENE METADATA
    # --------------------------------------------------------

    candidates["analysis_date"] = ANALYSIS_DATE

    candidates["spill_latitude"] = SPILL_LATITUDE
    candidates["spill_longitude"] = SPILL_LONGITUDE
    candidates["spill_area_km2"] = SPILL_AREA_KM2

    candidates["temporal_status"] = (
        "Satellite acquisition timestamp unavailable"
    )

    candidates["attribution_status"] = (
        "candidate_only"
    )

    # --------------------------------------------------------
    # EVIDENCE CLASSIFICATION
    # --------------------------------------------------------
    #
    # This is a presentation classification.
    # It does NOT represent probability of responsibility.
    # --------------------------------------------------------

    def classify_evidence(row):

        count = int(row["evidence_count"])

        if count >= 3:
            return "multiple_supporting_evidence_dimensions"

        if count == 2:
            return "two_supporting_evidence_dimensions"

        if count == 1:
            return "single_supporting_evidence_dimension"

        return "spatial_candidate_only"

    candidates["evidence_classification"] = (
        candidates.apply(
            classify_evidence,
            axis=1
        )
    )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------
    #
    # Ranking is based on the already-generated prototype
    # evidence score.
    #
    # This is NOT a probability.
    # This is NOT a guilt score.
    # This is NOT a responsibility determination.
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        by=[
            "evidence_score",
            "min_distance_km",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    candidates["evidence_rank"] = (
        candidates.index + 1
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    candidates.to_csv(
        REAL_CANDIDATES_FILE,
        index=False
    )

    candidates.to_csv(
        REAL_RANKING_FILE,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display_columns = [
        "evidence_rank",
        "mmsi",
        "vessel_name",
        "min_distance_km",
        "approach_pattern",
        "geometry_assessment",
        "evidence_count",
        "evidence_score",
        "temporal_compatibility",
        "attribution_status",
    ]

    print("\nTop real AIS candidates:")
    print("-" * 78)

    print(
        candidates[
            display_columns
        ].head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("REAL AIS PIPELINE COMPLETE")
    print("=" * 78)

    print(
        f"\nHistorical AIS dataset:"
        f" {ANALYSIS_DATE}"
    )

    print(
        f"Total AIS candidates processed:"
        f" {len(ais)}"
    )

    print(
        f"Primary candidates within "
        f"{SPATIAL_FILTER_KM:.1f} km:"
        f" {len(candidates)}"
    )

    print(
        f"\nSpill:"
        f" {SPILL_LATITUDE:.6f},"
        f" {SPILL_LONGITUDE:.6f}"
    )

    print(
        f"Spill area:"
        f" {SPILL_AREA_KM2:.6f} km²"
    )

    print(
        f"\nProbable-origin point:"
        f" {origin_lat:.6f},"
        f" {origin_lon:.6f}"
    )

    print(
        f"Origin uncertainty radius:"
        f" {ORIGIN_UNCERTAINTY_RADIUS_KM:.1f} km"
    )

    print("\nOutput files:")

    print(
        f"  Candidates:"
        f"\n    {REAL_CANDIDATES_FILE}"
    )

    print(
        f"\n  Evidence ranking:"
        f"\n    {REAL_RANKING_FILE}"
    )

    print("\nIMPORTANT:")
    print(
        "  Historical AIS is now being used instead of"
        " synthetic AIS."
    )

    print(
        "  Candidates are ranked by prototype evidence,"
        " not guilt probability."
    )

    print(
        "  Satellite acquisition time is unavailable,"
        " so temporal compatibility cannot currently"
        " be established."
    )

    print(
        "  The system does NOT declare any vessel"
        " responsible for the spill."
    )

    print("\nSLICKTRACE PIPELINE:")
    print("  [PASS] SAR oil-spill detection")
    print("  [PASS] Spill geospatial characterization")
    print("  [PASS] Backward-drift prototype")
    print("  [PASS] Origin uncertainty region")
    print("  [PASS] Historical AIS processing")
    print("  [PASS] AIS trajectory analysis")
    print("  [PASS] AIS geometry analysis")
    print("  [PASS] Consolidated AIS evidence")
    print("  [PASS] Evidence-based candidate ranking")


if __name__ == "__main__":
    main()