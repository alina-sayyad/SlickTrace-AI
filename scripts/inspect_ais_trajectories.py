import zstandard as zstd
import csv
import io
import math
import collections

LAT0 = 28.942782
LON0 = -88.834958
R = 6371.0

TARGETS = {
    "538009307",
    "367556640",
    "366748650",
    "563096700",
    "368099430",
    "636023613",
    "369371000",
    "366983730",
    "367391920",
    "367510030",
}


def distance_km(lat, lon):
    lat1 = math.radians(LAT0)
    lat2 = math.radians(lat)
    dlat = math.radians(lat - LAT0)
    dlon = math.radians(lon - LON0)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


ais_path = r"data\ais\raw\ais-2025-01-08.csv.zst"

vessels = collections.defaultdict(list)

with open(ais_path, "rb") as f:
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(f)
    text = io.TextIOWrapper(reader)

    csv_reader = csv.DictReader(text)

    for row in csv_reader:
        if row["mmsi"] in TARGETS:
            vessels[row["mmsi"]].append(row)

    text.detach()


print("MMSI | Vessel | Min km | Time at min | Lat | Lon | First km | Last km")
print("-" * 120)

for mmsi, rows in vessels.items():

    rows.sort(key=lambda x: x["base_date_time"])

    first = rows[0]
    last = rows[-1]

    first_distance = distance_km(
        float(first["latitude"]),
        float(first["longitude"]),
    )

    last_distance = distance_km(
        float(last["latitude"]),
        float(last["longitude"]),
    )

    closest_row = min(
        rows,
        key=lambda x: distance_km(
            float(x["latitude"]),
            float(x["longitude"]),
        ),
    )

    min_distance = distance_km(
        float(closest_row["latitude"]),
        float(closest_row["longitude"]),
    )

    print(
        mmsi,
        "|",
        first["vessel_name"],
        "|",
        round(min_distance, 2),
        "|",
        closest_row["base_date_time"],
        "|",
        closest_row["latitude"],
        "|",
        closest_row["longitude"],
        "|",
        round(first_distance, 2),
        "|",
        round(last_distance, 2),
    )