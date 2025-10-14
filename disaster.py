import json
import math
from datetime import datetime, timedelta

# Disaster parameters
CENTER = (121.4388617427341, 25.17023490455234)  # (lon, lat)
EXPANSION_RATE = 0.5       # meters per second
RADIUS_START = 0           # meters
DURATION = 1800             # total seconds (10 minutes)
STEP = 10                  # output every 10 seconds
START_TIME = datetime(2025, 10, 7, 12, 0, 0)

features = []

for t in range(0, DURATION + STEP, STEP):
    radius_m = RADIUS_START + EXPANSION_RATE * t
    timestamp = START_TIME + timedelta(seconds=t)

    # Convert meters to degrees
    # 1 deg latitude ≈ 111_320 m, longitude scales by cos(lat)
    coords = []
    for deg in range(0, 360, 10):
        rad = math.radians(deg)
        dlat = (radius_m / 111_320) * math.sin(rad)
        dlon = (radius_m / (111_320 * math.cos(math.radians(CENTER[1])))) * math.cos(rad)
        coords.append([CENTER[0] + dlon, CENTER[1] + dlat])
    coords.append(coords[0])  # close polygon

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords]
        },
        "properties": {
            "timestamp": timestamp.isoformat() + "Z",
            "radius_m": radius_m
        }
    })

geojson = {"type": "FeatureCollection", "features": features}

with open("output_files/expanding_disaster.geojson", "w") as f:
    json.dump(geojson, f, indent=2)

print("Saved expanding_disaster.geojson")
