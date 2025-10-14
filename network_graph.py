import csv
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# -----------------------------
# Shelter coordinates (lon, lat)
# -----------------------------
SHELTER_COORDS = [
    (121.4503963662213, 25.174749947684575),
    (121.44445025347713, 25.17827306137527),
    (121.43907880175816, 25.179057230917174)
]

# -----------------------------
# Retrieve the street network
# -----------------------------
CENTER = (25.17023490455234, 121.4388617427341)  # (lat, lon)
DIST = 1500  # meters
G = ox.graph_from_point(CENTER, dist=DIST, network_type="all")

# Project graph to metric CRS for distance calculations
METRIC_CRS = "EPSG:3826"
G = ox.project_graph(G, to_crs=METRIC_CRS)

# Convert nodes to GeoDataFrame for spatial queries
nodes_gdf = ox.graph_to_gdfs(G, edges=False, nodes=True)

# -----------------------------
# Add shelters as nodes and connect to nearest road node
# -----------------------------
for lon, lat in SHELTER_COORDS:
    # Project to metric CRS
    pt_metric = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]
    x, y = pt_metric.x, pt_metric.y
    shelter_node = (x, y)

    # Add shelter node with x/y attributes
    G.add_node(shelter_node, shelter=True, x=x, y=y)

    # Find nearest road node
    nearest_node_id = nodes_gdf.geometry.distance(pt_metric).idxmin()
    nearest_point = nodes_gdf.loc[nearest_node_id].geometry

    # Compute distance and add edge
    dist = pt_metric.distance(nearest_point)
    G.add_edge(shelter_node, nearest_node_id, weight=dist, road_type="shelter_link")
    G.add_edge(nearest_node_id, shelter_node, weight=dist, road_type="shelter_link")

# -----------------------------
# Convert edges to GeoDataFrame for Kepler.gl
# -----------------------------
edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
edges_gdf = edges_gdf.to_crs(epsg=4326)  # convert back to lon/lat

edges_gdf.to_file("output_files/graph_edges.geojson", driver="GeoJSON")
print(f"Exported {len(edges_gdf)} edges with shelter links to GeoJSON")


def export_shelters():
    with open("output_files/shelters.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shelter_id", "lon", "lat"])
        writer.writeheader()
        for i, s in enumerate(SHELTER_COORDS):
            writer.writerow({
                "shelter_id": i,
                "lon": s[0],
                "lat": s[1],
            })
    print("Exported shelters to shelters.csv")

export_shelters()