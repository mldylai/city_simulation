import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import pandas as pd
import json

FILE_PATH = "tamsui.geojson"
SHELTER_COORDS = [(25.174749947684575, 121.4503963662213),
                  (25.17827306137527, 121.44445025347713),
                  (25.179057230917174, 121.43907880175816)]

def convert_geojson_to_graph():
    gdf = gpd.read_file(FILE_PATH)
    gdf_m = gdf.to_crs(epsg=3826)  # metric CRS

    roads_m = gdf_m[gdf_m["highway"].notnull()]
    buildings_m = gdf_m[gdf_m["building"].notnull()]

    gdf_wgs_roads = gdf.loc[roads_m.index]
    gdf_wgs_buildings = gdf.loc[buildings_m.index]

    G = nx.Graph()

    # ---- Add road edges ----
    for (_, row_m), (_, row_wgs) in zip(roads_m.iterrows(), gdf_wgs_roads.iterrows()):
        geom_m = row_m.geometry
        geom_wgs = row_wgs.geometry
        if isinstance(geom_m, LineString):
            coords_m = list(geom_m.coords)
            coords_wgs = list(geom_wgs.coords)
            for i in range(len(coords_m)-1):
                u_m, v_m = coords_m[i], coords_m[i+1]
                u_wgs, v_wgs = coords_wgs[i], coords_wgs[i+1]
                dist_m = Point(u_m).distance(Point(v_m))
                G.add_edge(u_wgs, v_wgs, weight=dist_m, road_type=row_m["highway"])

    # ---- Add building centroids as nodes ----
    road_nodes = list(G.nodes)
    for (_, row_m), (_, row_wgs) in zip(buildings_m.iterrows(), gdf_wgs_buildings.iterrows()):
        centroid_m = row_m.geometry.centroid  # projected (for distance)
        centroid_wgs = row_wgs.geometry.centroid

        building_node = (centroid_wgs.x, centroid_wgs.y)
        G.add_node(building_node, building=True)

        # connect building centroid to nearest road node
        nearest = min(
            road_nodes,
            key=lambda n: Point(centroid_wgs).distance(Point(n))
        )
        dist = centroid_m.distance(row_m.geometry)   # distance in meters
        G.add_edge(building_node, nearest, weight=dist, road_type="building_link")

    #TODO: Add shelters as nodes

    print("Conversion to graph completed.")
    return G

def convert_graph_to_csv(G):
    """
    Export graph nodes and edges to CSV files for Kepler.gl visualization.
    Nodes contain lat/lon coordinates.
    Edges contain geometry in GeoJSON format and weight in meters.
    """
    # Export nodes (lat/lon for Kepler.gl)
    nodes_df = pd.DataFrame([
        {"latitude": n[1], "longitude": n[0], **attr} for n, attr in G.nodes(data=True)
    ])
    nodes_df.to_csv("nodes.csv", index=False)

    # Export edges (lat/lon coordinates, weight in kilometers)
    edges_df = pd.DataFrame([
        {
            "geometry": json.dumps(LineString([u, v]).__geo_interface__),
            **attr
        }
        for u, v, attr in G.edges(data=True)
    ])
    edges_df.to_csv("edges.csv", index=False)
    print("Conversion to CSV completed.")
