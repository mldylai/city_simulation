import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import pandas as pd
import json

def convert_geojson_to_graph(file_path):
    """
    Convert GeoJSON roads and buildings to a NetworkX graph.
    Roads are edges with distances in meters.
    Buildings are nodes at their centroids.
    Coordinates are kept in WGS84 (lat/lon) for visualization in Kepler.gl.
    """

    # Load GeoJSON in WGS84 (lat/lon)
    gdf = gpd.read_file(file_path)

    # Project to metric CRS (meters) for distance calculations
    gdf_m = gdf.to_crs(epsg=3826)  # TWD97 / TM2 zone 121

    # Split roads and buildings in projected CRS
    roads_m = gdf_m[gdf_m["highway"].notnull()]
    buildings_m = gdf_m[gdf_m["building"].notnull()]

    # Align WGS84 rows for visualization
    gdf_wgs_roads = gdf.loc[roads_m.index]
    gdf_wgs_buildings = gdf.loc[buildings_m.index]

    # Initialize graph
    G = nx.Graph()

    # Add road edges
    for (_, row_m), (_, row_wgs) in zip(roads_m.iterrows(), gdf_wgs_roads.iterrows()):
        geom_m = row_m.geometry
        geom_wgs = row_wgs.geometry
        if isinstance(geom_m, LineString):
            coords_m = list(geom_m.coords)
            coords_wgs = list(geom_wgs.coords)
            for i in range(len(coords_m)-1):
                u_m, v_m = coords_m[i], coords_m[i+1]
                u_wgs, v_wgs = coords_wgs[i], coords_wgs[i+1]
                # Compute distance in meters
                dist_m = Point(u_m).distance(Point(v_m))
                G.add_edge(u_wgs, v_wgs, weight=dist_m, road_type=row_m["highway"])

    # Add building centroids as nodes (WGS84)
    for (_, row_m), (_, row_wgs) in zip(buildings_m.iterrows(), gdf_wgs_buildings.iterrows()):
        centroid_wgs = row_wgs.geometry.centroid
        G.add_node((centroid_wgs.x, centroid_wgs.y), building=True)

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
