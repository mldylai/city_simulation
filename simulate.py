from convert import convert_geojson_to_graph, convert_graph_to_csv

FILE_PATH = "tamsui.geojson"
G = convert_geojson_to_graph(FILE_PATH)

convert_graph_to_csv(G)

