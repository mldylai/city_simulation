from convert import convert_geojson_to_graph, convert_graph_to_csv
from agent import Agent
import random
import csv
import networkx as nx
import datetime

FILE_PATH = "tamsui.geojson"

def run_simulation(out_csv="agents.csv", n_agents=10,
                   sim_time=300, dt=1, start_time=None):
    """
    Run agent simulation and export to CSV for Kepler.gl.
    
    Parameters
    ----------
    geojson_file : str
        Path to GeoJSON with roads and buildings.
    out_csv : str
        Output CSV file path.
    n_agents : int
        Number of agents to simulate.
    sim_time : int
        Total simulation time in seconds.
    dt : int
        Timestep size in seconds.
    start_time : datetime (optional)
        Base datetime for simulation timestamps.
    """
    G = convert_geojson_to_graph(FILE_PATH)

    # Default base time = now if not given
    if start_time is None:
        start_time = datetime.datetime.utcnow().replace(microsecond=0)

    # Pick random building nodes
    building_nodes = [n for n, d in G.nodes(data=True) if d.get("building")]
    agents = []
    for i in range(n_agents):
        start, end = random.sample(building_nodes, 2)
        try:
            path = nx.shortest_path(G, source=start, target=end, weight="weight")
            agents.append(Agent(i, path))
        except nx.NetworkXNoPath:
            continue  # skip if disconnected

    # Write to CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_id", "timestamp", "lat", "lon"])
        writer.writeheader()

        for t in range(0, sim_time, dt):
            # Convert timestep to ISO8601 datetime
            timestamp = (start_time + datetime.timedelta(seconds=t)).isoformat() + "Z"

            for agent in agents:
                pos = agent.update(G, dt)
                if pos:
                    writer.writerow({
                        "agent_id": agent.id,
                        "timestamp": timestamp,
                        "lat": pos[1],
                        "lon": pos[0],
                    })

    print(f"Simulation done. Results saved to {out_csv}")

# -------------------------------
# Example usage
# -------------------------------

run_simulation(out_csv="agents.csv")
