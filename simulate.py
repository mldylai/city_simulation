from graph import convert_geojson_to_graph
from agent import Agent
import networkx as nx
import random
import csv
import datetime

G = convert_geojson_to_graph()

def run_simulation(n_agents=50, sim_time=1000, dt=1, start_time=None):
    """
    Run agent simulation and export to CSV for Kepler.gl.
    
    Parameters
    ----------
    n_agents : int
        Number of agents to simulate.
    sim_time : int
        Total simulation time in seconds.
    dt : int
        Timestep size in seconds.
    start_time : datetime (optional)
        Base datetime for simulation timestamps.
    """

    # Default base time = now if not given
    if start_time is None:
        start_time = datetime.datetime.now(datetime.UTC).replace(microsecond=0)

    # Get building nodes
    building_nodes = [n for n, d in G.nodes(data=True) if d.get("building")]

    # Get shelter nodes
    shelters = [n for n, d in G.nodes(data=True) if d.get("shelter")]
    non_shelters = [n for n in building_nodes if n not in shelters]
    
    # Create agents
    agents = []
    for i in range(n_agents):
        start = random.choice(non_shelters)
        try:
            # Pick nearest shelter
            end = min(
                shelters,
                key=lambda s: nx.shortest_path_length(G, source=start, target=s, weight="weight")
            )
            path = nx.shortest_path(G, source=start, target=end, weight="weight")
            shelter_id = shelters.index(end)  # assign index of shelter
            agent = Agent(i, path)
            agent.shelter_id = shelter_id
            agents.append(agent)
        except (nx.NetworkXNoPath, ValueError):
            continue

    # --- Write agent positions ---
    with open("agents.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_id", "timestamp", "lat", "lon", "shelter_id"])
        writer.writeheader()

        for t in range(0, sim_time, dt):
            timestamp = (start_time + datetime.timedelta(seconds=t)).isoformat() + "Z"
            for agent in agents:
                pos = agent.update(G, dt)
                if pos:
                    writer.writerow({
                        "agent_id": agent.id,
                        "timestamp": timestamp,
                        "lat": pos[1],
                        "lon": pos[0],
                        "shelter_id": agent.shelter_id,
                    })
    print(f"Simulation done.")


run_simulation()
