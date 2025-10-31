# evacuation_sim.py
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from datetime import datetime, timedelta

START_TIME = datetime(2025, 10, 7, 12, 0, 0) # default start time for simulation records

# ---------------------------
# Agent and Simulation Classes
# ---------------------------

@dataclass
class Agent:
    id: int
    start_node: Any
    destination_node: Any
    path: List[Any] = field(default_factory=list)  # list of nodes
    edge_index: int = 0  # index in path for current edge (u = path[edge_index], v = path[edge_index+1])
    progress: float = 0.0  # 0..1 progress along current edge
    status: str = "waiting"  # waiting / moving / arrived / stuck
    speed_factor: float = 1.0  # multiplier on walking speed (optional)


class EvacuationSim:
    def __init__(
        self,
        G: nx.MultiDiGraph,
        shelter_node_attribute: str = "shelter",
        base_speed_m_s: float = 1.5,
        default_capacity: int = 20,
        alpha: float = 1.0,
        beta: float = 1.0,
        time_step: float = 1.0,
        random_seed: Optional[int] = 0,
    ):
        """
        G: projected graph (metric CRS, lengths in meters)
        base_speed_m_s: nominal free-flow speed (m/s)
        default_capacity: default agents per edge if not specified
        alpha, beta: congestion function parameters (travel_time = base * (1 + alpha*(load/cap/capacity)**beta))
        time_step: seconds per simulation step
        """
        if random_seed is not None:
            random.seed(random_seed)

        self.G = G.copy()  # we will modify attributes
        self.crs_projected = None
        self.crs_wgs = "EPSG:4326"
        self.base_speed = float(base_speed_m_s)
        self.default_capacity = int(default_capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dt = float(time_step)

        # identify shelter nodes
        self.shelter_nodes = [n for n, d in self.G.nodes(data=True) if d.get(shelter_node_attribute)]
        if not self.shelter_nodes:
            # allow for user-provided shelter nodes (e.g., nodes where 'shelter' attr isn't set)
            # raise warning but continue
            print("Warning: no nodes with shelter attribute found. Provide shelter nodes explicitly if needed.")

        # init edges
        self._initialize_edge_attributes()

        # agents list
        self.agents: List[Agent] = []

        # For output recording
        self.records: List[Dict] = []

        # store projected CRS if present
        n0 = next(iter(self.G.nodes(data=True)), None)
        # If nodes have x,y attributes they came from a projected graph
        # We'll detect later on export.
        self._detect_node_coordinate_attrs()

    def _detect_node_coordinate_attrs(self):
        # OSMnx projected graphs usually add 'x' and 'y' in nodes in metric CRS
        n, d = next(iter(self.G.nodes(data=True)))
        if "x" in d and "y" in d:
            self.node_coords_projected = True
        else:
            self.node_coords_projected = False

    def _initialize_edge_attributes(self):
        # For MultiDiGraph, edges are keyed: G[u][v][k]
        for u, v, k, data in self.G.edges(keys=True, data=True):
            length = data.get("length")  # should be in meters (projected graph)
            if length is None:
                # fallback to euclidean from node coordinates if available
                nu = self.G.nodes[u]
                nv = self.G.nodes[v]
                if "x" in nu and "y" in nu and "x" in nv and "y" in nv:
                    dx = nu["x"] - nv["x"]
                    dy = nu["y"] - nv["y"]
                    length = math.hypot(dx, dy)
                else:
                    length = 1.0
            data["length"] = float(length)
            # base travel time in seconds
            data["base_travel_time"] = float(length / self.base_speed)
            # dynamic state
            data.setdefault("capacity", self.default_capacity)
            data["capacity"] = int(data["capacity"])
            data.setdefault("current_load", 0)
            data["current_load"] = int(data["current_load"])
            # current travel time will be updated each step
            data["current_travel_time"] = float(data["base_travel_time"])

    def spawn_agents(self,
                    n_agents: int = None,
                    origin_nodes: Optional[List[Any]] = None,
                    start_positions: Optional[List[Tuple[float, float]]] = None):
        """
        Spawn agents either:
        - randomly from given origin_nodes or all non-shelter nodes, OR
        - at specific lon/lat coordinates (start_positions).
        """
        if start_positions is not None:
            # Convert lon/lat positions to nearest graph nodes
            start_nodes = []
            for lon, lat in start_positions:
                try:
                    pt_metric = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3826").iloc[0]
                    x, y = pt_metric.x, pt_metric.y
                    nearest_node = ox.distance.nearest_nodes(self.G, X=x, Y=y)
                    start_nodes.append(nearest_node)
                except Exception as e:
                    print(f"Warning: failed to find nearest node for ({lon}, {lat}) : {e}")
            if not start_nodes:
                raise ValueError("No valid start nodes found for given coordinates.")
        else:
            # Default: random spawn among origin_nodes
            if origin_nodes is None:
                origin_nodes = [n for n, d in self.G.nodes(data=True) if not d.get("shelter")]
            if not origin_nodes:
                raise ValueError("No origin nodes available to spawn agents.")
            start_nodes = random.choices(origin_nodes, k=n_agents)

        # Create agents
        for i, start in enumerate(start_nodes):
            # Choose nearest shelter as destination
            if self.shelter_nodes:
                def dist_to_shelter(s):
                    nu = self.G.nodes[start]
                    ns = self.G.nodes[s]
                    if "x" in nu and "y" in nu and "x" in ns and "y" in ns:
                        return math.hypot(nu["x"] - ns["x"], nu["y"] - ns["y"])
                    return float("inf")

                dest = min(self.shelter_nodes, key=dist_to_shelter)
            else:
                dest = random.choice(origin_nodes)

            agent = Agent(id=i, start_node=start, destination_node=dest, status="waiting")
            self.agents.append(agent)
    
    def _update_edge_travel_times(self):
        # update current_travel_time for all edges based on current_load and capacity
        for u, v, k, data in self.G.edges(keys=True, data=True):
            load = data.get("current_load", 0)
            cap = data.get("capacity", self.default_capacity) or 1
            base = data.get("base_travel_time", 1.0)
            # congestion multiplier
            # ensure no division by zero
            frac = float(load) / float(cap) if cap > 0 else float(load)
            multiplier = 1.0 + self.alpha * (frac ** self.beta)
            data["current_travel_time"] = float(base * multiplier)

    def _compute_shortest_path(self, source, target):
        # Create a (temporary) DiGraph with edge weight = current_travel_time
        # networkx shortest_path can operate directly on MultiDiGraph using a weight attribute, but it's safer to pass attribute name
        try:
            path = nx.shortest_path(self.G, source, target, weight="current_travel_time")
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _enter_edge(self, agent: Agent):
        """Try to occupy the next edge for agent; return True if entered, False if blocked."""
        if agent.edge_index >= len(agent.path) - 1:
            return False
        u = agent.path[agent.edge_index]
        v = agent.path[agent.edge_index + 1]
        # For MultiDiGraph choose the edge with minimal current_travel_time between u->v
        best_key = None
        best_tt = float("inf")
        for k, data in self.G[u][v].items():
            if data["current_load"] < data["capacity"]:
                if data["current_travel_time"] < best_tt:
                    best_tt = data["current_travel_time"]
                    best_key = k
        if best_key is None:
            # all parallel edges full; blocked.
            return False
        # Occupy that edge (increment load)
        self.G[u][v][best_key]["current_load"] += 1
        # store the edge key in agent for book-keeping (we'll store as attribute)
        agent._edge_key_in_use = best_key
        agent.status = "moving"
        return True

    def _leave_edge(self, agent: Agent):
        """Remove agent from current edge load when agent finishes traversing edge."""
        if agent.edge_index >= len(agent.path) - 1:
            return
        u = agent.path[agent.edge_index]
        v = agent.path[agent.edge_index + 1]
        key = getattr(agent, "_edge_key_in_use", None)
        if key is not None:
            # safe decrement
            data = self.G[u][v][key]
            data["current_load"] = max(0, data.get("current_load", 1) - 1)
            delattr(agent, "_edge_key_in_use")

    def _node_to_point_projected(self, node):
        d = self.G.nodes[node]
        if "x" in d and "y" in d:
            return (d["x"], d["y"])
        # otherwise, if node uses geometry
        if "geometry" in d:
            geom = d["geometry"]
            if geom is not None:
                return (geom.x, geom.y)
        raise RuntimeError("Node has no projected x/y coords.")

    def step(self, t: float):
        """
        Single simulation step at time t (seconds). Updates agents, loads, and records positions.
        """
        # 1) update edge travel times based on current loads
        self._update_edge_travel_times()

        # 2) for each agent decide action
        for agent in self.agents:
            if agent.status == "arrived":
                # record position at destination node
                x, y = self._node_to_point_projected(agent.destination_node)
                self._record(agent, t, x, y)
                continue

            if agent.status == "waiting":
                # compute path if none or re-evaluate
                path = self._compute_shortest_path(agent.start_node if not agent.path else agent.path[agent.edge_index], agent.destination_node)
                if not path:
                    agent.status = "stuck"
                    continue
                agent.path = path
                agent.edge_index = 0
                agent.progress = 0.0
                # try to enter first edge
                entered = self._enter_edge(agent)
                if not entered:
                    # optionally try rerouting to avoid full edges
                    # compute path from current node (still start_node) again; if path same, remain waiting
                    # leave as waiting for now
                    agent.status = "waiting"
                    self._record_agent_node_position(agent, t)
                else:
                    # successfully entered edge
                    self._record_progress_position(agent, t)
                continue

            if agent.status == "moving":
                # find the edge in use
                # compute travel time on that edge
                u = agent.path[agent.edge_index]
                v = agent.path[agent.edge_index + 1]
                key = getattr(agent, "_edge_key_in_use", None)
                if key is None:
                    # surprising: agent marked moving but no edge key; attempt to enter
                    if not self._enter_edge(agent):
                        agent.status = "waiting"
                        self._record_agent_node_position(agent, t)
                        continue
                    key = getattr(agent, "_edge_key_in_use")

                data = self.G[u][v][key]
                travel_time = data["current_travel_time"] / agent.speed_factor
                # advance
                if travel_time <= 0:
                    travel_time = 1e-6
                agent.progress += (self.dt / travel_time)
                if agent.progress >= 1.0:
                    # finish edge
                    self._leave_edge(agent)
                    agent.edge_index += 1
                    agent.progress = 0.0
                    # arrived at node v
                    if agent.edge_index >= len(agent.path) - 1:
                        # arrived at destination
                        agent.status = "arrived"
                        self._record(agent, t, *self._node_to_point_projected(agent.destination_node))
                        continue
                    else:
                        # attempt to enter next edge
                        # first check if next edge has capacity; if not, attempt reroute
                        next_u = agent.path[agent.edge_index]
                        next_v = agent.path[agent.edge_index + 1]
                        can_enter = False
                        # check if any parallel edge is not full
                        for k, d in self.G[next_u][next_v].items():
                            if d["current_load"] < d["capacity"]:
                                can_enter = True
                                break
                        if not can_enter:
                            # try reroute from current node to destination
                            new_path = self._compute_shortest_path(next_u, agent.destination_node)
                            if new_path is None or new_path == agent.path[agent.edge_index:]:
                                # no better path: wait at node until space frees
                                agent.status = "waiting"
                                self._record_agent_node_position(agent, t)
                                continue
                            else:
                                # adopt new path, reset edge_index
                                # new_path is full path from next_u to destination; need to set agent.path = [current_node] + new_path
                                agent.path = agent.path[: agent.edge_index + 1] + new_path[1:]
                                # attempt to enter next edge (recursively in next step)
                                agent.status = "waiting"
                                self._record_agent_node_position(agent, t)
                                continue
                        else:
                            # enter next edge
                            entered = self._enter_edge(agent)
                            if not entered:
                                agent.status = "waiting"
                                self._record_agent_node_position(agent, t)
                            else:
                                agent.status = "moving"
                                self._record_progress_position(agent, t)
                                continue
                else:
                    # still traversing current edge; record interpolated position
                    self._record_progress_position(agent, t)
                    continue

            if agent.status == "stuck":
                # record last known node position (if any)
                self._record_agent_node_position(agent, t)

    def run(self, max_time_s: float = 3600.0, record_interval: float = 1.0, verbose: bool = True):
        """
        Run the simulation until all agents arrived or max_time_s is reached.
        record_interval: seconds between recorded positions (we record every self.dt, but you can change)
        """
        t = 0.0
        steps = 0
        while t <= max_time_s and any(a.status in ("moving", "waiting") for a in self.agents):
            self.step(t)
            t += self.dt
            steps += 1
            if verbose and (steps % max(1, int(1.0 / self.dt))) == 0:
                arrived = sum(1 for a in self.agents if a.status == "arrived")
                total = len(self.agents)
                print(f"time={t:.0f}s  arrived={arrived}/{total}")
        # final record
        self.step(t)
        if verbose:
            arrived = sum(1 for a in self.agents if a.status == "arrived")
            print(f"Simulation finished at t={t:.1f}s: arrived={arrived}/{len(self.agents)}")

    # -----------------------
    # Recording / Exporting
    # -----------------------
    def _record(self, agent: Agent, t: float, x: float, y: float):
        self.records.append(
            {
                "agent_id": agent.id,
                "time": float(t),
                "timestamp": (START_TIME + timedelta(seconds=t)).isoformat(),
                "x_proj": float(x),
                "y_proj": float(y),
                "status": agent.status,
            }
        )

    def _record_agent_node_position(self, agent: Agent, t: float):
        # record at current node position (either start node or current node)
        if agent.status == "arrived":
            node = agent.destination_node
        else:
            # when waiting or stuck, node is agent.path[agent.edge_index]
            if agent.edge_index < len(agent.path):
                node = agent.path[agent.edge_index]
            else:
                node = agent.start_node
        x, y = self._node_to_point_projected(node)
        self._record(agent, t, x, y)

    def _record_progress_position(self, agent: Agent, t: float):
        # interpolate between u and v on agent.path with agent.progress
        if agent.edge_index >= len(agent.path) - 1:
            # record at destination
            x, y = self._node_to_point_projected(agent.destination_node)
            self._record(agent, t, x, y)
            return
        u = agent.path[agent.edge_index]
        v = agent.path[agent.edge_index + 1]
        xu, yu = self._node_to_point_projected(u)
        xv, yv = self._node_to_point_projected(v)
        p = max(0.0, min(1.0, agent.progress))
        x = xu * (1 - p) + xv * p
        y = yu * (1 - p) + yv * p
        self._record(agent, t, x, y)

    def export_records_to_csv(self, path="agent_positions.csv", to_wgs84=True):
        """
        Export recorded agent positions to CSV. Converts projected coords back to lon/lat if possible.
        """
        df = pd.DataFrame(self.records)
        if df.empty:
            print("No records to export.")
            return df
        if to_wgs84:
            # convert projected x/y to lon/lat using GeoPandas if possible
            try:
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=[Point(xy) for xy in zip(df["x_proj"], df["y_proj"])],
                    crs=self._infer_projected_crs(),
                )
                gdf = gdf.to_crs(self.crs_wgs)
                df["lon"] = gdf.geometry.x
                df["lat"] = gdf.geometry.y
            except Exception as e:
                print("Warning: failed to convert projected coords to WGS84:", e)
        out = df[["agent_id", "timestamp", "status", "lon", "lat"]]
        out.to_csv(path, index=False)
        print(f"Exported {len(df)} records to {path}")
        return df

    def _infer_projected_crs(self):
        # attempt to infer projected CRS based on node geometry if present in graph
        # If not available, default to EPSG:3826 as your earlier code used that
        # We will try to inspect the graph for a 'crs' attribute
        crs = getattr(self.G, "graph", {}).get("crs", None)
        if crs:
            return crs
        # otherwise fallback to EPSG:3826 (Taiwan TM2) or let geopandas assume
        try:
            # if nodes have 'x' and 'y' and we know user used EPSG:3826 earlier, return that
            return "EPSG:3826"
        except Exception:
            return None
