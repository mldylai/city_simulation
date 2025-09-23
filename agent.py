
class Agent:
    def __init__(self, agent_id, path, speed=1.4):
        # speed ~ 1.4 m/s (walking)
        self.id = agent_id
        self.path = path  # list of nodes (lat,lon)
        self.speed = speed
        self.edge_index = 0
        self.dist_on_edge = 0.0
        self.finished = False

    def update(self, G, dt=1.0):
        if self.finished or self.edge_index >= len(self.path) - 1:
            self.finished = True
            return self.path[-1]

        u = self.path[self.edge_index]
        v = self.path[self.edge_index + 1]
        edge_len = G[u][v]['weight']

        self.dist_on_edge += self.speed * dt
        if self.dist_on_edge >= edge_len:
            # move to next edge
            self.edge_index += 1
            self.dist_on_edge = 0.0
            if self.edge_index >= len(self.path) - 1:
                self.finished = True
                return v
            else:
                return v
        else:
            # interpolate along edge
            frac = self.dist_on_edge / edge_len
            lat = u[1] + frac * (v[1] - u[1])
            lon = u[0] + frac * (v[0] - u[0])
            return (lon, lat)  # keep (x=lon, y=lat) for Kepler

