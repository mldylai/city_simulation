class Agent:
    def __init__(self, start_node, end_node, speed_m_s):
        self.current_node = start_node
        self.target_node = end_node
        self.path = []  # list of nodes to traverse
        self.speed_m_s = speed_m_s
        self.distance_along_edge = 0.0  # meters along the current edge
        self.completed = False

