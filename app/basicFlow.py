import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# average in july: 3.7in
# average in feb: 2.2in
# highest ever, july 1942: 34.5in
# 4 block graph: ~1500x1500ftsq = 209000m^2
# so 21250 m^3 over lets say an hour
# -> 6m^3/s over entire area

class DrainGraph:
    def __init__(self):
        """
        creates graph
        TODO: Explain layout
        """
        self.rainRate = 0.375 # (m^3/s) for each node, from calculations above
        self.G = nx.DiGraph()
        self.G.add_weighted_edges_from([ # 0 is runoff
                (1, 2, 0.1),
                (2, 3, 0.1),
                (2, 6, 0.1),
                (3, 7, 0.1),
                (4, 3, 0.1),
                (4, 8, 0.1),
                (5, 1, 0.1),
                (5, 6, 0.1),
                (6, 10, 0.1),
                (7, 6, 0.1),
                (7, 8, 0.1),
                (7, 11, 0.1),
                (8, 12, 0.1),
                (9, 5, 0.1),
                (9, 10, 0.1),
                (9, 13, 0.1),
                (10, 14, 0.1),
                (11, 15, 0.1),
                (11, 10, 0.1),
                (12, 11, 0.1),
                (13, 14, 0.1),
                (13, 0,  np.finfo(np.float32).max), #largest 32bit fl val
                (14, 15, 0.1),
                (14, 0,  np.finfo(np.float32).max), #largest 32bit fl val
                (15, 0,  np.finfo(np.float32).max), #largest 32bit fl val
                (16, 12, 0.1),
                (16, 15, 0.1),
                (16, 0,  np.finfo(np.float32).max) #largest 32bit fl val
                                        ])
    def draw(self):
        nx.draw_planar(self.G)
        for node in self.G.nodes:
            indeg = self.G.in_degree(node)
            outdeg = self.G.out_degree(node)
            print(f"Node {node}: in={indeg}, out={outdeg}")
        plt.savefig("figures/test.png")


if __name__ == "__main__":
    graph = DrainGraph()
    graph.draw()

