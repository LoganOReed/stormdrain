import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# average in july: 3.7in
# average in feb: 2.2in
# highest ever, july 1942: 34.5in
# 4 block graph: ~1500x1500ftsq = 209000m^2
# so 21250 m^3 over lets say an hour
# -> 6m^3/s over entire area

# According to chatgpt, drains range from 0.006 to 0.12 m^3/s

class DrainGraph:
    def __init__(self):
        """
        creates graph
        TODO: Explain layout
        """
        self.fps = 10
        self.T = 3600
        self.resolution = 10 # animation changes every "resolution" number of T steps
        self.rainRate = 0.375 # (m^3/s) for each node, from calculations above
        self.G = nx.DiGraph()
        self.G.add_nodes_from([i for i in range(0,17)])
        drainRates = [0.08 for i in range(1,17)]

        self.G.nodes[0]["amount"] = 0.0
        self.G.nodes[0]["drainRate"] = 0.0
        for i in range(1,17):
            self.G.nodes[i]["amount"] = 0.0
            self.G.nodes[i]["drainRate"] = drainRates[i-1]
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
        print(self.G.nodes.data())

    def update(self, frame):
        tCurr = frame*self.resolution
        for i in range(self.resolution):
            for n in range(1,17):
                self.G.nodes[n]['amount'] += 10  # Example update logic

    def draw(self):
        def get_colors():
            amounts = [self.G.nodes[n]['amount'] for n in range(0,17)]
            # Normalize for colormap
            # TODO: Update this when using an actual update function
            # norm = plt.Normalize(min(amounts), max(amounts))
            norm = plt.Normalize(0, 10000)
            cmap = plt.cm.viridis
            return [cmap(norm(a)) for a in amounts]

        fig, ax = plt.subplots()
        pos = []
        pos.append(np.array([5,1.5]))
        for i in range(1,17):
            pos.append(([(i-1) // 4, (i-1) % 4]))
        nodes = nx.draw_networkx_nodes(self.G, pos, node_color=get_colors(), ax=ax)
        edges = nx.draw_networkx_edges(self.G, pos, ax=ax)

        def update_visuals(frame):
            # Example: Increment each 'amount' randomly
            for n in self.G.nodes():
                self.update(frame)
            colors = get_colors()
            nodes.set_color(colors)
            return (nodes,)

        
        ani = animation.FuncAnimation(fig, update_visuals, frames=self.T // self.resolution, interval=1000/self.fps, blit=True)
        ani.save('figures/drainGraph.mp4', writer='ffmpeg', fps=self.fps )

        # edge_labels = nx.get_edge_attributes(self.G, 'weight')
        # nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        for node in self.G.nodes:
            indeg = self.G.in_degree(node)
            outdeg = self.G.out_degree(node)
            print(f"Node {node}: in={indeg}, out={outdeg}")


if __name__ == "__main__":
    graph = DrainGraph()
    graph.draw()

