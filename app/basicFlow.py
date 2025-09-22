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
# TODO: FIX

# According to chatgpt, drains range from 0.006 to 0.12 m^3/s

class DrainGraph:
    def __init__(self):
        """
        creates graph
        TODO: Explain layout
        """
        self.fps = 10
        self.T = 3600
        self.drainedAmount = 0
        self.runoffAmount = 0
        self.resolution = 10 # animation changes every "resolution" number of T steps
        self.rainRate = 1.25 # (m^3/s) for each node, from calculations above
        self.peakDischarge = 0.0 # peak discharge at a time step of runoff + drainage
        self.newFraction = 0.0 # currently runoff/totalwater
        self.G = nx.DiGraph()
        self.G.add_nodes_from([i for i in range(0,17)])
        drainRates = [0.08 for i in range(1,17)]

        self.G.nodes[0]["amount"] = 0.0
        self.G.nodes[0]["drainRate"] = 0.0
        for i in range(1,17):
            self.G.nodes[i]["amount"] = 0.0
            self.G.nodes[i]["drainRate"] = drainRates[i-1]
        self.G.add_weighted_edges_from([ # 0 is runoff
                (1, 2, 0.15),
                (2, 3, 0.15),
                (2, 6, 0.15),
                (3, 7, 0.15),
                (4, 3, 0.15),
                (4, 8, 0.15),
                (5, 1, 0.15),
                (5, 6, 0.15),
                (6, 10, 0.15),
                (7, 6, 0.15),
                (7, 8, 0.15),
                (7, 11, 0.15),
                (8, 12, 0.15),
                (9, 5, 0.15),
                (9, 10, 0.15),
                (9, 13, 0.15),
                (10, 14, 0.15),
                (11, 15, 0.15),
                (11, 10, 0.15),
                (12, 11, 0.15),
                (13, 14, 0.15),
                (13, 0,  np.finfo(np.float32).max), #largest 32bit fl val
                (14, 15, 0.15),
                (14, 0,  np.finfo(np.float32).max), #largest 32bit fl val
                (15, 0,  np.finfo(np.float32).max), #largest 32bit fl val
                (16, 12, 0.15),
                (16, 15, 0.15),
                (16, 0,  np.finfo(np.float32).max) #largest 32bit fl val
                                        ])
        print(self.G.nodes.data())

    def update(self, frame):
        tCurr = frame*self.resolution

        # Actual update steps
        for i in range(self.resolution):
            drainedAmt = 0.0
            runoffAmt = 0.0
            for n in range(1,17):
        # 1. add water from rain 
                self.G.nodes[n]['amount'] += self.rainRate
        # 2. remove water through drains
                dAmt = self.G.nodes[n]['drainRate']
                if dAmt <= self.G.nodes[n]['amount']:
                    self.drainedAmount += dAmt
                    drainedAmt += dAmt
                    self.G.nodes[n]['amount'] -= dAmt
                else:
                    self.drainedAmount += self.G.nodes[n]['amount']
                    drainedAmt += self.G.nodes[n]['amount']
                    self.G.nodes[n]['amount'] = 0.0
        # 3. update from edges
                available = self.G.nodes[n]['amount']
                normal_edges = []
                runoff_edges = []
                for _, v, attr in self.G.out_edges(n, data=True):
                    if v == 0:
                        runoff_edges.append((v, attr))
                    else:
                        normal_edges.append((v, attr))
                for v, attr in normal_edges:
                    capacity = attr.get('weight', 1)
                    # Only distribute if there is available flow
                    if available == 0:
                        break
                    # Distribute up to capacity or what's available
                    sent = min(capacity, available)
                    self.G.nodes[v]['amount'] += sent
                    available -= sent  # update available for remaining edges
                # After distributing, update the source node's amount
                self.G.nodes[n]['amount'] = available
                for v, attr in runoff_edges:
                    capacity = attr.get('weight', 1)
                    # Only distribute if there is available flow
                    if available == 0:
                        break
                    # Distribute up to capacity or what's available
                    sent = min(capacity, available)
                    self.G.nodes[v]['amount'] += sent
                    self.runoffAmount += sent
                    runoffAmt += sent
                    available -= sent  # update available for remaining edges
                # After distributing, update the source node's amount
                self.G.nodes[n]['amount'] = available

            # update metadata info
            if self.peakDischarge <= drainedAmt + runoffAmt:
                self.peakDischarge = drainedAmt + runoffAmt
                print(self.peakDischarge)



                    

    def draw(self):
        def get_colors():
            amounts = [self.G.nodes[n]['amount'] for n in self.G.nodes()]
            # Normalize for colormap
            # TODO: Update this when using an actual update function
            norm = plt.Normalize(min(amounts), max(amounts))
            # norm = plt.Normalize(0, 10000)
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
        print(f"Runoff: {self.runoffAmount}, Drainage: {self.drainedAmount}, NewFraction: {self.runoffAmount / (self.runoffAmount + self.drainedAmount)}, PeakDischarge: {self.peakDischarge}")


if __name__ == "__main__":
    graph = DrainGraph()
    graph.draw()

