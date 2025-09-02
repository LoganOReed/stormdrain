import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def networkxExample():
    G = nx.dodecahedral_graph()
    shells = [[2, 3, 4, 5, 6], [8, 1, 0, 19, 18, 17, 16, 15, 14, 7], [9, 10, 11, 12, 13]]
    nx.draw_shell(G, nlist=shells)
    plt.savefig("figures/test.png")



if __name__ == "__main__":
    networkxExample()

