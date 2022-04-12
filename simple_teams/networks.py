import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import scipy.spatial
from matplotlib import animation
from IPython.display import HTML


def graph_from_coordinates(coords, radius):

    """Return a random geometric graph from an array of coordinates.
        Functionality is cribbed from networkx.

    Args:
        coords (array): An array of coordinates.
        radius (float): A radius of circles or spheres.

    Returns:
        g: A networkx-graph.
    """

    kdtree = sp.spatial.cKDTree(coords)
    edge_indexes = kdtree.query_pairs(radius)
    g = nx.Graph()
    g.add_nodes_from(list(range(200)))
    g.add_edges_from(edge_indexes)
    #     G = nx.from_edgelist(edge_indexes)
    return g


def jittering_geometric_graphs(
    n_nodes=200, n_steps=400, dim=2, radius=0.1, temperature=0.01
):
    coords = np.random.rand(n_nodes, dim)

    graph_list = []
    coords_list = []
    for x in range(0, n_steps):
        G = graph_from_coordinates(coords, radius)
        graph_list.append(G)
        coords_list.append(coords)
        coords = coords + (np.random.rand(n_nodes, dim) - 0.5) * temperature
        # maybe change to circular walk?
    return graph_list, coords_list


def simple_update(num, graph_list, coords_list, ax):
    ax.clear()
    G, coords = graph_list[num], coords_list[num]
    nx.draw_networkx_nodes(G, coords, node_size=30, node_color="#1a2340", alpha=0.7)
    nx.draw_networkx_edges(G, coords, edge_color="grey", width=1, alpha=1)


def plot_graphlist(graph_list, coords_list, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ani = animation.FuncAnimation(
        fig, simple_update, frames=len(graph_list), fargs=(graph_list, coords_list, ax)
    )
    HTML(ani.to_jshtml())


def plot_graphgrid(graph_list, coords_list, figsize=(10, 10), node_color="#1a2340"):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(nrows=3, ncols=3)

    graphs_to_plot = [int(x) for x in np.linspace(0, len(graph_list) - 1, 9)]
    counter = 0
    for x in range(0, 3):
        for y in range(0, 3):
            this_axis = fig.add_subplot(gs[x, y])
            this_axis = nx.draw_networkx_nodes(
                graph_list[graphs_to_plot[counter]],
                coords_list[graphs_to_plot[counter]],
                node_size=50,
                node_color=node_color,
                alpha=0.6,
            )
            this_axis = nx.draw_networkx_edges(
                graph_list[graphs_to_plot[counter]],
                coords_list[graphs_to_plot[counter]],
                edge_color="grey",
                width=1,
                alpha=1,
            )
            counter += 1
    plt.show()
