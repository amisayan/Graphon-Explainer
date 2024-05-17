import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def plot_pyg_graph(data):
    """
    Plot a PyTorch Geometric data object as a graph using NetworkX.

    Parameters:
    - data: PyTorch Geometric data object

    Returns:
    - None (plots the graph)
    """

    # Convert PyTorch Geometric data object to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Get node labels if available in the data object
    node_labels = None
    if data.x is not None and len(data.x) > 0:
        node_labels = {i: str(i) for i in range(len(data.x))}

    # Plot the graph
    pos = nx.spring_layout(G)  # You can choose a different layout algorithm if needed
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=500, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
    plt.show()

# Example usage:
# Assuming you have a PyTorch Geometric data object named 'my_data'
# plot_pyg_graph(my_data)
