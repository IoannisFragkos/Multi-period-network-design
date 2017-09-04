__author__ = 'ioannis'
"""
Helper functions for graphs of the networkx module.
"""
from helpers import read_data, get_2d_index
import networkx as nx
import numpy as np


def make_graph(data):
    """
    Makes a networkx graph from the data
    :param data:    problem data
    :return:        graph object
    """

    graph = nx.DiGraph()
    edge_list = get_2d_index(data.arcs, data.nodes)
    cost = data.variable_cost
    graph.add_weighted_edges_from(zip(
        edge_list[0]-1, edge_list[1]-1, cost))
    arc = 0
    for n1, n2 in zip(*get_2d_index(data.arcs, data.nodes)):
        graph[n1-1][n2-1]['arc_id'] = arc
        arc += 1
    return graph


def test():

    data = read_data('small7.dow')
    graph = make_graph(data)
    make_adjacency_matrix(graph, data)


def make_adjacency_matrix(graph, data):
    """
    Returns an adjacency matrix of the following structure:
    first index: arc pointer, i.e., order of arc as it is read from the input file (zero-based)
    second index: commodity pointer, i.e., order of commodity as it is read from the input file (zero-based)
    third index: for a specific arc and commodity, there exists a path from the origin to all other nodes. The third
    index stands for the entry of each path in which the arc participates
    value: for an arc, commodity pair, the vector of nodes shows in which nodes' paths balance equations the arc
    participates. An arc "participates" in a path if the arc is adjacent to the path (either starts from a node in the
    path and goes away, or ends in the path from outside the path)
    Note: the return values (nodes) are 1-index based, because we need to indicate both the node label AND whether
    the arc has a positive or negative sign. In zero-based indexing we cannot do that for node 0
    """
    paths = np.empty(shape=data.commodities, dtype=object)
    # The last element of the last dimension of the adj matrix holds the actual length of each path
    adjacency_matrix = np.zeros(shape=(data.arcs.size, data.commodities, data.nodes+1), dtype=int)

    # origins and destinations are 1-based index. All functions assume 0-based index
    origins, destinations = get_2d_index(data.od_pairs, data.nodes)
    arc_origins, arc_destinations = get_2d_index(data.arcs, data.nodes)

    for index, origin in enumerate(origins):
        # For each commodity, calculate a shortest path from its origin to all the nodes
        # paths[index] is a dictionary of the form {"node_label1" : [node_1, node_2, ..., node_n, "node_label2": etc ]
        paths[index] = nx.single_source_dijkstra_path(graph, origin-1)
        # Remove path to destination, this constraint is redundant
        del paths[index][destinations[index]-1]
        # Add each path to the adjacency matrices of each arcs
        for path in paths[index].values():
            for node in path:
                # Find arcs that depart or arrive at this node
                node_origins_ptr = np.where((arc_origins == node + 1))   # pointers of arcs that have node as origin
                node_destinations_ptr = np.where(arc_destinations == node + 1)
                if node_origins_ptr[0].size > 0:
                    for arc_pointer in np.nditer(node_origins_ptr):
                        arc_end = get_2d_index(data.arcs[arc_pointer], data.nodes)[1] - 1
                        if arc_end not in path:
                            length = adjacency_matrix[arc_pointer, index, data.nodes]
                            adjacency_matrix[arc_pointer, index, length] = path[len(path)-1]+1  # Adding +1 for 1-index
                            adjacency_matrix[arc_pointer, index, data.nodes] += 1
                if node_destinations_ptr[0].size > 0:
                    for arc_pointer in np.nditer(node_destinations_ptr):
                        arc_start = get_2d_index(data.arcs[arc_pointer], data.nodes)[0] - 1
                        if arc_start not in path:
                            length = adjacency_matrix[arc_pointer, index, data.nodes]
                            adjacency_matrix[arc_pointer, index, length] = - (path[len(path)-1] + 1)
                            adjacency_matrix[arc_pointer, index, data.nodes] += 1
    return adjacency_matrix

if __name__ == '__main__':
    test()
