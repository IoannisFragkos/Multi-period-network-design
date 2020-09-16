"""
Helper functions
April 2015, Ioannis Fragkos
"""

from collections import namedtuple
import numpy as np


def read_data(filename):
    """
    :param filename:    name of the file (string)
    :return             a data object, encapsulating everything
    """

    data = namedtuple('data', 'commodities nodes arcs od_pairs  periods '
                    'capacity fixed_cost variable_cost demand origins '
                    'destinations')

    with open(filename, 'r') as f:
        count = 0
        line = f.readline().split()
        data.nodes, no_of_arcs, data.commodities, data.periods = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        data.arcs, data.od_pairs, data.variable_cost, data.capacity, data.fixed_cost, data.demand = \
            np.zeros(shape=no_of_arcs, dtype=int), np.zeros(shape=data.commodities, dtype=int), \
            np.zeros(shape=no_of_arcs), np.zeros(shape=no_of_arcs), \
            np.zeros(shape=(data.periods, no_of_arcs)), np.zeros(shape=(data.periods, data.commodities))
        for line in f:
            line = line.split()
            from_arc, to_arc = int(line[0]), int(line[1])
            if count < no_of_arcs:
                arc_index = get_1d_index(from_arc, to_arc, data.nodes)
                data.arcs[count] = arc_index
                data.variable_cost[count], data.capacity[count] = float(line[2]), float(line[3])
                for period in range(data.periods):
                    data.fixed_cost[period, count] = float(line[period + 4])
            else:
                commodity_no = count - no_of_arcs
                for period in range(data.periods):
                    origin, destination = int(line[0]), int(line[1])
                    data.demand[period, commodity_no] = float(line[period + 2])
                    data.od_pairs[commodity_no] = get_1d_index(origin, destination, data.nodes)
            count += 1
        origins, destinations = get_2d_index(data.od_pairs, data.nodes)
        origins, destinations = origins - 1, destinations - 1
        data.origins = origins.astype(int)
        data.destinations = destinations.astype(int)
        data.feasibility_cuts, data.optimality_cuts = 0, 0
        data.arc_org, data.arc_dest = get_2d_index(data.arcs, data.nodes)
        data.arc_org, data.arc_dest = data.arc_org - 1, data.arc_dest - 1
        data.b = np.zeros(shape=(data.commodities, data.nodes))
        for com in range(data.commodities):
            data.b[com, data.origins[com]] = -1.
            data.b[com, data.destinations[com]] = 1.
        return data


def get_2d_index(indx, width2):
    """
    Accepts array indexes assuming 1 dimensional indexing
    :param indx:
    :param width2:
    :return: returns again 1 dimensional indexing
    """
    rows = indx / width2
    columns = indx % width2
    rows += 1
    columns += 1
    return rows, columns


def get_1d_index(idx1, idx2, width2, idx3=0, width3=1):
    """
    Takes indexing that starts from 1, and returns an index on a zero-based array
    """
    idx1 -= 1
    idx2 -= 1
    return int(idx1 * width2 * width3 + idx2 * width3 + idx3)


def set_1d_index(array, value, idx1, idx2, width2):
    idx = get_1d_index(idx1, idx2, width2)
    array[idx] = value
