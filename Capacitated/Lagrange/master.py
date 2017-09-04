__author__ = 'ioannis'
"""
Master problem formulation for the multi-period fixed charge multicommodity network design problem
Ioannis Fragkos, April 2015
"""
import gurobipy as grb
import numpy as np
from heuristic import heuristic_main
from helpers import get_2d_index, read_data, get_1d_index
import array


def make_master(data, heur_solution):
    """
    Populates the master model
    :param data:            problem data
    :param heur_solution:   initial heuristic solution that is to be added to the model
    :return:                master model
    """
    master_model = grb.Model("Master-Problem")
    master_model.params.OutputFlag = 0

    arcs, nodes, periods, commodities = data.arcs, data.nodes, data.periods, data.commodities

    master_model._arc_index = np.empty(shape=arcs.size, dtype=object)

    origins, destinations = get_2d_index(data.od_pairs, nodes)

    heuristic = np.empty(shape=arcs.size, dtype=object)
    heur_coeffs = heur_solution.flow

    for arc in xrange(arcs.size):
        master_model._arc_index[arc] = array.array('i')
        heuristic[arc] = master_model.addVar(lb=0.0, ub=1.0, obj=heur_solution.objective,
                                             vtype=grb.GRB.CONTINUOUS, name='heur_var{}'.format(arc))
    master_model.update()

    count = 0

    for node in xrange(nodes):
        in_arcs = get_2d_index(arcs, nodes)[1] == node + 1
        out_arcs = get_2d_index(arcs, nodes)[0] == node + 1
        for commodity in xrange(commodities):
            for period in xrange(periods):
                rhs = 0.
                if node + 1 == origins[commodity]:
                    rhs = 1.
                if node + 1 == destinations[commodity]:
                    rhs = -1.
                lhs = grb.quicksum(heur_coeffs[period, commodity, out_arcs] * heuristic[out_arcs]) - \
                      grb.quicksum(heur_coeffs[period, commodity, in_arcs] * heuristic[in_arcs])
                master_model.addConstr(lhs=lhs, sense=grb.GRB.GREATER_EQUAL, rhs=rhs,
                                       name='p-n-c{},{},{}'.format(period + 1, node + 1, commodity + 1))
                for arc in np.where(out_arcs)[0]:
                    master_model._arc_index[arc].append(count)
                for arc in np.where(in_arcs)[0]:
                    master_model._arc_index[arc].append(-count)
                count += 1
    for arc in xrange(arcs.size):
        master_model.addConstr(heuristic[arc] <= 1., name='convexity_{}'.format(arc))

    master_model._arc_open = np.zeros(shape=(periods, arcs.size), dtype=float)
    master_model._convex_duals = np.zeros_like(arcs, dtype=float)
    master_model._node_duals = np.zeros(shape=(nodes, commodities, periods))
    master_model.update()

    return master_model


def add_variables(model, variables, data):
    """
    Adds a variable to the existing master model
    :param model:       Master model object
    :param variables:    A deque with variable objects (named tuples that hold: arc, no, objective, flow)
    :return:            Nothing
    """

    nodes, periods, commodities, arcs = data.nodes, data.periods, data.commodities, data.arcs
    constraints = model.getConstrs()

    for count_arc, arc in enumerate(data.arcs):
        for count_col, variable in enumerate(variables[count_arc]):
            no, obj = model.numvars + count_arc + count_col, variable.objective
            flow = variable.flow
            coefficients = np.zeros(shape=2 * np.sum(flow > 10e-6) + 1, dtype=float)
            node_in, node_out = get_2d_index(arc, nodes)
            constrs = []

            for c in xrange(commodities):
                for t in xrange(periods):
                    # node_out goes first
                    if flow[t, c] > 1e-6:
                        idx = len(constrs)
                        coefficients[idx] = -flow[t, c]
                        idx = len(constrs) + 1
                        coefficients[idx] = +flow[t, c]
                        # node_out index of constraint of (c,t)
                        idx = get_1d_index(idx1=node_out, idx2=c + 1, idx3=t, width2=commodities, width3=periods)
                        constrs.append(constraints[idx])
                        idx = get_1d_index(idx1=node_in, idx2=c + 1, idx3=t, width2=commodities, width3=periods)
                        constrs.append(constraints[idx])

            idx = nodes * periods * commodities + count_arc
            coefficients[len(constrs)] = 1.
            constrs.append(constraints[idx])

            column = grb.Column(coefficients, constrs)
            model.addVar(lb=0., ub=1., obj=obj, column=column, name='col_{}_{}'.format(count_arc, no))

    model.update()
    # model.write('master_first.lp')


def optimize(model, data):
    """
    Optimizes the master model and returns the dual prices
    :param model:   gurobi model
    :return:        adds to the model the dual prices of all the constraints: convex_duals and
    """

    periods, nodes, commodities = data.periods, data.nodes, data.commodities
    node_constraints = model._node_duals.size
    node_duals = np.zeros(shape=node_constraints)

    model.optimize()

    for constr in xrange(model.numconstrs):
        if constr < node_constraints:
            node_duals[constr] = model.__constrs[constr].Pi
        else:
            model._convex_duals[constr - node_constraints] = model.__constrs[constr].Pi

    model._node_duals = node_duals.reshape(nodes, commodities, periods)

    # positive_vars = (x for x in model.getVars() if x.X > 0)

    # for var in positive_vars:
    #     print '{}: {}'.format(var.VarName, var.X)
    # model.write('master_first.mps')


def test():
    data = read_data('c33_R_H_5.dow')
    heur_solution = heuristic_main(data=data, return_primal=True, track_time=True)[2]
    master_model = make_master(data=data, heur_solution=heur_solution)


if __name__ == '__main__':
    test()