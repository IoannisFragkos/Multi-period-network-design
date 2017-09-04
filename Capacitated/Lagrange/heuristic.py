"""
Heuristic solution for the Multi-period Network Design problem. Detailed description to follow
April 2015, Ioannis Fragkos
"""

from helpers import get_2d_index, read_data
from collections import namedtuple
from itertools import product
import numpy as np
import gurobipy as grb
import time

DEBUG = False


def solve_shortest_path(data, commodity, cost, model=None):
    """
    Solves a capacitated shortest path problem for a single commodity and a single period
    Network solver of Gurobi is used. If a model is already defined, we just update the cost coefficients and the
    source-destination constraints. Otherwise, the model is populated from scratch.

    April 2015, Ioannis Fragkos

    :param data:    Problem data: commodities, nodes, arcs, od_pairs,  periods, capacity, fixed_cost, variable_cost,
                    demand
    :param cost:    either variable cost or the sum of fixed and variable cost
    :return:        Model object, primal solution
    """

    nodes, arcs = data.nodes, data.arcs
    origin, destination = get_2d_index(data.od_pairs[commodity], nodes)
    counter = 0

    flow, flow_solution, dual_solution = np.empty_like(arcs, dtype=object), np.empty_like(arcs, dtype=float), \
                                         np.array(xrange(nodes), dtype=float)

    if model:
        variables, constraints = model.getVars(), model.getConstrs()
        new_cost = cost.reshape(arcs.size).tolist()
        new_objective = grb.LinExpr(new_cost, variables)
        model.setObjective(new_objective)
        origin, destination = get_2d_index(data.od_pairs[commodity], nodes)
        if constraints[origin - 1].RHS != 1.0:
            constraints[origin - 1].setAttr('rhs', 1.0)
            constraints[destination - 1].setAttr('rhs', -1.0)
    else:
        model = grb.Model('shortest_path')
        model.setParam('OutputFlag', 0)

        for arc in arcs:
            arc_from, arc_to = get_2d_index(arc, nodes)
            flow[counter] = model.addVar(lb=0.0, ub=1.0, obj=cost[counter], vtype=grb.GRB.CONTINUOUS,
                                         name='x({},{})'.format(arc_from, arc_to))
            counter += 1
        model.update()
        for node in xrange(nodes):
            rhs = 0.
            if node + 1 == origin:
                rhs = 1.
            if node + 1 == destination:
                rhs = -1.
            in_arcs = get_2d_index(arcs, nodes)[1] == node + 1
            out_arcs = get_2d_index(arcs, nodes)[0] == node + 1
            lhs = grb.quicksum(flow[out_arcs]) - grb.quicksum(flow[in_arcs])
            model.addConstr(lhs=lhs, sense=grb.GRB.EQUAL, rhs=rhs, name='node_{}'.format(node + 1))

    model.update()
    model.optimize()

    # Collect primal and dual solutions
    for arc in xrange(arcs.size):
        arc_from, arc_to = get_2d_index(arcs[arc], nodes)
        flow_solution[arc] = model.getVarByName('x({},{})'.format(arc_from, arc_to)).x
        if DEBUG and flow_solution[arc] > 0.01:
            print 'Flow: {} Arc: ({},{})'.format(flow_solution[arc], arc_from, arc_to, nodes)
    constraints = model.getConstrs()
    for node in xrange(nodes):
        dual_solution[node] = constraints[node].Pi
        if DEBUG and dual_solution[node] > 0.01:
            print 'Dual of node {}: {}'.format(node + 1, dual_solution[node])

    return model, flow_solution, dual_solution


def reset_model(origin, destination, model):
    """
    Resets the flow constraints of the model
    """
    constraints = model.getConstrs()
    constraints[origin - 1].setAttr('rhs', 0.)
    constraints[destination - 1].setAttr('rhs', 0.)


def test():
    time_start = time.time()
    global DEBUG
    DEBUG = False
    data = read_data('c38_R_H_5.dow')
    heuristic_main(data, return_primal=True)
    time_finish = time.time()
    print 'Total time: {} s'.format(time_finish - time_start)


def heuristic_main(data, return_primal=False, track_time=False, pi_only=False):
    if track_time:
        start = time.time()
    model = None
    arc_popularity = np.zeros(shape=(data.periods, data.arcs.size), dtype=float)
    network_duals = np.empty(shape=(data.periods, data.nodes, data.commodities), dtype=float)
    demand = data.demand
    period_demand = demand.sum(axis=1)
    commodity_demand = demand.sum(axis=0)
    for commodity in xrange(data.commodities):
        origin, destination = get_2d_index(data.od_pairs[commodity], data.nodes)
        model, flow, node_duals = solve_shortest_path(data, cost=data.variable_cost, commodity=commodity, model=model)
        arc_popularity[:, np.nonzero(flow)] += 1  * commodity_demand[commodity] / demand.sum()
        for period in xrange(data.periods):
            period_cost = data.variable_cost + data.fixed_cost[period, :] / (data.commodities * 0.1)
            model, flow, node_duals_period = solve_shortest_path(data, cost=period_cost, commodity=commodity,
                                                                 model=model)
            arc_popularity[period, np.nonzero(flow)] += 1 * demand[period, commodity] / period_demand[period]
            network_duals[period, :, commodity] = (node_duals + node_duals_period) / 2
        reset_model(origin, destination, model)
    if pi_only:
        return network_duals
    # collect for each arc the period that maximizes the frequency of the arc in the solutions
    max_periods = np.empty_like(data.arcs, dtype=int)
    max_periods[np.where(arc_popularity.max(axis=0) > 0)] = \
        arc_popularity.argmax(axis=0)[np.where(arc_popularity.max(axis=0) > 0)]
    max_periods[np.where(arc_popularity.max(axis=0) <= 0)] = -1

    if return_primal:
        objective, primal_solution, model = solve_reduced_problem(data=data, fixed=max_periods, return_primal=True)
        print 'objective: {}'.format(objective)
        if track_time:
            stop = time.time()
            print 'Heuristic time: {} seconds'.format(stop - start)
        return objective, network_duals, primal_solution, model
    else:
        objective, model = solve_reduced_problem(data=data, fixed=max_periods, return_primal=False)
        print 'objective: {}'.format(objective)
        if track_time:
            stop = time.time()
            print 'Heuristic time: {} seconds'.format(stop - start)
        return objective, network_duals, model


def solve_reduced_problem(data, fixed, model=None, return_primal=False, node_limit=1000, track_time=False):
    """
    Solves a multi-period capacitated network design problem where some arcs are fixed open. If an arc is fixed in
    period t and t is the first period that it is fixed, the arc can open any time in period t or after.
    :param data:    problem data
    :param fixed:   arcs/periods that are fixed to 1
    :return:        solution value (perhaps we should extend it to return the solution)
    """

    if track_time:
        start = time.time()

    commodities, arcs, capacity, variable_cost, fixed_cost, nodes, demand, periods = \
        data.commodities, data.arcs, data.capacity, data.variable_cost, data.fixed_cost, data.nodes, data.demand, \
        data.periods
    origins, destinations = get_2d_index(arcs, nodes)

    if not model:

        model = grb.Model('reduced-problem')
        model.setParam('OutputFlag', 1)
        model.setParam("TimeLimit", 100.)
        model.setParam("Threads", 2)
        model.setParam("NodeLimit", node_limit)
        model.setParam('MIPGap', 0.01)
        model.setParam("Heuristics", 1.0)

        model._flow, model._arc_open = np.empty(shape=(periods, commodities, arcs.size), dtype=object), \
                     np.empty(shape=(periods, arcs.size), dtype=object)
        flow, arc_open = model._flow, model._arc_open

        for t in xrange(periods):
            for arc in xrange(arcs.size):
                i, j = origins[arc], destinations[arc]
                arc_open[t, arc] = model.addVar(vtype=grb.GRB.BINARY, obj=fixed_cost[t, arc],
                                                name='open_({},{})_t{}'.format(i, j, t + 1))
                # if fixed[arc] <= t: maybe do this later
                for h in xrange(commodities):
                    flow[t, h, arc] = model.addVar(lb=0., ub=min(1., capacity[arc] / demand[t, h]),
                                                   obj=variable_cost[arc] * demand[t, h],
                                                   name='flow_c{0:d}_({1:d},{2:d})_t{3:d}'.format(h + 1, i, j, t + 1))
        model.update()
        # Arc capacity constraints
        for arc, t in product(xrange(arcs.size), xrange(periods)):
            i, j = origins[arc], destinations[arc]
            model.addConstr(grb.quicksum(grb.LinExpr(demand[t, h], flow[t, h, arc]) for h in xrange(commodities)) <=
                            capacity[arc] * grb.quicksum(arc_open[s, arc] for s in xrange(t + 1)),
                            'cap_({0:d},{1:d})_t{2:d}'.format(i, j, t + 1))
            lhs, rhs = grb.quicksum(arc_open[l, arc] for l in xrange(0, t + 1)), 1.
            name = 'unique_setup({0:d},{1:d})_t{2:d}'.format(i, j, t + 1)
            sign = grb.GRB.LESS_EQUAL  if fixed[arc] <= t + 4 else grb.GRB.EQUAL
            model.addConstr(lhs=lhs, rhs=rhs, name=name, sense=sign)
        # Flow conservation constraints
        for commodity in xrange(commodities):
            origin, destination = get_2d_index(data.od_pairs[commodity], nodes)
            for node in xrange(nodes):
                rhs = 0.
                if node + 1 == origin:
                    rhs = 1.
                if node + 1 == destination:
                    rhs = -1.
                in_arcs = get_2d_index(arcs, nodes)[1] == node + 1
                out_arcs = get_2d_index(arcs, nodes)[0] == node + 1
                for period in xrange(periods):
                    lhs = grb.quicksum(flow[period, commodity, out_arcs]) - grb.quicksum(
                        flow[period, commodity, in_arcs])
                    model.addConstr(lhs=lhs, sense=grb.GRB.EQUAL, rhs=rhs,
                                    name='node_{}_c{}_t{}'.format(node + 1, commodity + 1, period + 1))

        model.update()
    else:
        model.setParam("Nodes", 100.)
        model.setParam("TimeLimit", 100.)
        # Update arc capacity constraints, model is already populated
        for arc, t in product(xrange(arcs.size), xrange(periods)):
            i, j = origins[arc], destinations[arc]
            name = 'unique_setup({0:d},{1:d})_t{2:d}'.format(i, j, t + 1)
            constraint = model.getConstrByName(name)
            sign = grb.GRB.LESS_EQUAL if fixed[arc] <= t + 1 else grb.GRB.EQUAL
            constraint.setAttr("Sense", sign)

    model.optimize()
    if DEBUG:
        model.write('trial.lp')
        for var in model.getVars():
            if str(var.VarName[0]) == 'f' and var.X > 0.001:
                name = var.VarName.split('_')
                print 'Arc: \t {} \t Commodity: {} \t Period: {} \t Value: \t {}'.format(
                    name[2], int(name[1].replace('c', '')), int(name[3][1]),
                    var.X * demand[int(name[3][1]) - 1, int(name[1].replace('c', '')) - 1])

    if return_primal:
        flow, arc_open = model._flow, model._arc_open
        primal_solution = namedtuple('Solution', 'objective flow arc_open')
        primal_solution.flow, primal_solution.arc_open = np.zeros_like(flow, dtype=float), \
                                                         np.ones(shape=arcs.size, dtype=int) * (-1)

        print 'Problem status: {}'.format(model.status)

        # if model.status == grb.GRB.status.TIME_LIMIT:

        for arc in xrange(arcs.size):
            collect_flow = False
            for period in xrange(periods):
                if model._arc_open[period, arc].X > 0.5:
                    primal_solution.arc_open[arc] = period
                    collect_flow = True
                if collect_flow:
                    for commodity in xrange(commodities):
                        if flow[period, commodity, arc].X > 10e-5:
                            primal_solution.flow[period, commodity, arc] = flow[period, commodity, arc].X
        primal_solution.objective = model.getObjective().getValue()
        return primal_solution.objective, primal_solution, model
    if track_time:
        stop = time.time()
        print 'Heuristic time: {} seconds'.format(stop - start)

    return model.getObjective().getValue(), model


if __name__ == '__main__':
    test()