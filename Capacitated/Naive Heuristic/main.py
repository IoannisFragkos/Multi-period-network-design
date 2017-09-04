# -*- coding: utf-8 -*-
from __future__ import division
"""
Naive heuristic solution for the Multi-period Network Design problem
April 2017, Ioannis Fragkos
"""
from helpers import get_2d_index, read_data
from sys import argv
import numpy as np
import gurobipy as grb
import time
import platform


DEBUG = False


# @profile
def heuristic_main(data):
    """
    Naive heuristic
    Solves a single-period capacitated problem. Fixed arcs and iterates through periods
    """
    arcs, nodes, periods = data.arcs.size, data.nodes, data.periods
    commodities = data.commodities
    arc_origins, arc_destinations = get_2d_index(data.arcs, nodes)

    model = make_model(data)

    open_arcs = np.zeros((periods, arcs), dtype=np.double)

    objective = 0.

    for t in xrange(periods):
        fixed_cost, variable_cost = data.fixed_cost[t, :], data.variable_cost
        demand = data.demand[t, :]
        flow, arc_open = model._flow, model._arc_open

        for arc in xrange(arcs):
            i, j = arc_origins[arc], arc_destinations[arc]
            con_name = 'cap_{}-{}'.format(i, j)
            con = model.getConstrByName(con_name)
            # We pay for arcs that are not already open
            if arc_open[arc].lb == 0:
                arc_open[arc].obj = fixed_cost[arc]
            for c in xrange(commodities):
                model.chgCoeff(con, flow[c, arc], demand[c])
                flow[c, arc].obj = variable_cost[arc] * demand[c]
        model.optimize()
        if model.status == grb.GRB.status.INFEASIBLE:
            model.computeIIS()
            print 'model is infeasible'
            model.write(str(model.ModelName) + '_{}.ilp'.format(t))
        if model.SolCount > 0:
            objective += model.objVal
            # If we use an arc and it has not been opened before, we should
            # mark it as open now, and keep it open all along
            for count, var in enumerate(arc_open):
                if var.X > 0.1:
                    var.lb = 1.
                    var.obj = 0.
                    if np.sum(open_arcs[:t, count]) < 10e-5:
                        open_arcs[t, count] = 1.
            print 'Period : {} Objective value: {}'.format(t, objective)

    return objective, open_arcs


def make_model(data):
    """
    Formulates the regular (single period) MCND problem
    """
    commodities, arcs, capacity, variable_cost, nodes, demand = \
        data.commodities, data.arcs.size, data.capacity, \
        data.variable_cost, data.nodes, np.amax(data.demand, axis=0)
    origins, destinations = data.origins, data.destinations
    fixed_cost = data.fixed_cost[0, :]

    flow = np.empty(shape=(commodities, arcs), dtype=object)
    arc_open = np.empty(shape=arcs, dtype=object)
    capacities = np.empty(shape=arcs, dtype=object)
    arc_origins, arc_destinations = get_2d_index(data.arcs, nodes)


    model = grb.Model('MCND')
    model.params.threads = 1
    model.params.LogFile = ""
    lazy_cons = []

    # Variables
    for arc in xrange(arcs):
        i, j = arc_origins[arc], arc_destinations[arc]
        arc_open[arc] = model.addVar(
            vtype=grb.GRB.BINARY, obj=fixed_cost[arc],
            name='open_arc{}-{}'.format(i, j))
        for h in xrange(commodities):
            flow[h, arc] = model.addVar(
                obj=variable_cost[arc] * data.demand[0, h],
                lb=0., ub=1.,
                vtype=grb.GRB.CONTINUOUS,
                name='flow{}.{},{}'.format(h, i, j))
    model._arc_open = arc_open
    model._flow = flow
    model.update()

    # Constraints
    for arc in xrange(arcs):
        for h in xrange(commodities):
            lazy_cons.append(model.addConstr(flow[h, arc] <= arc_open[arc]))
        i, j = arc_origins[arc], arc_destinations[arc]
        capacities[arc] = model.addConstr(grb.quicksum(grb.LinExpr(
            data.demand[0, h] * flow[h, arc]) for h in xrange(
            commodities)) <= capacity[arc] * arc_open[arc],
            name='cap_{}-{}'.format(i, j))
    for h in xrange(commodities):
        for n in xrange(nodes):
                rhs = 0.
                if n == origins[h]:
                    rhs = 1.
                if n == destinations[h]:
                    rhs = -1.
                in_arcs = get_2d_index(data.arcs, nodes)[1] == n + 1
                out_arcs = get_2d_index(data.arcs, nodes)[0] == n + 1
                lhs = grb.quicksum(flow[h, out_arcs]) - grb.quicksum(
                    flow[h, in_arcs])
                model.addConstr(
                    lhs=lhs, rhs=rhs, sense=grb.GRB.EQUAL,
                    name='demand_n{}c{}'.format(n, h))

    model._capacities = capacities
    model.setParam('OutputFlag', 0)
    model.params.TimeLimit = 1000.
    model.params.MIPGap = 0.01
    model.params.Threads = 1
    model.update()
    for con in lazy_cons:
        con.Lazy = 3
    # model.write(str(model.ModelName) + '.lp')
    return model


def print_model_status(model):
	"""
	Prints out why the optimization status of model
	"""
	status = model.status
	if status == 2:
		grb_message = 'Model solved to optimality'
	if status == 8:
		grb_message = 'Model hits node limit'
	if status == 9:
		grb_message = 'Model hits time limit'
	try:
		print grb_message
	except NameError:
		print 'unknown model exist status code: {}'.format(status)


def test():
    time_start = time.time()
    global DEBUG
    DEBUG = False
    root_path = './'
    r_trial = 'r03.1_R_H_20.dow'
    filename = root_path 
    filename += r_trial if len(argv) <= 1 else argv[1]
    filename = argv[1] if platform.system() == 'Linux' else filename
    data = read_data(filename)
    # Uncomment for uncapacitated problems
    # data.capacity = np.array([10e+9] * len(data.capacity), dtype=float)
    objective, open_arcs = heuristic_main(data)
    print 'objective: {}'.format(objective)
    time_finish = time.time()
    print 'Total time: {} s'.format(time_finish - time_start)


if __name__ == '__main__':
    test()
