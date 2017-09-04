# -*- coding: utf-8 -*-
from __future__ import division
"""
Heuristic solution for the Multi-period Network Design problem. Detailed
description to follow
June 2015, Ioannis Fragkos
"""
from helpers import get_2d_index, read_data
from sys import argv
import numpy as np
import gurobipy as grb
import time

DEBUG = False


def heuristic_main(data):
    """
    Heuristic that implements the following logic.
    1. Formulate a single period instance
    2. Select the maximum demand for each commodity
    3. Multiply the variable costs by the ratio of (total demand) / (peak
        demand) of each commodity. This makes the variable costs commodity-
       dependent.
    3. Select a weighted average of costs of all periods
    4. Solve a single-period problem with these data. Store the arcs that
       opening a set (potential arcs)
    5. Select the period of total maximum demand and solve again a
       single-period problem, as before
    6. For each period t, solve a single-period problem, with the restrictions
       that (i) arcs that opened in previous periods remain open and (ii) only
       arcs from the set of potential arcs are allowed to open. Demand of each
       period should be replaced by a smoothed average of the next few periods
    7. Finally, solve the entire problem by fixing the binary variables across
       the horizon.
    :return     objective function of heuristic and values of arc variables
    """
    arcs, nodes, periods = data.arcs.size, data.nodes, data.periods
    commodities = data.commodities
    arc_origins, arc_destinations = get_2d_index(data.arcs, nodes)
    weights = np.empty(shape=periods, dtype=float)
    if periods > 1:
        weights[0], weights[1], weights[2:] = 0.5, 0.25, 0.25/(periods-2)
    else:
        weights[0], weights[1:] = 1, 0.
    fixed_cost = np.average(data.fixed_cost, axis=0, weights=weights)
    upper_bounds = np.ones(shape=(data.periods, data.arcs.size))
    lower_bounds = np.zeros_like(upper_bounds)

    # Store the arcs that are open in the single-shot problem
    open_arcs = np.zeros(shape=(periods, arcs), dtype=int)

    # Take into account the peak demand only initially
    model = make_model(data, fixed_cost)
    model.optimize()

    first_arcs = model._arc_open
    if model.status == grb.GRB.status.INFEASIBLE:
        print "Model is infeasible"
        potential_arcs = set([
            first_arcs[arc].VarName for arc in xrange(data.arcs.size)])
    else:
        potential_arcs = set()
        for arc in xrange(data.arcs.size):
            if first_arcs[arc].x > 0.001:
                potential_arcs.add(str(first_arcs[arc].VarName))

    # Opens all arcs, takes max demand per commodity
    modify_model(model, data)
    model.optimize()
    all_arcs = model._arc_open
    flow = model._flow

    # Close arcs that were not opened in the single-shot problem
    if model.SolCount > 0:
        for arc, var in enumerate(all_arcs):
            var.lb = 0.
            arc_flow = sum(
                [flow[h, arc].x for h in xrange(data.commodities)])
            if arc_flow > 10e-5:
                potential_arcs.add(str(var.VarName))
            else:
                var.ub = 0
    model.update()

    # With this set of arcs (potential_arcs) solve single-period problems with
    # modified demand (so that it takes into account demand of future periods).
    # Do this to keep the arc opening variables only
    alpha = 1.
    for t in xrange(periods):
        t_max = min(t+1, data.periods-1)
        fixed_cost = alpha * data.fixed_cost[t, :] + (1 - alpha) * np.average(
            data.fixed_cost[t_max:, :], axis=0)
        variable_cost = data.variable_cost * \
            np.max(data.demand[t, :]) / np.average(data.demand[t, :])
        demand = data.demand[t, :]
        arc_open = model._arc_open
        for arc in xrange(arcs):
            i, j = arc_origins[arc], arc_destinations[arc]
            con_name = 'cap_{}-{}'.format(i, j)
            con = model.getConstrByName(con_name)
            if arc_open[arc].varName not in potential_arcs:
                arc_open[arc].ub = 0.
            # We don't pay for arcs that are already open
            if np.sum(open_arcs[:t, arc], axis=0) > 0.5:
                arc_open[arc].obj = 0.
            else:
                arc_open[arc].obj = fixed_cost[arc]
            for c in xrange(commodities):
                model.chgCoeff(con, flow[c, arc], demand[c])
                flow[c, arc].obj = variable_cost[arc] * demand[c]
        model.update()
        model.optimize()
        for arc in xrange(arcs):
            if arc_open[arc].x > 0.01:
                lower_bounds[t, arc] = 1.
                if np.sum(open_arcs[:t, arc]) < 10e-5:
                    open_arcs[t, arc] = 1
            else:
                upper_bounds[t, arc] = 0
        alpha -= 1./periods

    # Solve single-period problems, select among arcs that were open at the
    # initial problem. We can use arcs that opened in previous periods for free
    objective = 0.
    for t in xrange(periods):
        fixed_cost, variable_cost = data.fixed_cost[t, :], data.variable_cost
        demand = data.demand[t, :]
        flow = model._flow
        arc_open = model._arc_open
        for arc in xrange(arcs):
            arc_open[arc].ub = upper_bounds[t, arc]
            arc_open[arc].lb = lower_bounds[t, arc]
            i, j = arc_origins[arc], arc_destinations[arc]
            con_name = 'cap_{}-{}'.format(i, j)
            con = model.getConstrByName(con_name)
            # We dont pay for arcs that are already open
            if np.sum(open_arcs[:t, arc], axis=0) > 0.5:
                arc_open[arc].obj = 0.
            else:
                arc_open[arc].obj = fixed_cost[arc]
            for c in xrange(commodities):
                model.chgCoeff(con, flow[c, arc], demand[c])
                flow[c, arc].obj = variable_cost[arc] * demand[c]
        model.update()
        model.optimize()
        if model.status == grb.GRB.status.INFEASIBLE:
            model.computeIIS()
            print 'model is infeasible'
            model.write(str(model.ModelName) + '_{}.ilp'.format(t))
        if model.SolCount > 0:
            objective += model.objVal
            # If we use an arc and it has not been opened before, we should
            # mark it as open now, and keep it open all along
            for count, var in enumerate(all_arcs):
                if var.X > 0.1:
                    var.lb = 1.
                    if np.sum(open_arcs[:t, count]) < 10e-5:
                        open_arcs[t, count] = 1.
            print 'Period : {} Objective value: {}'.format(t, objective)

    return objective, open_arcs


def make_model(data, fixed_cost):
    """
    Formulates the regular (single period) MCND problem
    """
    commodities, arcs, capacity, variable_cost, nodes, demand = \
        data.commodities, data.arcs.size, data.capacity, \
        data.variable_cost, data.nodes, np.amax(data.demand, axis=0)
    origins, destinations = data.origins, data.destinations

    flow = np.empty(shape=(commodities, arcs), dtype=object)
    arc_open = np.empty(shape=arcs, dtype=object)
    capacities = np.empty(shape=arcs, dtype=object)
    arc_origins, arc_destinations = get_2d_index(data.arcs, nodes)

    model = grb.Model('MCND')

    for arc in xrange(arcs):
        i, j = arc_origins[arc], arc_destinations[arc]
        arc_open[arc] = model.addVar(
            vtype=grb.GRB.BINARY, obj=fixed_cost[arc],
            name='open_arc{}-{}'.format(i, j))
        for h in xrange(commodities):
            # Multiply flow cost with the demand factor of each period
            demand_factor = np.sum(data.demand[:, h]) / demand[h]
            flow[h, arc] = model.addVar(
                obj=variable_cost[arc]*demand[h] * demand_factor,
                lb=0., ub=min(1., capacity[arc] / demand[h]),
                vtype=grb.GRB.CONTINUOUS,
                name='flow{}.{},{}'.format(h, i, j))
    model._arc_open = arc_open
    model._flow = flow
    model.update()

    for arc in xrange(arcs):
        i, j = arc_origins[arc], arc_destinations[arc]
        capacities[arc] = model.addConstr(grb.quicksum(grb.LinExpr(
            demand[h], flow[h, arc]) for h in xrange(
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
    model.params.BarConvTol = .1
    model.params.NodeLimit = 300
    model.params.TimeLimit = 100.
    model.params.Threads = 2
    model.update()
    # model.write(str(model.ModelName) + '.lp')
    return model


def modify_model(model, data):
    """
    Modifies the existing single-period model so that if solves a bigger
    version of it, assuming all arcs are open
    """
    # Demand of each commodity is maximum across its period
    demand = np.amax(data.demand, axis=0)
    arc_open = model._arc_open
    capacities = model._capacities
    flow = model._flow
    for arc in xrange(data.arcs.size):
        arc_var = arc_open[arc]
        arc_var.lb = 1.
        arc_var.Obj = 0.
        for h in xrange(data.commodities):
            flow_var = flow[h, arc]
            flow_var.Obj = data.variable_cost[arc] * demand[h]
            model.chgCoeff(capacities[arc], flow_var, demand[h])
    model.update()
    # model.write(str(model.ModelName) + '.lp')
    model.optimize()


def make_local_branching_model(data, kappa, open_arcs, cutoff, model=None):
    """
    Constructs a local branching model that searches a kappa-sized
    neighborhood per period, starting from the feasible solution open_arcs
    :param data:        Problem data
    :param kappa:       Local branching neighborhood (per period)
    :param open_arcs:   binary solution that defined the neighborhood
    :param cutoff:      cutoff value for feasible solutions
    :param model:         if given, there is a populated model
    """
    commodities, arcs, capacity, variable_cost, nodes, demand = \
        data.commodities, data.arcs.size, data.capacity, \
        data.variable_cost, data.nodes, data.demand
    origins, destinations = data.origins, data.destinations
    periods, fixed_cost = data.periods, data.fixed_cost

    if model is None:

        flow = np.empty(shape=(periods, commodities, arcs), dtype=object)
        arc_open = np.empty(shape=(periods, arcs), dtype=object)
        capacities = np.empty(shape=(periods, arcs), dtype=object)
        arc_origins, arc_destinations = get_2d_index(data.arcs, nodes)

        model = grb.Model('local_branching')

        for period in xrange(periods):
            for arc in xrange(arcs):
                i, j = arc_origins[arc], arc_destinations[arc]
                arc_open[period, arc] = model.addVar(
                    vtype=grb.GRB.BINARY, obj=fixed_cost[period, arc],
                    name='open_arc{}-{}_{}'.format(i, j, period))
                for h in xrange(commodities):
                    flow[period, h, arc] = model.addVar(
                        obj=variable_cost[arc]*demand[period, h],
                        lb=0., ub=min(1., capacity[arc] / demand[period, h]),
                        vtype=grb.GRB.CONTINUOUS,
                        name='flow{}.{},{}_{}'.format(h, i, j, period))
        model._arc_open = arc_open
        model._flow = flow
        model.update()

        for period in xrange(periods):
            for arc in xrange(arcs):
                # Add initial vector of binary variables previously found
                arc_open[period, arc].start = open_arcs[period, arc]
                i, j = arc_origins[arc], arc_destinations[arc]
                capacities[period, arc] = model.addConstr(
                    grb.quicksum(
                        grb.LinExpr(
                            demand[period, h], flow[period, h, arc]) for h in
                        xrange(commodities)) <= capacity[arc] *
                    grb.quicksum(arc_open[t, arc] for t in xrange(period+1)),
                    name='cap_{}-{}_{}'.format(i, j, period))
        for h in xrange(commodities):
            for n in xrange(nodes):
                    rhs = 0.
                    if n == origins[h]:
                        rhs = 1.
                    if n == destinations[h]:
                        rhs = -1.
                    in_arcs = get_2d_index(data.arcs, nodes)[1] == n + 1
                    out_arcs = get_2d_index(data.arcs, nodes)[0] == n + 1
                    for t in xrange(periods):
                        lhs = grb.quicksum(
                            flow[t, h, out_arcs]) - grb.quicksum(
                            flow[t, h, in_arcs])
                        model.addConstr(
                            lhs=lhs, rhs=rhs, sense=grb.GRB.EQUAL,
                            name='demand_n{}c{}p{}'.format(n, h, t))
        for arc in xrange(arcs):
            model.addConstr(
                grb.quicksum(arc_open[t, arc] for t in xrange(periods)) <= 1,
                name='sum_{}'.format(arc))

        # Local branching constraints go here
        for period in xrange(periods):
            lhs = grb.quicksum([
                    arc_open[period, arc] for arc in xrange(arcs) if
                    open_arcs[period, arc] == 0])
            lhs += grb.quicksum([
                    1. - arc_open[period, arc] for arc in xrange(arcs) if
                    open_arcs[period, arc] == 1])
            model.addConstr(
                lhs <= kappa, name='local_branch.{}'.format(period))

        model._capacities = capacities
        model._open_arcs_vals = open_arcs
        model._time = time.time()
        model.setParam('OutputFlag', 0)
        model.params.TimeLimit = 100.
        model.params.NodeLimit = 500
        model.params.MIPGap = 0.01
        model.params.Threads = 2
        model.params.Heuristics = 1.
        model.params.Cutoff = cutoff + 0.0001
    else:
        # model.reset()
        arc_open = model._arc_open
        for period in xrange(periods):
            constr_name = 'local_branch.{}'.format(period)
            constr = model.getConstrByName(constr_name)
            constr.rhs = kappa+np.sum(open_arcs[period, :])
            for arc in xrange(arcs):
                val = 1 if open_arcs[period, arc] < 0.001 else -1.
                model.chgCoeff(constr, arc_open[period, arc], val)
    model.update()
    model.optimize(local_branching_callback)
    # model.write('trial.lp')
    print 'solutions found: {}'.format(model.SolCount)
    # print 'best objective value: {}'.format(model.objVal)
    n_sols = min(model.SolCount, 10)
    solutions = np.zeros(shape=(n_sols, periods, arcs), dtype=int)
    for sol in xrange(n_sols):
        model.setParam('SolutionNumber', sol)
        # print "--- Solution number: {} ---".format(sol)
        for period in xrange(periods):
            for arc in xrange(arcs):
                if arc_open[period, arc].Xn > 0:
                    solutions[sol, period, arc] = 1
                    # print 'Period: {} Arc: {}'.format(period, arc)
    objective = model.ObjVal if n_sols else np.infty
    return objective, model


def local_branching_callback(model, where):
    if where == grb.GRB.callback.MIP:
        model._nodecnt = model.cbGet(grb.GRB.callback.MIP_NODCNT)
    elif where == grb.GRB.callback.MIPNODE and model._nodecnt == 0:
        model.cbSetSolution(
            model._arc_open.flatten(), model._open_arcs_vals.flatten())
    if time.time() - model._time > 3600:
        print 'Hit the time limit while local branching. Aborting...'
        model.terminate()


def heuristic(data, kappa):
    time_start = time.time()
    objective, open_arcs = heuristic_main(data)
    print 'objective before Local Branching: {}'.format(objective)
    print 'time before Local Branching: {}'.format(time.time() - time_start)
    objective, model = make_local_branching_model(
        data, kappa, open_arcs, objective)
    print 'objective after Local Branching: {}'.format(objective)
    print 'time after Local Branching: {}'.format(time.time() - time_start)
    return objective, model


def test():
    time_start = time.time()
    global DEBUG
    DEBUG = False
    filename = 'r01.1_R_H_10.dow' if len(argv) <= 1 else argv[1]
    data = read_data(filename)
    objective, open_arcs = heuristic_main(data)
    objective, model = make_local_branching_model(
        data, 2, open_arcs, objective)
    print 'objective: {}'.format(objective)
    time_finish = time.time()
    print 'Total time: {} s'.format(time_finish - time_start)


if __name__ == '__main__':
    test()
