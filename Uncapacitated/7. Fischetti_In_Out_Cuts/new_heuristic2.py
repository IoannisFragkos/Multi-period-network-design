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
import platform


DEBUG = False

# @profile
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
        weights = np.cumsum(data.demand.sum(axis=1)[::-1])[::-1]
        weights /= weights[0]
        weights /= weights.sum()
        # weights[0], weights[1], weights[2:] = 0.7, 0.2, 0.1 / (periods - 2)
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
    print_model_status(model)

    potential_arcs = set()
    first_arcs = model._arc_open
    for arc in xrange(data.arcs.size):
        if first_arcs[arc].x > 0.001:
            potential_arcs.add(str(first_arcs[arc].VarName))

    # Opens all arcs, takes max demand per commodity
    modify_model(model, data)
    model.optimize()
    print_model_status(model)
    # model.NumObj = 1
    all_arcs = model._arc_open
    flow = model._flow

    # Close arcs that were not opened in the single-shot problem
    if model.SolCount > 0:
        for arc, var in enumerate(all_arcs):
            var.lb = 0.
            arc_flow = sum(
                [flow[h, arc].x for h in xrange(data.commodities)])
            if arc_flow > 10e-2:
                potential_arcs.add(str(var.VarName))
            else:
                var.ub = 0
    # model.update()
    # With this set of arcs (potential_arcs) solve single-period problems with
    # modified demand (so that it takes into account demand of future periods).
    # Do this to keep the arc opening variables only
    alpha = 1.
    for t in xrange(periods):
        t_max = min(t + 1, data.periods - 1)
        fixed_cost = weights[t] * data.fixed_cost[t, :] + \
        (1 - weights[t]) * np.average(data.fixed_cost[t_max:, :], axis=0)
        variable_cost = data.variable_cost * \
            np.max(data.demand[t, :]) / np.average(data.demand[t, :])
        demand = data.demand[t, :]
        arc_open = model._arc_open
        # if t == 0:
        #     for var in model._flow.flatten():
        #         var.vtype = grb.GRB.BINARY
        # if t == 1:
        #     for var in model._flow.flatten():
        #         var.vtype = grb.GRB.CONTINUOUS
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
        model.optimize()
        # In case we found no solution we solve the relaxation and round the solution..
        if model.SolCount > 0:
            for arc in xrange(arcs):
                is_open = np.sum(open_arcs[:t, arc]) > 10e-3
                if arc_open[arc].x > 0.01 or is_open:
                    lower_bounds[t, arc] = 1.
                    if not is_open:
                        open_arcs[t, arc] = 1
                else:
                    upper_bounds[t, arc] = 0
        else:
            print 'using root relaxation for period {}'.format(t)
            model.reset()
            model.optimize(heur_callback)
            arc_rel = model._arc_open_var_rel
            for arc in xrange(arcs):
                is_open = np.sum(open_arcs[:t, arc]) > 0.1
                if arc_rel[arc] > 0.001 or is_open:
                    lower_bounds[t, arc] = 1.
                    if not is_open:
                        open_arcs[t, arc] = 1
                # else:
                #     upper_bounds[t, arc] = 0
        
        alpha -= 1. / periods

    # Solve single-period problems, select among arcs that were open at the
    # initial problem. We can use arcs that opened in previous periods for free
    objective = 0.
    flow_cost = np.zeros(shape=(periods, commodities), dtype=float)
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
        # model.update()
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
                for c in xrange(commodities):
                    if flow[c, count].X > 0.0001:
                        flow_cost[t, c] += flow[c, count].X * \
                            flow[c, count].obj
            print 'Period : {} Objective value: {}'.format(t, objective)

    return objective, open_arcs, flow_cost


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
    demand_factor = np.zeros_like(data.origins)

    model = grb.Model('MCND')
    lazy_cons = []

    for arc in xrange(arcs):
        i, j = arc_origins[arc], arc_destinations[arc]
        arc_open[arc] = model.addVar(
            vtype=grb.GRB.BINARY, obj=fixed_cost[arc],
            name='open_arc{}-{}'.format(i, j))
        for h in xrange(commodities):
            # Multiply flow cost with the demand factor of each period            
            demand_factor[h] = demand[h] / np.average(data.demand[:, h])
            flow[h, arc] = model.addVar(
                obj=variable_cost[arc] * data.demand[0, h] * demand_factor[h],
                # obj= variable_cost[arc] * demand[h],
                lb=0., ub=1.,
                vtype=grb.GRB.CONTINUOUS,
                name='flow{}.{},{}'.format(h, i, j))
    model._arc_open = arc_open
    model._arc_open_var_rel = np.zeros_like(arc_open, dtype=np.double)
    model._flow = flow
    # model.update()

    for arc in xrange(arcs):
        for h in xrange(commodities):
            lazy_cons.append(model.addConstr(flow[h, arc] <= arc_open[arc]))
        i, j = arc_origins[arc], arc_destinations[arc]
        capacities[arc] = model.addConstr(grb.quicksum(grb.LinExpr(
            data.demand[0, h] * demand_factor[h],
            flow[h, arc]) for h in xrange(
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
    # model.params.BarConvTol = .1
    model.params.NodeLimit = 5000
    model.params.TimeLimit = 100
    model.params.MIPGap = 0.01
    model.params.Threads = 2
    model.setAttr('Lazy', lazy_cons, [3.]*len(lazy_cons))

    # model.write(str(model.ModelName) + '.lp')
    model.update()
    return model


def modify_model(model, data):
    """
    Modifies the existing single-period model so that it solves a bigger
    version of it, assuming all arcs are open
    """
    # Demand of each commodity is maximum across its period
    # max_t = data.demand.sum(axis=1).argmax()
    # demand = data.demand[max_t, :]
    demand = np.amax(data.demand, axis=0)
    arc_open = model._arc_open
    capacities = model._capacities
    flow = model._flow
    # model.NumObj = 2
    # model.Params.ObjNumber = 1
    for arc in xrange(data.arcs.size):
        arc_var = arc_open[arc]
        arc_var.lb = 1.
        arc_var.Obj = 0.
        # arc_var.Obj = data.fixed_cost[0, arc]
        for h in xrange(data.commodities):
            flow_var = flow[h, arc]
            flow_var.Obj = data.variable_cost[arc] * demand[h]
            # flow_var.Obj, flow_var.ObjN = 0., data.variable_cost[arc]*demand[h]
            model.chgCoeff(capacities[arc], flow_var, demand[h])
    # model.update()
    # model.write(str(model.ModelName) + '.lp')


def make_local_branching_model(data, open_arcs, cutoff):
    """
    Constructs a local branching model that searches a kappa-sized
    neighborhood per period, starting from the feasible solution open_arcs
    :param data:        Problem data
    :param open_arcs:   binary solution that defined the neighborhood
    :param cutoff:      cutoff value for feasible solutions
    """
    commodities, arcs, capacity, variable_cost, nodes, demand = \
        data.commodities, data.arcs.size, data.capacity, \
        data.variable_cost, data.nodes, data.demand
    origins, destinations = data.origins, data.destinations
    periods, fixed_cost = data.periods, data.fixed_cost

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
                    obj=variable_cost[arc] * demand[period, h],
                    lb=0., ub=min(1., capacity[arc] / demand[period, h]),
                    vtype=grb.GRB.CONTINUOUS,
                    name='flow{}.{},{}_{}'.format(h, i, j, period))
    model._arc_open = arc_open
    model._flow = flow
    # model.update()

    lazy_cons = []
    priority = [t for t in xrange(periods) for arc in xrange(arcs)]

    for period in xrange(periods):
        for arc in xrange(arcs):
            # Add initial vector of binary variables previously found
            arc_open[period, arc].start = open_arcs[period, arc]
            i, j = arc_origins[arc], arc_destinations[arc]
            capacities[period, arc] = model.addConstr(
                grb.quicksum(grb.LinExpr(
                    demand[period, h], flow[period, h, arc]) for h in xrange(
                    commodities)) <= capacity[arc].tolist() *
                grb.quicksum(arc_open[t, arc] for t in xrange(period + 1)),
                name='cap_{}-{}_{}'.format(i, j, period))
            # if period <= data.periods / 2:
            #     for h in xrange(commodities):
            #         if open_arcs[:period + 1, arc].sum(axis=0) > 0.5:
            #             lazy_cons.append(model.addConstr(
            #                 flow[period, h, arc] <= grb.quicksum(
            #                     arc_open[:period + 1, arc])))
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
                    lhs = grb.quicksum(flow[t, h, out_arcs]) - grb.quicksum(
                        flow[t, h, in_arcs])
                    model.addConstr(
                        lhs=lhs, rhs=rhs, sense=grb.GRB.EQUAL,
                        name='demand_n{}c{}p{}'.format(n, h, t))
    weights = np.array([np.exp(-t) for t in xrange(periods)])
    weights = weights[::-1] / weights.sum()
    for arc in xrange(arcs):
        model.addSOS(
            type=grb.GRB.SOS_TYPE1, vars=arc_open[:, arc].tolist(),
            wts=weights.tolist())
        # model.addConstr(
        #     grb.quicksum(arc_open[t, arc] for t in xrange(periods)) <= 1,
        #     name='sum_{}'.format(arc))
    model.setAttr(
        'BranchPriority', arc_open.flatten().tolist(), priority)

    arcs_in_search = np.zeros(arcs)
    # for arc in xrange(arcs):
    #     if open_arcs[:, arc].sum() > 0.5:
    #         max_period = open_arcs[:, arc].argmax()
    #         if max_period < 3:
    #             end = min(periods, max_period + 9)
    #             lhs = grb.quicksum([
    #                 arc_open[period, arc] for period in xrange(
    #                     max_period, end)])
    #             model.addConstr(lhs >= 1, name='local_search.{}'.format(arc))
    #             arcs_in_search[arc] = 1

    model._capacities = capacities
    model._arcs_in_search = arcs_in_search
    model.params.LogFile = ""
    # model.params.presolve = 0
    # model.params.cuts = 0
    # model.setParam('OutputFlag', 0)
    model.params.TimeLimit = 50
    model.params.NodeLimit = 100
    model.params.MIPFocus = 1
    model.params.MIPGap = 0.01
    model.params.Threads = 2
    model.params.Cutoff = cutoff + 0.0001
    # model.params.ImproveStartTime = 200
    # model.params.normadjust = 3
    # for con in lazy_cons:
    #     con.Lazy = 3
    model.setAttr('Lazy', lazy_cons, [3]*len(lazy_cons))
    # model.update()
    # model.write('eyes.lp')
     # model.optimize()
    # print_model_status(model)
    # print 'solutions found: {}'.format(model.SolCount)
    # print 'best objective value: {}'.format(model.objVal)
    # n_sols = min(model.SolCount, 10)
    # solutions = np.zeros(shape=(n_sols, periods, arcs), dtype=int)
    # for sol in xrange(n_sols):
    #     model.setParam('SolutionNumber', sol)
    #     # print "--- Solution number: {} ---".format(sol)
    #     for period in xrange(periods):
    #         for arc in xrange(arcs):
    #             if arc_open[period, arc].Xn > 0:
    #                 solutions[sol, period, arc] = 1
                    # print 'Period: {} Arc: {}'.format(period, arc)

    # return model.ObjVal, model  # solutions[0]
    # model.write('local_branching.lp')
    return model


def solve_local_branching_model(data, open_arcs, lb_model, cutoff):
    """
    Solves the LB model of above
    :param data:
    :param open_arcs:
    :param lb_model:
    :param cutoff:
    :return:
    """

    arcs, periods = data.arcs.size, data.periods
    arcs_in_search = lb_model._arcs_in_search
    arc_open = lb_model._arc_open
    counter = 0
    for arc in xrange(arcs):
        if open_arcs[1:, arc].sum() > 0.5:
            max_period = open_arcs[:, arc].argmax()
            end = min(periods, max_period + 9)
            lhs = grb.quicksum([
                arc_open[period, arc] for period in xrange(max(
                    max_period - 4,0), end)])
            lb_model.addConstr(
                lhs >= 1, name='local_search.{}'.format(arc))
            counter += 1
            arcs_in_search[arc] = 1
        else:
            arcs_in_search[arc] = 0

    lb_model.setAttr(
        'Start', arc_open.flatten().tolist(), open_arcs.flatten().tolist())
    # fixed_vars = arc_open[8:, :].flatten().tolist()
    # fixed_vals = open_arcs[8:, :].flatten().tolist()
    # lb_model.setAttr('ub', fixed_vars, fixed_vals)
    # lb_model.setAttr('lb', fixed_vars, fixed_vals)

    lb_model.params.Cutoff = cutoff + 0.0001

    arcs_one = arc_open.flatten()[open_arcs.flatten().nonzero()[0]].tolist()
    arcs_zero = arc_open.flatten(
    )[np.where(open_arcs.flatten() == 0)[0]].tolist()


    expr = grb.LinExpr(
        [-1.] * len(arcs_one) + [1.] * len(arcs_zero), 
        arcs_one + arcs_zero)
    lb_model.addConstr(
        lhs=expr, rhs=20, sense=grb.GRB.LESS_EQUAL, name='lb')
    counter += 1

    lb_model.optimize()
    # lb_model.write('local_branching.lp')
    print_model_status(lb_model)
    if lb_model.SolCount > 0:
        if lb_model.objVal < cutoff:
            cutoff = lb_model.objVal
            open_arcs = np.array(
                lb_model.getAttr(
                    'x', arc_open.flatten().tolist())).reshape(
                periods, arcs)
            lb_model.setAttr(
                'Start', arc_open.flatten().tolist(), 
                open_arcs.flatten().tolist())
            lb_model._arcs_in_search = arcs_in_search

    constraints = lb_model.getConstrs()[-counter:]
    for con in constraints:
        lb_model.remove(con)

    return cutoff, open_arcs, lb_model


def solve_fixing_model(
    data, model, primal_sol, prop, cutoff, incumbent_arcs):
    """
    Solves a fixed model, where the prop% of variables closer to 
    0.5 are relaxed and all the rest remain fixed
    model:      model to be solved
    primal_sol: fractional primal solution
    prop:       proportion of the fractional solution that 
                remains unfixed
    cut
    """

    periods, arcs = data.periods, data.arcs.size
    num_vars = primal_sol.open_arc.size
    ub, lb = np.ones(num_vars), np.zeros(num_vars)
    int_vars = int(num_vars * prop)
    sol_1d = primal_sol.open_arc.ravel()
    idxs = np.argsort(np.abs(sol_1d - 0.5))[:int_vars]
    integers = sol_1d.take(idxs)
    lb_threshold = max(integers.min(), 0.1)
    ub_threshold = min(integers.max(), 0.7)
    # fix variables
    ub[sol_1d < lb_threshold] = 0.
    lb[sol_1d > ub_threshold] = 1.
    model.update()
    arc_vars = model._arc_open.ravel().tolist()
    model.setAttr('ub', arc_vars, ub.tolist())
    model.setAttr('lb', arc_vars, lb.tolist())
    model.params.Cutoff = cutoff
    model.optimize()
    if model.SolCount > 0:
        if model.obj_val < cutoff:
            cutoff = model.obj_val
            incumbent_arcs = np.array(
                model.getAttr('x', arc_vars)).reshape(
                periods, arcs)
            model.setAttr(
                'Start', arc_vars, incumbent_arcs.flatten().tolist())

    model.setAttr('ub', arc_vars, [1.]*len(arc_vars))
    model.setAttr('lb', arc_vars, [0.]*len(arc_vars))

    return cutoff, incumbent_arcs, model



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
        print 'unknown model exit status code: {}'.format(status)


def heur_callback(model, where):
    """
    Defines a callback that solves the problem in the root node and
    retrieves the lp values of the arc_open variables
    """
    if where == grb.GRB.callback.MIPNODE and model.cbGet(
        grb.GRB.Callback.MIPNODE_STATUS) == grb.GRB.OPTIMAL:
        # print 'we are here'
        node_count = int(model.cbGet(grb.GRB.callback.MIPNODE_NODCNT))
        if node_count == 0:
            model._arc_open_var_rel = model.cbGetNodeRel(model._arc_open)
        else:
            model.terminate()


def heuristic(data, kappa, mode='c', local_branching=False):
    time_start = time.time()
    if mode == 'u':
        data.capacity = np.array([10e+9] * len(data.capacity), dtype=float)
    objective, open_arcs, flow_cost = heuristic_main(data)
    print 'objective before Local Branching: {}'.format(objective)
    print 'time before Local Branching: {}'.format(time.time() - time_start)
    if local_branching:
        objective, model = make_local_branching_model(
            data, kappa, open_arcs, objective)
        print 'objective after Local Branching: {}'.format(objective)
        print 'time after Local Branching: {}'.format(time.time() - time_start)
        return objective, model  # open_arcs, flow_cost
    else:
        return objective, open_arcs, flow_cost


def test():
    time_start = time.time()
    global DEBUG
    DEBUG = False
    # root_path = '../../../DataDeterministicFC/'
    root_path = '../../../MPMCFP_DataGen/'
    data_path_c = 'c_Instances_Dec_Fixed_Cost/'
    data_path_r = 'r_Instances_Dec_Fixed_Cost/'
    c_trial, r_trial = 'c33_R_H_10.dow', 'r09.1_R_H_10.dow'
    filename = root_path + data_path_c
    filename += r_trial if len(argv) <= 1 else argv[1]
    filename = argv[1] if platform.system() == 'Linux' else filename
    data = read_data(filename)
    # data.capacity = np.array([10e+9] * len(data.capacity), dtype=float)
    objective, open_arcs, flow_cost = heuristic_main(data)
    time_finish = time.time()
    print 'Time before local branching: {} s'.format(time_finish - time_start)
    # objective, open_arcs = make_local_branching_model(
    #     data, 1, open_arcs, objective)
    print 'objective: {}'.format(objective)
    time_finish = time.time()
    print 'Total time: {} s'.format(time_finish - time_start)


if __name__ == '__main__':
    test()
