from __future__ import division
from helpers import read_data, get_2d_index
from sys import argv
from time import time
from itertools import product
from networkx import shortest_path_length, shortest_path
from gurobipy import Model, GRB, LinExpr, quicksum
from graph_helpers import make_graph
from collections import namedtuple
from new_heuristic2 import heuristic
import numpy as np
import networkx as nx

__author__ = 'ioannis'
# File to gather data and Implement Benders Decomposition
# Ioannis Fragkos, June 2015

# Container that holds the subproblem dual vectors. The vectors should be
# entered as numpy arrays.
# Dimensions: flow_duals(node, commodity, period); bounds_duals(period, arc)
# We carry numpy arrays of Subproblem_Duals so that we add many cuts at once
Subproblem_Duals = namedtuple('Subproblem_Duals',
                              'flow_duals bounds_duals optimality_dual')
LOG_LEVEL = 0
EPSILON = .01


def main():
    filename = 'r03.1_R_H_20.dow' if len(argv) <= 1 \
        else argv[1]
    data = read_data(filename)
    start = time()
    data.graph = make_graph(data)
    # objective, open_arcs, flow_cost = heuristic(data, 2)
    # print 'Heuristic objective value: {}'.format(objective)
    subproblems = populate_dual_subproblem(data)
    master = populate_master(data)
    master_callback = callback_data(subproblems, data)
    master.optimize(master_callback)
    stop = time()
    print 'Total time: {} seconds'.format(round(stop - start, 0))


def populate_master(data, open_arcs=None):
    """
    Function that populates the Benders Master problem
    :param data:   Problem data structure
    :param open_arcs: If given, it is a MIP start feasible solution
    :rtype:        Gurobi model object
    """
    master = Model('master-model')
    arcs, periods = xrange(data.arcs.size), xrange(data.periods)
    commodities = xrange(data.commodities)
    graph, origins, destinations = data.graph, data.origins, data.destinations
    variables = np.empty(shape=(data.periods, data.arcs.size), dtype=object)
    bin_vars_idx = np.empty_like(variables, dtype=int)
    continuous_variables = np.empty(
        shape=(len(periods), len(commodities)), dtype=object)
    cont_vars_idx = np.empty_like(continuous_variables, dtype=int)

    start_given = open_arcs is not None
    count = 0

    # length of shortest path, shortest path itself
    arc_com, arc_obj = [], []
    lbs = [shortest_path_length(
        graph, origins[com], destinations[com], 'weight') for com in commodities]
    sps = [shortest_path(
        graph, origins[com], destinations[com], 'weight') for com in commodities]
    # resolve sp by removing one arc, check the increase in value
    for com in commodities:
        incr, best_arc = 0., 0
        for n1, n2 in zip(sps[com], sps[com][1:]):
            weight = graph[n1][n2]['weight']
            graph[n1][n2]['weight'] = 10000. * weight
            spl = shortest_path_length(
                graph, origins[com], destinations[com], 'weight')
            if spl > incr:
                 incr = spl
                 best_arc = graph[n1][n2]['arc_id']
            graph[n1][n2]['weight'] = weight
        arc_com.append(best_arc)
        arc_obj.append(spl)

    # Add variables
    for period in periods:
        for arc in arcs:
            # Binary arc variables
            variables[period, arc] = master.addVar(
                vtype=GRB.BINARY, obj=data.fixed_cost[period, arc],
                name='arc_open{}_{}'.format(period, arc))
            bin_vars_idx[period, arc] = count
            count += 1
        for com in commodities:
            lb = lbs[com] * data.demand[period, com]
            # Continuous flow_cost variables (eta)
            continuous_variables[period, com] = master.addVar(
                lb=lb, obj=1., vtype=GRB.CONTINUOUS, name='flow_cost{}'.format(
                    (period, com)))
            cont_vars_idx[period, com] = count
            count += 1
    master.update()

    # If feasible solution is given, use it as a start
    if start_given:
        for period in periods:
            for arc in arcs:
                # variables[period, arc].start = open_arcs[period, arc]
                variables[period, arc].VarHintVal = open_arcs[period, arc]
                variables[period, arc].VarHintPri = 1

    # Add constraints
    # Add Origin - Destination Cuts for each Commodity
    cuts_org, cuts_dest = set(), set()
    for commodity in commodities:
        arc_origin = data.origins[commodity]
        arc_destination = data.destinations[commodity]
        if arc_origin not in cuts_org:
            out_origin = get_2d_index(data.arcs, data.nodes)[0] - 1 == arc_origin
            master.addConstr(
                lhs=np.sum(variables[0, out_origin]), rhs=1.,
                sense=GRB.GREATER_EQUAL, name='origins_c{}'.format(commodity))
            cuts_org.add(arc_origin)
        if arc_destination not in cuts_dest:
            in_dest = get_2d_index(data.arcs, data.nodes)[1] - 1 == arc_destination
            master.addConstr(
                lhs=np.sum(variables[0, in_dest]), rhs=1.,
                sense=GRB.GREATER_EQUAL,
                name='destinations_c{}'.format(commodity))
            cuts_dest.add(arc_destination)


    # Add that an arc can open at most once
    for arc in arcs:
        master.addSOS(
            GRB.SOS_TYPE1, variables[:, arc].tolist(), list(periods)[::-1])

    # Add extra constraints for lower bound improvement
    for com in commodities:
        arc = arc_com[com]
        base_coeffs = lbs[com] - arc_obj[com]
        for period in periods:
            lhs = LinExpr()
            coeffs = [cf * data.demand[period, com]
                      for cf in [base_coeffs] * (period + 1)]
            lhs.addTerms(coeffs, variables[:period+1, arc].tolist())
            lhs.add(-continuous_variables[period, com])
            lhs.addConstant(arc_obj[com] * data.demand[period, com])
            master.addConstr(
                lhs, sense=GRB.LESS_EQUAL, rhs=0,
                name='strengthening_{}{}'.format(period, com))

    master.params.LazyConstraints = 1
    # Find feasible solutions quickly, works better
    master.params.TimeLimit = 7200
    master.params.threads = 2
    master.params.BranchDir = 1
    # Store the variables inside the model, we cannot access them later!
    master._variables = np.array(master.getVars())
    master._cont_vars_idx = cont_vars_idx
    master._bin_vars_idx = bin_vars_idx
    return master


def populate_dual_subproblem(data):
    """
    Function that populates the Benders Dual Subproblem, as suggested by the
    paper "Minimal Infeasible Subsystems and Bender's cuts" by Fischetti,
    Salvagnin and Zanette.
    :param data:        Problem data structure
    :param upper_cost:  Link setup decisions fixed in the master
    :param flow_cost:   This is the cost of the continuous variables of the
                        master problem, as explained in the paper
    :return:            Numpy array of Gurobi model objects
    """

    # Gurobi model objects
    subproblems = np.empty(
        shape=(data.periods, data.commodities), dtype=object)

    # Construct model for period/commodity 0.
    # Then, copy this and change the coefficients
    subproblem = Model('subproblem_(0,0)')

    # Ranges we are going to need
    arcs, periods, commodities, nodes = xrange(data.arcs.size), xrange(
        data.periods), xrange(data.commodities), xrange(data.nodes)

    # Other data
    demand, var_cost = data.demand, data.variable_cost

    # Origins and destinations of commodities
    origins, destinations = data.origins, data.destinations

    # We use arrays to store variable indexes and variable objects. Why use
    # both? Gurobi wont let us get the values of individual variables
    # within a callback.. We just get the values of a large array of
    # variables, in the order they were initially defined. To separate them
    # in variable categories, we will have to use index arrays
    flow_vars = np.empty_like(arcs, dtype=object)

    # Populate all variables in one loop, keep track of their indexes
    # Data for period = 0, com = 0
    for arc in arcs:
        flow_vars[arc] = subproblem.addVar(
            obj=demand[0, 0]*var_cost[arc], lb=0., ub=1., name='flow_a{}'.format(arc))

    subproblem.update()
    # Add constraints
    for node in nodes:
        out_arcs = get_2d_index(data.arcs, data.nodes)[0] == node + 1
        in_arcs = get_2d_index(data.arcs, data.nodes)[1] == node + 1
        lhs = quicksum(flow_vars[out_arcs]) - quicksum(flow_vars[in_arcs])
        subproblem.addConstr(lhs == 0., name='flow_bal{}'.format(node))
    subproblem.update()

    # Store variables
    subproblem._all_variables = flow_vars.tolist()

    # Set parameters
    subproblem.setParam('OutputFlag', 0)
    subproblem.modelSense = GRB.MINIMIZE
    subproblem.params.threads = 2
    subproblem.params.LogFile = ""
    subproblem.update()

    subproblems[0, 0] = subproblem

    for period, com in product(periods, commodities):
        if (period, com) != (0, 0):
            model = subproblem.copy()
            model.ModelName = 'subproblem_({},{})'.format(period, com)
            flow_cost = data.demand[period, com] * var_cost
            model.setObjective(LinExpr(flow_cost.tolist(), model.getVars()))
            model.setAttr('rhs', model.getConstrs(), [0.0] * data.nodes)

            model._all_variables = model.getVars()
            model.update()
            subproblems[period, com] = model

    return subproblems


def callback_data(subproblems, data):
    """
    This is a closure that passes whatever data we want to the actual
    callback function. We have to use this because gurobi callbacks have a
    certain signature (model, where)
    :param subproblems: Gurobi subproblem models (one per period)
    :param data:       Problem data
    :param subproblems_po:  subproblems that solve for Pareto Optimal cuts
    :return:           master_callback function

    """
    # @profile
    def solve_dual_subproblem(setup_vars):
        """
        Solves the dual Benders subproblems.
        :param setup_vars:  Setup variables in master solution
        :param org_cuts:    Origin cuts that have been added so far
        :param dest_cuts:   Destination cuts that have been added so far
        :return:            Gurobi status message, Subproblem_Duals object
        """
        periods,  commodities = xrange(data.periods), xrange(data.commodities)
        origins, destinations = data.origins, data.destinations
        demand = data.demand

        # Return arrays
        status_arr = np.zeros(
            shape=(len(periods), len(commodities)), dtype=int)
        duals_arr = np.empty(
            shape=(len(periods), len(commodities)), dtype=object)

        # Prepare DiGraph
        arcs_rem = np.where(1. - setup_vars[0, :] <= 10e-5)[0]
        graph_t1 = nx.DiGraph()
        graph_t1.add_nodes_from(xrange(data.nodes))
        graph_t1.add_weighted_edges_from(zip(data.arc_org[arcs_rem], data.arc_dest[arcs_rem], data.variable_cost))
        skip_com = np.array([False]*len(commodities))

        for com in commodities:
            if not nx.has_path(graph_t1, source=origins[com], target=destinations[com]):
                skip_com[com] = True
                # Add cuts for the origin or destination component.
                # Prioritize smallest one
                desc_org = list(nx.descendants(graph_t1, origins[com]))
                desc_org.append(origins[com])
                asc_dest = list(nx.ancestors(graph_t1, destinations[com]))
                asc_dest.append(destinations[com])
                if len(desc_org) < len(asc_dest):
                    mask = np.in1d(data.arc_org, desc_org)
                    compl = set(xrange(data.nodes)).difference(set(desc_org))
                    mask &= np.in1d(data.arc_dest, list(compl))
                else:
                    mask = np.in1d(data.arc_dest, asc_dest)
                    compl = set(xrange(data.nodes)).difference(set(asc_dest))
                    mask &= np.in1d(data.arc_org, list(compl))
                if mask is not None:
                    arcs_to_add = np.nonzero(mask)[0]
                    flow_duals_vals = np.zeros(shape=data.nodes)
                    ubound_duals_vals = np.zeros(shape=data.arcs.size)
                    ubound_duals_vals[arcs_to_add] = 1.
                    flow_duals_vals[origins[com]] = 1.
                    duals = Subproblem_Duals(
                        flow_duals=flow_duals_vals,
                        bounds_duals=ubound_duals_vals,
                        optimality_dual=1.)
                    duals_arr[0, com] = duals
                    status_arr[0, com] = GRB.status.INFEASIBLE
        # Interior point
        setup = setup_vars.cumsum(axis=0)
        y0 = np.copy(setup)
        # y0[np.where(y0 > 10e-5)] = 1. - 10e-2
        y0[np.where(y0 > 10e-5)] = 1 - 10e-2
        y0[np.where(y0 < 10e-3)] = 10e-2

        # Magnanti point
        x0 = y0.sum(axis=1)
        upper_bounds = y0 + np.tile(x0, (data.arcs.size, 1)).transpose() * setup

        # Solve each subproblem and store the solution
        for com in commodities:
            if not skip_com[com]:
                for period in periods:
                	# If msk is False, we can skip solving the LP
                    msk = True
                    if period > 0:
                        msk = not (np.sum(setup_vars[period, :]) < 10e-4)
                        if msk:
                            # Commodity-specific check: if some arcs open, are the corresponding duals nonzero?
                            msk = np.any(ubound_duals_vals[np.nonzero(setup_vars[period, :])[0]])
                    if msk:
                        subproblem = subproblems[period, com]
                        all_variables = subproblem._all_variables
                        subproblem.getConstrByName('flow_bal{}'.format(origins[com])).rhs = 1. + x0[period]
                        subproblem.getConstrByName('flow_bal{}'.format(destinations[com])).rhs = - 1. - x0[period]
                        subproblem.setAttr('ub', subproblem._all_variables, upper_bounds[period, :].tolist())

                        subproblem.optimize()
                        status_arr[period, com] = subproblem.status

                        if subproblem.status == GRB.status.OPTIMAL:
                            # We need to add a cut, grab the duals
                            flow_duals_vals = np.array(
                                subproblem.getAttr("Pi", subproblem.getConstrs()))
                            ubound_duals_vals = np.array(
                                subproblem.getAttr('RC', all_variables))
                            zero_flow_idx = np.where(np.array(
                                subproblem.getAttr('X', all_variables)) <= 10e-3)[0]
                            ubound_duals_vals = np.negative(ubound_duals_vals)
                            ubound_duals_vals[zero_flow_idx] = 0.
                        elif status_arr[period, com] in (3, 4, 5):
                            raise RuntimeWarning('Subproblem unbounded/infeasible. Something went wrong..')
                        else:
                            raise RuntimeWarning('Something else went wrong..')
                    else:
                        ubound_duals_vals = demand[period, com] / demand[period - 1, com] * ubound_duals_vals
                        flow_duals_vals = demand[period, com] / demand[period - 1, com] * flow_duals_vals
                        status_arr[period, com] = subproblem.status

                    # Here are the cut coefficients
                    duals = Subproblem_Duals(
                        flow_duals=flow_duals_vals[:],
                        bounds_duals=ubound_duals_vals[:],
                        optimality_dual=1.)

                    duals_arr[period, com] = duals

        return status_arr, duals_arr

    def master_callback(model, where):
        if where == GRB.callback.MIPSOL:
            variables = np.array(model.cbGetSolution(model._variables))
            setup_vars = np.take(variables, model._bin_vars_idx)
            subproblem_status_arr, duals_arr = solve_dual_subproblem(
                setup_vars)
            for period in xrange(data.periods):
                for com in xrange(data.commodities):
                    subproblem_status = subproblem_status_arr[period, com]
                    duals = duals_arr[period, com]
                    if subproblem_status in \
                            (GRB.status.OPTIMAL, 3, 4, 5):
                        lhs = populate_benders_cut(
                            duals, model, period, com, data, subproblem_status)
                        model.cbLazy(lhs=lhs, rhs=0., sense=GRB.GREATER_EQUAL)
        elif where == GRB.callback.MIPNODE:
            node_count = int(model.cbGet(GRB.callback.MIPNODE_NODCNT))
            if (node_count % 1000 == 0 or node_count < 1000) and model.cbGet(
                    GRB.callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                variables = model.cbGetNodeRel(model._variables)
                setup_vars = np.take(variables, model._bin_vars_idx)
                subproblem_status_arr, duals_arr = solve_dual_subproblem(
                    setup_vars)
                for period in xrange(data.periods):
                    for com in xrange(data.commodities):
                        subproblem_status = subproblem_status_arr[period, com]
                        duals = duals_arr[period, com]
                        if subproblem_status in \
                            (GRB.status.OPTIMAL, 3, 4, 5):
                            lhs = populate_benders_cut(
                                duals, model, period, com, data,
                                subproblem_status)
                            if node_count < 10:
                                model.cbCut(
                                    lhs=lhs, rhs=0., sense=GRB.GREATER_EQUAL)
                            else:
                                model.cbLazy(
                                    lhs=lhs, rhs=0., sense=GRB.GREATER_EQUAL)

    return master_callback


def populate_benders_cut(duals, master, period, com, data, status):
    """
    Returns the lhs and rhs parts of a benders cut. It does not determine if
    the cut is an optimality or a feasibility one (their coefficients are the
    same regardless)

    :param duals:       model dual values (structure Subproblem_Duals)
    :param master:      master gurobi model
    :param period:      period in which we add the cut
    :param com:         commodity in which we add the cut
    :param data:        problem data
    :param status:      subproblem status
    :return:            rhs (double), lhs (Gurobi linear expression)
    """

    # Grab cut coefficients from the subproblem
    flow_duals = duals.flow_duals
    ubound_duals = duals.bounds_duals
    optimality_dual = duals.optimality_dual

    origin, destination = data.origins[com], data.destinations[com]

    continuous_variable = master.getVarByName(
        'flow_cost{}'.format((period, com)))
    setup_variables = np.take(
        master._variables, master._bin_vars_idx)[:period + 1, :]

    const = -flow_duals[origin] + flow_duals[destination]

    lhs = LinExpr(const)
    if status == GRB.status.OPTIMAL:
        lhs.add(continuous_variable)

    ubound_duals_nz_idx = np.nonzero(ubound_duals)[0]
    if ubound_duals_nz_idx.tolist():
        y_vars = setup_variables.take(
            ubound_duals_nz_idx, axis=1).flatten('F').tolist()
        coeffs = ubound_duals[ubound_duals_nz_idx].repeat(period+1)
        lhs.addTerms(coeffs, y_vars)

    return lhs


if __name__ == '__main__':
    main()
