from __future__ import division
from helpers import read_data, get_2d_index
from solution_stats import recover_solution_data
from sys import argv
from time import time
from itertools import product
from networkx import shortest_path_length, shortest_path
from gurobipy import Model, GRB, LinExpr, quicksum
from graph_helpers import make_graph
from collections import namedtuple
import numpy as np

__author__ = 'ioannis'
# File to gather data and Implement Benders Decomposition
# Ioannis Fragkos, August 200

# Container that holds the subproblem dual vectors. The vectors should be
# entered as numpy arrays.
# Dimensions: flow_duals(node, commodity, period); bounds_duals(period, arc)
# We carry numpy arrays of Subproblem_Duals so that we add many cuts at once
Subproblem_Duals = namedtuple('Subproblem_Duals',
                              'flow_duals bounds_duals optimality_dual')


def main():
    filename = 'r03.1_R_H_20.dow' if len(argv) <= 1 \
        else argv[1]
    data = read_data(filename)
    start = time()
    data.graph = make_graph(data)
    subproblems = populate_dual_subproblem(data)
    master = populate_master(data)
    master = add_in_out_cuts(master, subproblems, data)
    master_callback = callback_data(subproblems, data)
    master.optimize(master_callback)
    stop = time()
    print('Total time: {} seconds'.format(round(stop - start, 0)))


def populate_master(data, open_arcs=None):
    """
    Function that populates the Benders Master problem
    :param data:   Problem data structure
    :param open_arcs: If given, it is a MIP start feasible solution
    :rtype:        Gurobi model object
    """
    master = Model('master-model')
    arcs, periods = range(data.arcs.size), range(data.periods)
    commodities = range(data.commodities)
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
    master.addConstrs(
        (quicksum(variables[:, arc]) <= 1. for arc in arcs), name='one')

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
    master.params.TimeLimit = 7200
    master.params.threads = 1
    master.params.BranchDir = 1
    master.params.PreCrush = 1
    # Store the variables inside the model, we cannot access them later!
    master._variables = np.array(master.getVars())
    master._cont_vars_idx = cont_vars_idx
    master._bin_vars_idx = bin_vars_idx
    return master


def populate_dual_subproblem(data, upper_cost=None, flow_cost=None):
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
    dual_subproblem = Model('dual_subproblem_(0,0)')

    # Ranges we are going to need
    arcs, periods, commodities = range(data.arcs.size), range(
        data.periods), range(data.commodities)

    # Origins and destinations of commodities
    origins, destinations = data.origins, data.destinations

    # We use arrays to store variable indexes and variable objects. Why use
    # both? Gurobi wont let us get the values of individual variables
    # within a callback.. We just get the values of a large array of
    # variables, in the order they were initially defined. To separate them
    # in variable categories, we will have to use index arrays
    flow_index = np.zeros(shape=data.nodes, dtype=int)
    flow_duals = np.empty_like(flow_index, dtype=object)
    ubounds_index = np.zeros(shape=len(arcs), dtype=int)
    ubounds_duals = np.empty_like(ubounds_index, dtype=object)

    # Makes sure we don't add variables more than once
    flow_duals_names = set()

    if upper_cost is None:
        upper_cost = np.zeros(shape=(len(periods), len(arcs)), dtype=float)
    if flow_cost is None:
        flow_cost = np.zeros(shape=(len(periods), len(commodities)),
                             dtype=float)

    # Populate all variables in one loop, keep track of their indexes
    # Data for period = 0, com = 0
    count = 0
    for arc in arcs:
        ubounds_duals[arc] = dual_subproblem.addVar(
            obj=-upper_cost[0, arc], lb=0., name='ubound_dual_a{}'.format(arc))
        ubounds_index[arc] = count
        count += 1
        start_node, end_node = get_2d_index(data.arcs[arc], data.nodes)
        start_node, end_node = start_node - 1, end_node - 1
        for node in (start_node, end_node):
            var_name = 'flow_dual_n{}'.format(node)
            if var_name not in flow_duals_names:
                flow_duals_names.add(var_name)
                obj, ub = 0., GRB.INFINITY
                if data.origins[0] == node:
                    obj = 1.
                if data.destinations[0] == node:
                    obj = -1.
                    ub = 0.
                flow_duals[node] = \
                    dual_subproblem.addVar(
                        obj=obj, lb=0., name=var_name)
                flow_index[node] = count
                count += 1
    opt_var = dual_subproblem.addVar(
        obj=-flow_cost[0, 0], lb=0., name='optimality_var')
    dual_subproblem.params.threads = 1
    dual_subproblem.params.LogFile = ""
    dual_subproblem.update()

    # Add constraints
    demand = data.demand[0, 0]
    for arc in arcs:
        start_node, end_node = get_2d_index(data.arcs[arc], data.nodes)
        start_node, end_node = start_node - 1, end_node - 1
        lhs = flow_duals[start_node] - flow_duals[end_node] \
              - ubounds_duals[arc] - \
              opt_var * data.variable_cost[arc] * demand
        dual_subproblem.addConstr(lhs <= 0., name='flow_a{}'.format(arc))

    # Original Fischetti model
    lhs = quicksum(ubounds_duals) + opt_var
    dual_subproblem.addConstr(lhs == 1, name='normalization_constraint')

    # Store variable indices
    dual_subproblem._ubounds_index = ubounds_index
    dual_subproblem._flow_index = flow_index
    dual_subproblem._all_variables = np.array(dual_subproblem.getVars())
    dual_subproblem._flow_duals = np.take(
        dual_subproblem._all_variables, flow_index)
    dual_subproblem._ubound_duals = np.take(
        dual_subproblem._all_variables, ubounds_index)

    dual_subproblem.setParam('OutputFlag', 0)
    dual_subproblem.modelSense = GRB.MAXIMIZE
    dual_subproblem.update()

    subproblems[0, 0] = dual_subproblem

    for period, com in product(periods, commodities):
        if (period, com) != (0, 0):
            model = dual_subproblem.copy()
            optimality_var = model.getVarByName('optimality_var')
            optimality_var.Obj = -flow_cost[period, com]
            demand = data.demand[period, com]
            for node in range(data.nodes):
                variable = model.getVarByName('flow_dual_n{}'.format(node))
                if origins[com] == node:
                    obj = 1.
                elif destinations[com] == node:
                    obj = -1.
                else:
                    obj = 0.
                variable.obj = obj
            for arc in arcs:
                variable = model.getVarByName('ubound_dual_a{}'.format(arc))
                variable.Obj = -np.sum(upper_cost[:period + 1, arc])
                constraint = model.getConstrByName('flow_a{}'.format(arc))
                model.chgCoeff(
                    constraint, optimality_var,
                    -demand * data.variable_cost[arc])
            model._all_variables = np.array(model.getVars())
            model.update()
            subproblems[period, com] = model
    return subproblems


def add_in_out_cuts(master, subproblems, data):
    """
    Takes the master problem and generates in-out cuts as described in 
    Fischetti et al. (EJOR 2016). Parameter tuning can probably be improved
    """
    periods, commodities = data.periods, data.commodities
    bin_vars_idx = master._bin_vars_idx
    cont_vars_idx = master._cont_vars_idx

    alpha, llambda = 0.9, 0.1

    max_iter = 300

    master.update()
    master.params.OutputFlag = 0
    all_vars = master.getVars()
    y_vars = np.take(all_vars, bin_vars_idx)
    master.setAttr('vtype', all_vars, [GRB.CONTINUOUS]*master.numVars)

    master.optimize()

    vals = np.array(master.getAttr('X', all_vars))
    y_star = np.take(vals, bin_vars_idx)
    obj = master.getObjective()
    best_lb = master.ObjVal

    # Initialize y_tilde by solving max {y} s.t. master cuts + constraints
    master.setObjective(y_vars.sum(), GRB.MAXIMIZE)
    # setup the Barrier algorithm and disable crossover
    master.params.method = 2
    master.params.crossover = 0
    master.optimize()
    vals = np.array(master.getAttr('X', all_vars))
    y_tilde = np.take(vals, bin_vars_idx)
    flow_vals = np.take(vals, cont_vars_idx)

    # Revert parameter changes
    master.setObjective(obj, GRB.MINIMIZE)
    master.params.method = -1
    master.params.crossover = -1

    num_init_constrs = master.numConstrs

    # Here we have initialized y_star, y_tilde, move on with the cut loop
    cnt, lb_impr_cnt = 0, 0
    while True:
        stop = False
        y_tilde = alpha * y_tilde + (1-alpha) * y_star
        y_sep = llambda * y_star + (1-llambda) * y_tilde

        subproblem_status_arr, duals_arr = solve_dual_subproblem(
            y_sep, subproblems=subproblems, data=data, flow_cost=flow_vals)
        for period in range(periods):
            for com in range(commodities):
                subproblem_status = subproblem_status_arr[period, com]
                duals = duals_arr[period, com]
                if subproblem_status == GRB.status.OPTIMAL:
                    lhs = populate_benders_cut(duals, master, period, com, data)
                    master.addConstr(lhs=lhs, rhs=0., sense=GRB.GREATER_EQUAL,
                                     name='in_out_{}-{}-{}'.format(
                                         cnt, period, com))

        master.optimize()
        vals = np.array(master.getAttr('X', all_vars))
        y_star = np.take(vals, bin_vars_idx)
        flow_vals = np.take(vals, cont_vars_idx)
        this_lb = master.ObjVal

        if this_lb > best_lb:
            imprv = (this_lb - best_lb) / best_lb
            stop = (imprv < 0.0001) and (cnt > max_iter)
            best_lb = this_lb
            lb_impr_cnt = 0
        else:
            lb_impr_cnt += 1

        if lb_impr_cnt >= 5:
            llambda = 1.
        else:
            llambda = 0.1

        # terminate if no improvement for 10 rounds or marginal improvement
        # after max_iter rounds
        if lb_impr_cnt == 10 or stop:
            break

        cnt += 1
        if cnt % 5 == 0:
            # remove positive slack cuts every five iterations
            constrs = master.getConstrs()
            slacks = master.getAttr('slack', constrs)[num_init_constrs:]
            rem = [constr for (idx, constr) in enumerate(
                constrs[num_init_constrs:]) if slacks[idx] < -10e-6]
            master.remove(rem)

        print('iteration:\t {}\t bound:{}'.format(cnt, best_lb))

    print('In-out terminated. Solving once more...')
    master.optimize()
    print('Final in-out bound: {}'.format(master.objval))

    # remove positive slack constrains for the last time
    constrs = master.getConstrs()
    slacks = master.getAttr('slack', constrs)[num_init_constrs:]
    rem = [constr for (idx, constr) in enumerate(
        constrs[num_init_constrs:]) if slacks[idx] < -10e-6]
    master.remove(rem)

    master.setAttr('vtype', y_vars.flatten().tolist(), [GRB.BINARY]*y_vars.size)
    master.params.OutputFlag = 1

    return master


def solve_dual_subproblem(setup_vars, subproblems, data, flow_cost=None):
    """
    Solves the dual Benders subproblems.
    :param flow_cost:   Continuous variables of Benders master problem
    :param setup_vars:  Setup variables in master solution
    :subproblems:       Subproblems structure
    :data:              Problem data
    :return:            Gurobi status message, Subproblem_Duals object
    """
    periods,  commodities = range(data.periods), range(data.commodities)

    # Indices of subproblem variables
    flow_index = subproblems[0, 0]._flow_index
    ubound_index = subproblems[0, 0]._ubounds_index

    # Return arrays
    status_arr = np.zeros(
        shape=(len(periods), len(commodities)), dtype=int)
    duals_arr = np.empty(
        shape=(len(periods), len(commodities)), dtype=object)

    sum_setup = np.negative(setup_vars).cumsum(axis=0)

    # Solve each subproblem and store the solution
    for period in periods:
        for com in commodities:
            subproblem = subproblems[period, com]
            all_variables = subproblem._all_variables
            optimality_var = all_variables[-1]
            all_variables = all_variables[:-1]
            flow_duals = np.take(all_variables, flow_index)
            ubound_duals = np.take(all_variables, ubound_index)

            # Modify the objective function
            subproblem.setAttr("obj",
                ubound_duals.tolist(), sum_setup[period, :].tolist())

            if flow_cost is not None:
                optimality_var.obj = - flow_cost[period, com]

            subproblem.optimize()
            status_arr[period, com] = subproblem.status

            if status_arr[period, com] == GRB.status.OPTIMAL:
                # We need to add a cut. First, grab the duals
                all_duals = np.array(
                    subproblem.getAttr("X", subproblem.getVars()))
                flow_duals_vals = all_duals.take(flow_index)
                ubound_duals_vals = all_duals.take(ubound_index)

                # Here are the cut coefficients
                duals = Subproblem_Duals(
                    flow_duals=flow_duals_vals,
                    bounds_duals=ubound_duals_vals,
                    optimality_dual=optimality_var.X)
                duals_arr[period, com] = duals
            else:
                raise RuntimeWarning('Something went wrong..')

    return status_arr, duals_arr


def callback_data(subproblems, data):
    """
    This is a closure that passes whatever data we want to the actual
    callback function. We have to use this because gurobi callbacks have a
    certain signature (model, where)
    :param subproblems: Gurobi subproblem models (one per period)
    :param data:       Problem data
    :return:           master_callback function

    """
    def master_callback(model, where):
        if where == GRB.callback.MIPSOL:
            node_count = int(model.cbGet(GRB.callback.MIPSOL_NODCNT))
            variables = np.array(model.cbGetSolution(model._variables))
            flow_cost = np.take(variables, model._cont_vars_idx)
            setup_vars = np.take(variables, model._bin_vars_idx)
            subproblem_status_arr, duals_arr = solve_dual_subproblem(
                flow_cost=flow_cost, data=data, subproblems=subproblems,
                setup_vars=setup_vars)
            for period in range(data.periods):
                for com in range(data.commodities):
                    subproblem_status = subproblem_status_arr[period, com]
                    duals = duals_arr[period, com]
                    if subproblem_status == GRB.status.OPTIMAL:
                        lhs = populate_benders_cut(duals, model,
                                                   period, com, data)
                        model.cbLazy(
                            lhs=lhs, rhs=0., sense=GRB.GREATER_EQUAL)
                    else:
                        raise RuntimeWarning('Subproblem unknown status')
        elif where == GRB.callback.MIPNODE:
            node_count = int(model.cbGet(GRB.callback.MIPNODE_NODCNT))
            if (node_count % 1000 == 0 or node_count < 1000) and model.cbGet(
                    GRB.callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                variables = model.cbGetNodeRel(model._variables)
                flow_cost = np.take(variables, model._cont_vars_idx)
                setup_vars = np.take(variables, model._bin_vars_idx)
                subproblem_status_arr, duals_arr = solve_dual_subproblem(
                    flow_cost=flow_cost, data=data, subproblems=subproblems,
                    setup_vars=setup_vars)
                for period in range(data.periods):
                    for com in range(data.commodities):
                        subproblem_status = subproblem_status_arr[period, com]
                        duals = duals_arr[period, com]
                        if subproblem_status == GRB.status.OPTIMAL:
                            lhs = populate_benders_cut(duals, model,
                                                       period, com, data)
                            if node_count < 10:
                                model.cbCut(
                                    lhs=lhs, rhs=0., sense=GRB.GREATER_EQUAL)
                            else:
                                model.cbLazy(
                                    lhs=lhs, rhs=0., sense=GRB.GREATER_EQUAL)
                        else:
                            raise RuntimeWarning('Subproblem unknown status')

    return master_callback


def populate_benders_cut(duals, master, period, com, data):
    """
    Returns the lhs and rhs parts of a benders cut. It does not determine if
    the cut is an optimality or a feasibility one (their coefficients are the
    same regardless)

    :param duals:       model dual values (structure Subproblem_Duals)
    :param master:      master gurobi model
    :param period:      period in which we add the cut
    :param com:         commodity in which we add the cut
    :param data:        problem data
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
    lhs.add(continuous_variable, optimality_dual) 

    if abs(optimality_dual) <= 10e-5:
        data.feasibility_cuts += 1
    else:
        data.optimality_cuts += 1

    ubound_duals_nz_idx = np.nonzero(ubound_duals)[0]
    if ubound_duals_nz_idx.tolist():
        y_vars = setup_variables.take(
            ubound_duals_nz_idx, axis=1).flatten('F').tolist()
        coeffs = ubound_duals[ubound_duals_nz_idx].repeat(period+1)
        # lhs_trial = LinExpr(const + continuous_variable * optimality_dual)
        lhs.addTerms(coeffs, y_vars)

    return lhs


if __name__ == '__main__':
    main()
