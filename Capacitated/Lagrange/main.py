import numpy as np
import hashlib
import time
from sys import argv
from collections import namedtuple, deque
from subproblem import solve_subproblem
from helpers import get_2d_index, read_data
from heuristic import heuristic_main
from new_heuristic2 import heuristic, make_local_branching_model, \
    solve_local_branching_model, solve_fixing_model
from incremental_heuristics import Model
import master

# Variable object (Structure) to
# hold variables to be added to the master problem
Variable = namedtuple('Variable', 'objective flow open_arc')
# Remove scientific notation from numpy
np.set_printoptions(suppress=True)

# @profile
def lagrange_relaxation(data, time_limit=7200):
    """
    Calculates the arc-based lagrange relaxation of the multi-period
    multi-commodity network design problem
    Ioannis Fragkos, March 2015
    """

    start_time = time.time()

    # Initialize Lagrange upper bound and dual prices
    pi_iter = heuristic_main(data, pi_only=True)
    upper_bound, incumbent_arcs, incumbent_flow_cost = heuristic(data, 4)
    heur_model = make_local_branching_model(
        data, open_arcs=incumbent_arcs, cutoff=upper_bound)
    lower_bound, heuristic_objective = 0., upper_bound
    # return
    # Initialize dual prices
    pi_best = np.empty(pi_iter.shape)
    np.copyto(pi_best, pi_iter)

    # Initialize quantities needed for Adding/Removing arcs
    srt_idx = np.argsort(np.diff(data.fixed_cost, axis=0), axis=1)
    is_same, cut_off = False, 0.7

    # Initialize models used for delaying arcs
    models = []
    for period in xrange(data.periods):
        model = Model(data, period)
        model.objective = incumbent_flow_cost[period, :].sum()
        inc = incumbent_arcs[:period + 1, :].sum(axis=0)
        for arc in xrange(data.arcs.size):
            if inc[arc] == 0.:
                start, end = data.arc_org[arc], data.arc_dest[arc]
                m_vars = model.vars.select('*', start, end)
                model.model.setAttr('ub', m_vars, [inc[arc]] * len(m_vars))
        models.append(model)

    # Initialize violations vector, defined as
    # b(t, i, k) - sum(out_arcs(t, k, ij) + sum(in_arcs(t, k, ji))
    violations = np.zeros(shape=(data.periods, data.nodes, data.commodities))
    primal_violations = np.zeros(violations.shape)

    # Initialize data structure that holds columns to be added to the master.
    # We add 10 columns at most per arc and iteration.
    # Also, initialize tolerance after which we switch to CG
    columns_to_add, hash_cols, cg_tol = np.empty(
        data.arcs.shape, dtype=object), np.empty(data.arcs.shape, dtype=object), 0.0
    for arc in xrange(data.arcs.size):
        columns_to_add[arc] = deque(maxlen=50)
        hash_cols[arc] = set()

    # Define the subgradient parameters
    max_iter, omega, max_lb_iter, decrease_factor, check_heuristic_iter = \
        1000, 1.99, 50, 0.99, 300

    # Initialize counter for lb change the step switching value
    # and the deflection parameter
    lb_iter, step_switch, alpha = 10, 100, 0.2

    # Initialize array of the constant term
    # (that adds or subtracts the dual variables)
    const_array = np.zeros(pi_iter.shape)

    # Initialize primal solution of the relaxation
    # (NOT of the original problem)
    primal_solution = namedtuple('Solution', 'objective_value open_arc flow')
    primal_solution.flow = np.zeros(shape=(
        data.periods, data.commodities, data.arcs.size))
    primal_solution.open_arc = np.zeros(shape=(data.periods, data.arcs.size))
    primal_solution.objective_value, primal_variable_cost = \
        upper_bound, np.outer(data.demand, data.variable_cost).reshape(
            data.periods, data.commodities, data.arcs.size)
    kappa = 4
    open_arcs = np.zeros(primal_solution.open_arc.shape)
    arc_popularity = np.zeros(shape=(  # arc popularity in a batch of iters
        data.periods, data.arcs.size), dtype=float)
    local_arc_popularity = np.zeros(arc_popularity.shape)
    max_periods = np.empty(
        data.arcs.shape, dtype=int)  # used for arc popularity as well
    denominator = np.array(xrange(1, max_iter + 1), dtype=float)
    denominator **= kappa
    denominator = denominator.cumsum()

    # Initialize switch to column generation.
    solve_cg = False

    # We have to initialize some element to +1/-1 depending
    # on the origin/destination pair of each commodity
    origins, destinations = get_2d_index(data.od_pairs, data.nodes)
    origins -= 1
    destinations -= 1
    for commodity in xrange(data.commodities):
        origin, destination = origins[commodity], destinations[commodity]
        const_array[:, origin, commodity] += 1
        const_array[:, destination, commodity] -= 1

    # Here is the main subgradient loop
    for iteration in xrange(max_iter):

        # Initialize the objective function value of this iteration and
        # of possible ip solution / initialize violations
        obj_val_iter = 0.
        violations *= 0.
        primal_violations *= 0.

        if time.time() - start_time > time_limit:
            'Lagrange time limit reached. Breaking out!'
            break

        # Update approximate primal solution
        nominator = denominator[iteration - 1] if iteration - 1 >= 0 else 0
        primal_solution.flow *= nominator / denominator[iteration]
        primal_solution.open_arc *= nominator / denominator[iteration]

        # Solve a subproblem for each arc
        for arc in xrange(len(data.arcs)):
            arc_origin, arc_dest = get_2d_index(data.arcs[arc], data.nodes)

            # Get vector of lagrange costs
            flow_coeffs = get_lagrange_cost(
                dual_prices=pi_iter, arc_pointer=arc, data=data)

            # Solve subproblem and get back solution vector
            subproblem_sol = solve_subproblem(
                lagrange_cost=flow_coeffs,
                demand=data.demand, fixed_cost=data.fixed_cost[:, arc],
                capacity=data.capacity[arc], period=data.periods - 1)

            # Add term to objective function for this iteration
            obj_val_iter += subproblem_sol.objective_value

            # We then check if the subproblem solution prices out
            # and add columns to the pool
            primal_objective = 0 if subproblem_sol.open_period == -1 else \
                np.sum(subproblem_sol.flow * data.demand) * \
                data.variable_cost[arc] + data.fixed_cost[
                    subproblem_sol.open_period, arc]
            if solve_cg:
                variable = Variable(
                    primal_objective, subproblem_sol.flow,
                    subproblem_sol.open_period)
                hashed_var = hashlib.sha1(variable.flow).hexdigest()
                if hashed_var not in hash_cols[arc]:
                    columns_to_add[arc].append(variable)
                    hash_cols[arc].update([hashed_var])

            # Update approximate primal solution and arc popularity
            primal_solution.flow[:, :, arc] += ((
                float(iteration) ** kappa) / denominator[
                iteration]) * subproblem_sol.flow
            if subproblem_sol.open_period >= 0:
                primal_solution.open_arc[subproblem_sol.open_period, arc] += (
                    float(iteration) ** kappa) / denominator[iteration]
                if np.random.random() > 0.01:
                    local_arc_popularity[subproblem_sol.open_period, arc] = \
                        primal_solution.open_arc[
                        subproblem_sol.open_period, arc]
                else:
                    local_arc_popularity[subproblem_sol.open_period, arc] = 1.
            elif np.random.random() > 0.01:
                local_arc_popularity[subproblem_sol.open_period, arc] = \
                    primal_solution.open_arc[subproblem_sol.open_period, arc]

            # Update node violations
            violations[:, arc_origin - 1, :] += subproblem_sol.flow
            violations[:, arc_dest - 1, :] -= subproblem_sol.flow
            primal_violations[:, arc_origin - 1, :] += primal_solution.flow[
                :, :, arc]
            primal_violations[:, arc_dest - 1, :] -= primal_solution.flow[
                :, :, arc]

            # print iteration, arc, obj_val_iter

        # Update overall violations
        origins, destinations = get_2d_index(data.od_pairs, data.nodes)
        origins -= 1
        destinations -= 1
        for commodity in xrange(data.commodities):
            origin, destination = origins[commodity], destinations[commodity]
            violations[:, origin, commodity] -= 1
            violations[:, destination, commodity] += 1
            primal_violations[:, origin, commodity] -= 1
            primal_violations[:, destination, commodity] += 1

        # Add constant term to objective function
        obj_val_iter += np.sum(np.multiply(const_array, pi_iter))

        # Evaluate primal objective
        if iteration > 0:
            primal_solution.objective_value = get_primal_objective(
                primal_solution, data.fixed_cost, primal_variable_cost)

        # print 'primal violations: {} Max Violation: {}'.format(
        #     np.square(primal_violations).sum(), np.abs(
        # primal_violations).max())

        # Increase lower bound counter
        lb_iter += 1

        # Check if lower bound has improved
        if obj_val_iter > lower_bound:
            lower_bound = obj_val_iter
            omega = min(1.05 * omega, 1.2)
            alpha *= 1.05
            np.copyto(pi_best, pi_iter)
            np.copyto(arc_popularity, local_arc_popularity)
        else:
            alpha *= 0.95

        # If primal solution has small violations, we stop
        # Remember to add solve_cg = False here
        max_viol = np.abs(primal_violations).max()
        if max_viol < 0.01:
            for _ in xrange(2):
                heuristic_objective, incumbent_arcs, heur_model = \
                    solve_local_branching_model(
                        data, incumbent_arcs, lb_model=heur_model, \
                        cutoff=upper_bound)
            if heuristic_objective < upper_bound:
                print 'New upper bound found from heuristic: {}'.format(
                    heuristic_objective)
            upper_bound = heuristic_objective
            print 'Stopping due to very small primal violations'
            break

        # Reduce the omega multiplier
        if lb_iter > max_lb_iter:
            omega = max(omega * decrease_factor, 10e-2)
            # violations *= 1 + np.random.random()*10e-6
            lb_iter = 0

        # Check if we can find a better upper bound, based on arc popularity
        if iteration > 0 and iteration % check_heuristic_iter == 0:
            print 'checking heuristic.. iteration {}'.format(iteration)

            max_periods[np.where(arc_popularity.max(axis=0) > 0.2)] = \
                arc_popularity.argmax(axis=0)[np.where(arc_popularity.max(
                    axis=0) > 0.2)]
            max_periods[np.where(arc_popularity.max(axis=0) <= 0.2)] = -1

            # Round primal solution and define the search neighborhood
            open_arcs = get_rounded_solution(primal_solution.open_arc, cut_off)
            diff_arcs = incumbent_arcs != open_arcs
            cut_off = max(0.1, cut_off - 0.1)

            # Call the Add/Remove arcs heuristic
            # if not is_same:
            incumbent_arcs, heuristic_objective, is_same = \
                lagrange_heuristic(
                    data, models, srt_idx, incumbent_arcs, upper_bound,
                    diff_arcs)
            upper_bound = min(upper_bound, heuristic_objective)
            # else:
            #     diff_arcs.fill(True)
            #     incumbent_arcs, heuristic_objective = fixing_heuristic(
            #         data, models, incumbent_arcs, upper_bound)
            #     is_same = False

        if iteration in (max_iter - 1, ):
            # global start
            print 'Lower Bound: {}'.format(lower_bound)
            print 'Calculation Time: {}'.format(time.time() - start_time)
            improved = False
            heuristic_objective, incumbent_arcs, heur_model = \
            solve_fixing_model(
                data, heur_model, primal_solution, 0.1, upper_bound, 
                incumbent_arcs)
            # Record if upper bound got improved
            improved = heuristic_objective < upper_bound
            upper_bound = min(upper_bound, heuristic_objective)
            heuristic_objective, incumbent_arcs, heur_model = \
                solve_local_branching_model(
                    data, incumbent_arcs, lb_model=heur_model, \
                    cutoff=upper_bound)
            # Before the application of the lagrange heuristic, we need
            # to pass the incumbent objective and the upper bounds to the 
            # single period models..
            # Record again if upper bound got improved
            improved = improved or (heuristic_objective < upper_bound)
            # We only need to update the models if there exists an 
            # improvement, i.e., a new solution
            # improved = True
            if improved:
                flow_vars = np.array(
                   [var.X for var in heur_model.getVars() 
                   if 'flow' in var.VarName]).reshape(
                   data.periods, data.arcs.size, data.commodities)
                arc_open = np.cumsum(incumbent_arcs, axis=0)
                for t in xrange(data.periods):
                    model = models[t]
                    model.set_objective_and_bounds(
                        data, flow_vars[t, :, :].flatten(), arc_open[t, :])
                print 'check active here'
            for _ in xrange(4):
                upper_bound = min(upper_bound, heuristic_objective)                
                diff_arcs = incumbent_arcs != open_arcs
                incumbent_arcs, heuristic_objective, is_same = \
                lagrange_heuristic(
                    data, models, srt_idx, incumbent_arcs, upper_bound,
                    diff_arcs)
                upper_bound = min(upper_bound, heuristic_objective)
            heuristic_objective, incumbent_arcs, heur_model = \
            solve_fixing_model(
                data, heur_model, primal_solution, 0.3, upper_bound, 
                incumbent_arcs)
            upper_bound = min(upper_bound, heuristic_objective)

        if heuristic_objective < upper_bound:
            print 'New upper bound found from heuristic: {}'.format(
                heuristic_objective)
            upper_bound = heuristic_objective
            arc_popularity *= 0
            is_same = False
        else:
            is_same = True

        # Calculate relative gap
        gap = (upper_bound - lower_bound) / upper_bound
        # # True the first time we switch
        # if gap < cg_tol and not solve_cg:
        #     solve_cg = True
        #     master_model = master.make_master(
        #         data, heur_solution=heuristic_solution)
        # Subgradient step
        if iteration < step_switch:
            step, squared_viol = get_subgradient_step(
                'polyak', violations, lower_bound=obj_val_iter,
                upper_bound=upper_bound, omega=omega)
            alpha_hat = step * (iteration + 1)
        else:
            step, squared_viol = get_subgradient_step(
                'harmonic', violations, alpha_hat=alpha_hat,
                iteration=iteration)
        # Update the search direction - deflected subgradient!
        pi_iter -= step * (
            alpha * primal_violations + (1 - alpha) * violations)

        if iteration % 20 == 0:
            print iteration, lower_bound, max_viol, round(
                time.time() - start_time, 0)

    print "Upper bound: {} Lower bound: {}".format(upper_bound, lower_bound)
    # exact_cg(model=master_model, data=data, columns_to_add=columns_to_add)
    return lower_bound


def exact_cg(model, data, columns_to_add):
    """
    solves the master problem of the column generation formulation
    :param model:           master model
    :param data:            probelm data
    :param columns_to_add:  columns to warm start the master
    :return:        lower bound
    """

    reduced_cost, master_iter = -1., 0

    while reduced_cost < -10e-6:
        master.add_variables(model=model, data=data, variables=columns_to_add)
        master.optimize(model=model, data=data)
        master_objective = model.getObjective().getValue()
        convex_duals = model._convex_duals
        pi_iter = np.transpose(model._node_duals, [2, 0, 1])

        reduced_cost, obj_val_iter = 0., 0.

        # Solve a subproblem for each arc
        for arc in xrange(len(data.arcs)):

            # From now on, we add one variable per subproblem at a time
            columns_to_add[arc].clear()

            # Get vector of lagrange costs
            flow_coeffs = get_lagrange_cost(dual_prices=pi_iter, arc_pointer=arc, data=data)

            # Solve subproblem and get back solution vector
            subproblem_sol = solve_subproblem(lagrange_cost=flow_coeffs,
                              demand=data.demand, fixed_cost=data.fixed_cost[:, arc],
                              capacity=data.capacity[arc], period=data.periods - 1)

            # Add term to objective function for this iteration
            obj_val_iter += subproblem_sol.objective_value

            # We then check if the subproblem solution prices out and add columns to the pool
            primal_objective = 0 if subproblem_sol.open_period == -1 else \
                np.sum(subproblem_sol.flow * data.demand) * data.variable_cost[arc] + \
                data.fixed_cost[subproblem_sol.open_period, arc]
            if subproblem_sol.objective_value - convex_duals[arc] < -10e-8:
                variable = Variable(primal_objective, subproblem_sol.flow, subproblem_sol.open_period)
                columns_to_add[arc].append(variable)
                reduced_cost += subproblem_sol.objective_value - convex_duals[arc]

        master_iter += 1
        print 'reduced cost: {} master objective: {}'.format(reduced_cost, master_objective)

    print 'finished! Iterations: {}'.format(master_iter)
    # return master_objective


def get_subgradient_step(method, violations, **kwargs):
    """
    Returns the step size and the sum of squared infeasibilities
    :param method:  method that calculates the step size
    :param kwargs:  if method == polyak: {lower_bound, upper_bound, omega, step}
                    if method == harmonic: {alpha_hat, iteration}
    :return:        step size, sum of squared infeasibilities
    """

    sum_squared_infeasibilities = np.square(violations).sum()
    if method == 'polyak':
        lower_bound = kwargs['lower_bound']
        upper_bound = kwargs['upper_bound']
        omega = kwargs['omega']
        step = omega * (upper_bound - lower_bound) / sum_squared_infeasibilities
    elif method == 'harmonic':
        alpha_hat = kwargs['alpha_hat']
        iteration = kwargs['iteration']
        step = alpha_hat / (1.0 + iteration)
    return step, sum_squared_infeasibilities


def get_lagrange_cost(dual_prices, arc_pointer, data):
    """
    Returns the coefficients of the flow variables for the arc subproblem
    :param dual_prices:         dual prices of each node for each commodity in each period
    :param data:                problem data
    :return:                    a numpy array of cost coefficients
    """

    arc_no = data.arcs[arc_pointer]
    origin, destination = get_2d_index(arc_no, data.nodes)
    origin, destination = origin - 1, destination - 1
    arc_cost = data.variable_cost[arc_pointer]
    flow_cost = arc_cost * data.demand
    lagrange_diff = + dual_prices[:, origin, :] - dual_prices[:, destination, :]
    flow_cost -= lagrange_diff

    return flow_cost


def get_primal_objective(primal_solution, fixed_cost, variable_cost):
    flow = primal_solution.flow
    open_arc = primal_solution.open_arc
    primal_objective = 0.
    primal_objective += np.multiply(fixed_cost, open_arc).sum()
    primal_objective += np.multiply(flow, variable_cost).sum()
    # print 'Primal objective: {}'.format(primal_objective)
    return primal_objective


def get_rounded_solution(feasible_sol, threshold):
    """
    Given a threshold between 0 and 1, rounds feasible_sol above and below
    For each column of feasible_sol looks at the highest entry, and if above
    threshold it rounds it to 1, and rounds all else to zero. Otherwise, it
     keeps the entire column to zero.
     Ioannis Fragkos December 2016
    :param feasible_sol:        Numpy arr with volume solution (periods, arcs)
    :param threshold:           Real value between 0 and 1
    :return:                    Numpy arr (period, arcs); at most 1 per column
    """

    ret_array = np.zeros(feasible_sol.shape)
    # Pick up period that maximizes the value of each arc
    max_periods = feasible_sol.argmax(axis=0)
    arcs = feasible_sol.shape[1]
    for arc in xrange(arcs):
        if feasible_sol[max_periods[arc], arc] > threshold:
            ret_array[max_periods[arc], arc] = 1.

    return ret_array


def lagrange_heuristic(data, models, srt_idx, inc_sol, inc_obj, diff_arcs):
    """
    Heuristic that follows the following logic:
    For every period, it considers the arcs that are different between the
    incumbent solution and the rounded solution. It goes through the arcs and
    tries to either add an arc, if it is zero in the incumbent or to remove an
     arc, if it is 1 in the incumbent. It also updates the incumbent through
     this process, and finally returns a newly found incumbent. If the incumbent
     did not improves, it indicates that as well.
     Ioannis Fragkos, December 2016
    :param models:      Single period LP-models
    :param  data:       Problem data
    :param srt_idx:     Indexes of sorted fixed cost differences
    :param inc_sol:     Incumbent solution (periods, arcs) array
    :param inc_obj:     Objective of the incumbent
    :param diff_arcs:   Boolean showing if the arc is different between
                        the incumbent and the rounded vector
    :return:            inc_sol, inc_obj,
                        boolean showing if the new incumbent is better
    """
    initial_inc = inc_obj
    for t in xrange(data.periods - 1):
        # candidate_arcs = srt_idx[t, diff_arcs[t, :]]
        candidate_arcs = np.nonzero(diff_arcs[t, :])[0]
        model = models[t]
        for arc in xrange(len(candidate_arcs)):
            arc_idx = candidate_arcs[arc]
            # if arc_idx % 2 == 0:
            #     arc = candidate_arcs[len(candidate_arcs) - arc_idx/2 - 1]
            #     print 'add arc {}'.format(arc)
            #     # add_arc(candidate_arcs[len(candidate_arcs) - arc_idx/2 - 1])
            # else:
            #     arc = candidate_arcs[(arc_idx - 1)/2]
            #     print 'shift arc {}'.format(arc)

            if inc_sol[t, arc_idx] == 1:
                model.open_later(data=data, arc=arc_idx)
                if model.benefit > 0:
                    inc_sol[t, arc_idx], inc_sol[t + 1, arc_idx] = 0, 1
                    print 'objective improvement. New upper bound: {}'.format(
                        inc_obj - model.benefit)
                    inc_obj -= model.benefit
                    model.benefit = -1

    return inc_sol, inc_obj, initial_inc == inc_obj


def test():
    filename = 'r03.1_R_H_20.dow' if not len(argv) > 1 else argv[1]
    data = read_data(filename)
    start = time.time()
    lagrange_relaxation(data)
    stop = time.time()
    print 'Lagrange relaxation done. Time: {} seconds'.format(stop - start)


if __name__ == '__main__':
    test()
