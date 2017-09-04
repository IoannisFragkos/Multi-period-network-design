# coding=utf-8
__author__ = 'ioannis'
"""
File that contains the subroutines necessary to solve the subproblem
Portable for both Lagrange Relaxation and Column Generation
"""

import numpy as np
from collections import namedtuple
import gurobipy as grb

Solution = namedtuple('Solution', 'objective_value solution_vector')


def solve_subproblem_gurobi(lagrange_cost, demand, fixed_cost, capacity):
    """
    Solves the subproblem using gurobi (see below for details)
    """

    subproblem_sol = namedtuple('Solution', 'objective_value flow open_period')

    subproblem = grb.Model('subproblem')
    periods_range, commodities_range = xrange(len(fixed_cost)), xrange(len(demand[0, :]))
    flow = np.array([subproblem.addVar(lb=0., ub=1., vtype=grb.GRB.CONTINUOUS, obj=lagrange_cost[period, commodity],
                                       name='flow_{}{}'.format(period, commodity))
                     for period in periods_range for commodity in commodities_range]).reshape(len(periods_range),
                                                                                              len(commodities_range))
    open_arc = np.array([subproblem.addVar(lb=0., ub=1., vtype=grb.GRB.BINARY, name='open_arc{}'.format(period),
                                           obj=fixed_cost[period]) for period in periods_range])
    subproblem.update()
    subproblem.addConstr(grb.quicksum(open_arc) <= 1., name='all_periods')
    capacity_constr = [subproblem.addConstr(grb.LinExpr(demand[t, :].tolist(), flow[t, :].tolist()) <=
                                            capacity * grb.quicksum(open_arc[l] for l in xrange(0, t+1)),
                                            name='cap_{}'.format(t))
                       for t in periods_range]
    setup_constrs = [subproblem.addConstr(flow[t, c] <= min(1, capacity / demand[t, c]) *
                                          grb.quicksum(open_arc[l] for l in xrange(0, t+1)),
                                          name='setup_{}{}'.format(t, c))
                     for c in commodities_range for t in periods_range]
    subproblem.update()
    subproblem.params.outputFlag = 0
    # subproblem.write('subproblem.lp')
    subproblem.optimize()
    subproblem_sol.objective_value = subproblem.getObjective().getValue()
    subproblem_sol.flow = np.array([flow[t, c].X for t in periods_range for c in
                                    commodities_range]).reshape(len(periods_range), len(commodities_range))
    subproblem_sol.open_period = -1
    for t in periods_range:
        if open_arc[t].X > 10e-6:
            subproblem_sol.open_period = t
            break
    return subproblem_sol


def solve_subproblem(lagrange_cost, demand, fixed_cost, capacity, period, obj=0, open_period=-1, best_fixed_charge=0,
                     subproblem_sol=namedtuple('Solution', 'objective_value flow open_period')):
    """
    Solves the arc subproblem given the input parameters
    :param lagrange_cost:   Lagrangian cost, equal to cost ± node dual values
    :param demand:          Demand of each commodity in each period
    :param fixed_cost:      Cost to open the arc, varies by period
    :param capacity:        Arc capacity
    :return:                subproblem_sol, consisting of an objective value, a flow vector and an integer showing which
                            period the arc opens. (-1 indicates that the arc never opens)
    """


    if period == -1:
        if open_period == -1:
            subproblem_sol.flow = np.zeros(shape=(len(fixed_cost), np.shape(demand)[1]))
            subproblem_sol.objective_value = 0.
        elif open_period > 0:
            subproblem_sol.flow[0:open_period, :] = 0.
        subproblem_sol.open_period = open_period
        return subproblem_sol
    elif period == len(fixed_cost) - 1:
        subproblem_sol.objective_value = 0.
        subproblem_sol.open_period = -1.
        subproblem_sol.flow = np.zeros(shape=(demand.shape))
        best_fixed_charge, obj = 0., 0.

    knapsack_sol = solve_linear_knapsack(lagrange_cost=lagrange_cost[period, :],
                                         demand=demand[period, :], capacity=capacity)

    # We can store the solution because the process is greedy. If we don't like it, we remove it later
    if knapsack_sol.objective_value < 0.:
        obj += knapsack_sol.objective_value
        subproblem_sol.flow[period, :] = knapsack_sol.solution_vector

    fixed_charge = 0 if open_period < 0 else fixed_cost[open_period]
    if obj + (fixed_cost[period] - fixed_charge) < 0:
        obj += fixed_cost[period] - best_fixed_charge
        open_period = period
        subproblem_sol.objective_value += obj
        best_fixed_charge = fixed_cost[open_period]
        obj = 0.

    period -= 1
    return solve_subproblem(lagrange_cost, demand, fixed_cost, capacity, period, obj, open_period, best_fixed_charge,
                            subproblem_sol)


def solve_linear_knapsack(lagrange_cost, demand, capacity):
    """
    Solves a linear knapsack problem for each time period
    :param lagrange_cost:   The lagrangian cost of the flow (can be negative)
    :param demand:          Demand of all commodities for a certain period
    :param capacity:        Knapsack capacity
    :return:                returns a knapsack solution named tuple, that contains the optimal objective value of the
                            knapsack, and the solution vector of it
    """
    solution_vector, ratios = np.zeros(shape=(demand.shape)), np.zeros(shape=(demand.shape))
    used_capacity, sol_value, frac = 0., 0., 0.

    ratios = lagrange_cost / demand
    sorted_indexes = np.argsort(ratios)
    commodities_range = xrange(len(demand))

    for commodity in commodities_range:
        commodity_index = sorted_indexes[commodity]
        commodity_bound = 1 if capacity > demand[commodity_index] else capacity / demand[commodity_index]
        if ratios[commodity_index] < 0:
            if used_capacity + demand[commodity_index] * commodity_bound <= capacity:
                used_capacity += demand[commodity_index] * commodity_bound
                sol_value += lagrange_cost[commodity_index] * commodity_bound
                solution_vector[commodity_index] = commodity_bound
            else:
                frac = (capacity - used_capacity) / (demand[commodity_index])
                frac = min(frac, commodity_bound)
                solution_vector[commodity_index] = frac
                sol_value += frac * lagrange_cost[commodity_index]
                break
        else:
            break
    # knapsack_sol.solution_vector = solution_vector
    # knapsack_sol.objective_value = sol_value
    return Solution(sol_value, solution_vector)


def test_linear_knapsack():
    lagrange_cost, demand, capacity = np.array([-1., -2., 3., -4., -5.]), np.array([2., 3., 4., 5., 6.]), 5
    solve_linear_knapsack(lagrange_cost, demand, capacity)


def test_supbroblem():
    lagrange_cost = np.array([-408, -1229, -272, -284, -249, -1250,
                              -661, -1060, -1307, -1063, -1642, -1796,
                              -710, -1065, -538, -1640, -1085, -1561,
                              -350, -576, -1617, -1713, -1648, -1738,
                              -768, -1554, -1215, -942, -681, -1403], dtype=float).reshape((5, 6))
    demand = np.array([26, 43, 15, 47, 29, 19,
                       80, 12, 87, 31, 78, 61,
                       89, 40, 39, 28, 90, 56,
                       45, 99, 56, 21, 35, 24,
                       92, 86, 56, 80, 33, 47], dtype=float).reshape((5, 6))
    fixed_cost = np.array([5000, 4000, 3000, 2000, 1000], dtype=float)
    capacity = np.ones(5) * 30
    subproblem_sol.objective_value = 0
    subproblem_sol.open_period = -1
    subproblem_sol.flow = np.zeros(shape=(demand.shape))
    solve_subproblem(lagrange_cost, demand, fixed_cost, capacity, period=len(fixed_cost) - 1)


if __name__ == '__main__':
    test_supbroblem()