from __future__ import division
"""
Heuristic Procedures for the capacitated multi period network design problem
Sept 2018, Ioannis Fragkos
"""
from helpers import get_2d_index, read_data

import numpy as np
import gurobipy as grb
DEBUG = False


def recover_solution_data(incumbent_arcs, data):
    """
    Uses the arcs to recover the full solution of the network design model
    Ioannis Fragkos June 2018

    incumbent_arcs:     y_{ij}^t of the incumbent solution
    data:               regular data structure that holds all problem data

    Output: flows, arcs, fixed_cost_period, cont_cost_period_com
    flows: (Period, Arc, Commodity, Value)
    arcs : (Period, Arc)

    """

    periods, commodities, demand = data.periods, data.commodities, data.demand
    var_cost, fixed_cost = data.variable_cost, data.fixed_cost
    flows, arcs, cont_cost_period_com, fixed_cost_period = [], [], {}, []

    models = []
    for period in range(data.periods):
        model = Model(data, period)        
        models.append(model)

    # Calculate % of flow bifurcations &  % of arcs that open early/mid/late
    flows_dict, flows_bif, arcs_perc = {}, np.zeros(3), np.zeros(3)
    # Calculate how many commodities change their route over time
    flow_change, bucket = np.zeros(3), np.empty(periods, dtype=np.int)
    # Calculate percentage of variable cost and capacity utilization
    var_cost_perc, cap_util = np.zeros(3), np.zeros(3)
    """
    Logic:          Early   Mid-way     Late
    5 periods  --> {2}      ; {3, 4}   ; {5}
    10 periods --> {2, 3, 4}; {5, 6, 7}; {8, 9, 10}
    15 periods --> {2,...,6}; {7,...,10};{11,...,15}
    20 periods --> {2,...,7}; {8,...,14};{15,...,20}
    Divide the results in buckets where we report statistics
    """
    if periods == 10:
        bucket[0:4], bucket[4:7], bucket[7:] = 0, 1, 2
    elif periods == 15:
        bucket[0:6], bucket[6:10], bucket[10:] = 0, 1, 2
    elif periods == 20:
        bucket[0:7], bucket[7:14], bucket[14:] = 0, 1, 2
    elif periods == 5:
        bucket[0:2], bucket[2:4], bucket[4:] = 0, 1, 2
    elif periods == 40:
        bucket[0:13], bucket[13:27], bucket[27:] = 0, 1, 2
    elif periods == 60:
        bucket[0:20], bucket[20:40], bucket[40:] = 0, 1, 2
    elif periods == 80:
        bucket[0:26], bucket[26: 53], bucket[53:] = 0, 1, 2

    # Sum previous rows with each row
    all_arcs = incumbent_arcs.cumsum(axis=0)
    for t in range(periods):
        fixed_cost_period.append((t, np.inner(
            fixed_cost[t, :], incumbent_arcs[t, :])))
        current_UBs = np.tile(all_arcs[t, :], commodities).tolist()
        model = models[t].model
        model.setAttr("UB", model.getVars(), current_UBs)
        model.optimize()
        var_vals = model.getAttr("X", model.getVars())
        slacks = np.array(model.getAttr(
            'slack', model.getConstrs())[:data.arcs.size])
        slacks /= data.capacity
        period_util = np.sum((
            1. - slacks) * all_arcs[t, :]) / all_arcs[t, :].sum()
        cap_util[bucket[t]] += period_util
        for arc in range(data.arcs.size):
            if incumbent_arcs[t, arc] > 10e-4:
                arcs.append((t, arc))
            for com in range(commodities):
                idx = com * data.arcs.size + arc
                if var_vals[idx] > 10e-4:
                    flows.append((t, arc, com, var_vals[idx]))
                    if (t, com) in cont_cost_period_com:
                        cont_cost_period_com[t, com] += \
                            demand[t, com] * var_cost[arc] * var_vals[idx]
                        flows_dict[(t, com)].append(arc)
                    else:
                        cont_cost_period_com[t, com] = \
                            demand[t, com] * var_cost[arc] * var_vals[idx]
                        flows_dict[(t, com)] = [arc]
                        if var_vals[idx] <= 0.99:
                            # increase the number of coms that bifurcate
                            flows_bif[bucket[t]] += 1
    # REPORT
    # periods in bucket
    for t in range(3):
        p = (bucket == t).sum()
        flows_bif[t] /= (commodities * p)
    for com in range(commodities):
        path = set(flows_dict[(0, com)])
        for t in range(1, periods):
            if path != set(flows_dict[t, com]):
                flow_change[bucket[t]] += 1
                path = set(flows_dict[t, com])
    # REPORT
    for t in range(3):
        p = (bucket == t).sum()
        flow_change[t] /= (commodities * p)
        cap_util[t] /= p
    # Just for help
    fixed_cost_bucket = np.zeros(3)
    for t in range(periods):
        arcs_perc[bucket[t]] += incumbent_arcs[t, :].sum()
        var_cost_perc[bucket[t]] += sum(
            cont_cost_period_com[t, l] for l in range(commodities))
        fixed_cost_bucket[bucket[t]] += fixed_cost_period[t][1]
    # REPORT
    arcs_perc /= arcs_perc.sum()
    var_cost_perc /= (var_cost_perc + fixed_cost_bucket)

    print('% of arcs open: {}'.format(arcs_perc))
    print('% of variable cost: {}'.format(var_cost_perc))
    print('% of flow change: {}'.format(flow_change))
    print('% of flow bifurcation: {}'.format(flows_bif))
    print('% of capacity utilization: {}'.format(cap_util))

    write_to_file = False

    if write_to_file:
        with open(data.filename + '.sol', 'w') as o_file:
            print >> o_file, 'Instance Name: {}'.format(data.filename)
            print >> o_file, 'Period \t Arc \t Commodity \t Value'
            for flow in flows:
                print >> o_file, '{} \t {} \t {} \t {}'.format(
                    flow[0], flow[1], flow[2], flow[3])
            print >> o_file, '\nPeriod \t Arc'
            for arc in arcs:
                print >> o_file, '{} \t {}'.format(arc[0], arc[1])
            print >> o_file, '\n Period \t Cost \t (Fixed Costs)'
            for fcost in fixed_cost_period:
                print >> o_file, '{} \t {}'.format(fcost[0], fcost[1])
            print >> o_file, '\n Period \t Commodity \t Cost \t (Variable Costs)'
            for vkey, vcost in cont_cost_period_com.iteritems():
                print >> o_file, '{} \t {} \t {}'.format(
                    vkey[0], vkey[1], vcost)



class Model:
    """
    Class that holds the single-period linear network design models
    """
    def __init__(self, data, t):
        model = grb.Model('linear_nf{}'.format(t))
        arcs, coms, demand = data.arcs.size, data.commodities, data.demand[t, :]
        arc_org, arc_dest = data.arc_org, data.arc_dest
        nodes, b, cap = data.nodes, data.b, data.capacity
        obj_coeffs = np.outer(demand, data.variable_cost).flatten()

        # indexing: commodity, arc origin, arc destination
        indexes = [(i, j, k) for i in range(coms) for (j, k) in zip(
            arc_org, arc_dest)]

        try:

            model_vars = model.addVars(indexes, ub=1., obj=obj_coeffs, name='flow')
            model.update()

            cap_constrs = model.addConstrs((
                model_vars.sum('*', i, j) <= coms
                for (i, j) in enumerate(zip(arc_org, arc_dest))), name='cap')

            flow_cons = model.addConstrs((
                model_vars.sum(k, '*', i) - model_vars.sum(k, i, '*') == b[k, i]
                for k in range(coms) for i in range(nodes)), name='flow')
        except grb.GurobiError as e:
            print('Error code: {} : {}'.format(str(e.errno), str(e)))

        model.update()
        model.params.OutputFlag = 0
        model.params.threads = 1
        self.model = model
        self.vars = model_vars
        self.objective = 0.
        self.vals = np.zeros((data.commodities, data.arcs.size))
        self.status = 'unknown'
        self.period = t
        self.benefit = - 1

    def solve(self, data, arc_later=None):
        """
        Solves the model by taking into account which arcs are open and which
        are closed
        :param data:            Data object
        :param self:            This beloved object
        :param arc_later:       Arc number that is moved later
        :return:                If feasible, returns a solution and an
                                objective
                                Otherwise, it returns that the model is
                                infeasible
        """
        model, t = self.model, self.period
        if arc_later is not None:
            self.modify_arc(data, arc_later, mode='close')

        model.optimize()

        if model.status == 3:
            if arc_later is not None:
                self.modify_arc(data, arc_later, mode='open')
                message = \
                    'Removing arc {}, period {} makes' \
                    'the model infeasible'.format(arc_later, t)
            else:
                message = 'Model of period {} is infeasible'.format(t)
            if DEBUG:
                print(message)
            return

        # Something went wrong here, need to further catch this thing
        if model.status != 2:
            if arc_later is not None:
                self.modify_arc(data, arc_later, mode='open')
                message = 'unknown model status in period {}'.format(t)
            else:
                message = 'unknown model status in arc {}, period {}'.format(
                    arc_later, t)
            if DEBUG:
                print(message)
            return

        # Here model is feasible
        self.status = 'feasible'
        message = 'arc: {} period: {} objective: {}'.format(
            arc_later, t, model.ObjVal
        ) if arc_later is not None else 'period: {} objective: {}'.format(
            t, model.ObjVal)
        if DEBUG:
            print(message)

    def set_objective_and_bounds(self, data, flow_vars, open_vars):
        """
        Sets upper/lower bounds and objective cost for the model based
        on the input data (see below for description)
        :param data:            Problem data
        :param flow_vars:       Flow variables for period t (arc, com)
        :param open_vars:       Arc opening variables for period t
        :return:                Modified model object
        """
        coms = data.commodities
        flow_cost = np.outer(data.variable_cost, data.demand[self.period, :])
        model = self.model
        self.objective = np.inner(flow_cost.flatten(), flow_vars)

        model.setAttr('ub', model.getVars(), 
            np.repeat(open_vars, coms).tolist())

