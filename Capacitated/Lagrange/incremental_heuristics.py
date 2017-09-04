from __future__ import division
"""
Heuristic Procedures for the capacitated multi period network design problem
Dec 2016, Ioannis Fragkos
"""
from helpers import get_2d_index, read_data

import numpy as np
import gurobipy as grb
DEBUG = False


class Model:
    """
    Class that hold the single-period linear network design models
    """
    def __init__(self, data, t):
        model = grb.Model('linear_nf{}'.format(t))
        arcs, coms, demand = data.arcs.size, data.commodities, data.demand[t, :]
        arc_org, arc_dest = data.arc_org, data.arc_dest
        nodes, b, cap = data.nodes, data.b, data.capacity
        obj_coeffs = np.outer(demand, data.variable_cost).flatten()

        # indexing: commodity, arc origin, arc destination
        indexes = [(i, j, k) for i in xrange(coms) for (j, k) in zip(
            arc_org, arc_dest)]

        model_vars = model.addVars(indexes, ub=1., obj=obj_coeffs, name='flow')
        model.update()

        cap_constrs = model.addConstrs((
            grb.LinExpr(demand, model_vars.select('*', i, j)) <= cap[u]
            for u, (i, j) in enumerate(zip(arc_org, arc_dest))), name='cap')

        flow_cons = model.addConstrs((
            model_vars.sum(k, '*', i) - model_vars.sum(k, i, '*') == b[k, i]
            for k in xrange(coms) for i in xrange(nodes)), name='flow')

        model.update()
        model.params.OutputFlag = 0
        model.params.threads = 2
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
                print message
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
                print message
            return

        # Here model is feasible
        self.status = 'feasible'
        message = 'arc: {} period: {} objective: {}'.format(
            arc_later, t, model.ObjVal
        ) if arc_later is not None else 'period: {} objective: {}'.format(
            t, model.ObjVal)
        if DEBUG:
            print message

    def open_later(self, data, arc):
        """
        Heuristic that tries to open arc in period t+1 instead of t
        If profitable, it implements the change. Otherwise, it does not
        Ioannis Fragkos December 2016
        :param arc:     arc that we want to check
        :param data:    problem data
        :return:        modified model object
        """
        t, base_obj = self.period, self.objective
        diff_cost = 0 if t == data.periods - 1 else data.fixed_cost[t + 1, arc]
        delta_fc = data.fixed_cost[t, arc] - diff_cost

        self.solve(data=data, arc_later=arc)
        model = self.model

        if model.status == 2:
            obj_val = model.ObjVal
            # If increase of flow cost is greater than reduction of arc cost
            # move the arc later
            if obj_val - base_obj < delta_fc:
                self.objective = obj_val
                self.status = 'feasible'
                self.vals = np.array(
                    model.getAttr('x', model.getVars())).reshape(
                    data.commodities, data.arcs.size)
                self.benefit = delta_fc - (obj_val - base_obj)
            else:
                self.modify_arc(data, arc, mode='open')


    def modify_arc(self, data, arc, mode):
        """
        Modifies model so that it closes arc
        :param data:    Problem data
        :param arc:     Arc that is going to be closed
        :param mode:    Can be either 'open' or 'close'
        :return:        Modified model object
        """
        model = self.model
        start, end = data.arc_org[arc], data.arc_dest[arc]
        mdl_vars = self.vars.select('*', start, end)
        level = 0. if mode is 'close' else 1.
        model.setAttr('ub', mdl_vars, [level] * len(mdl_vars))
        model.update()

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


def fixing_heuristic(data, models, inc_sol, inc_obj):
    """
    Simple fixing heuristic that takes a rounded solution and solves the LP
    :param data:        Model data
    :param models:      Series of single-period models
    :param inc_sol:     Incumbent solution vector (arc states)
    :param inc_obj:     Incumbent objective
    :return:            (hopefully) improved inc_sol, inc_obj
    """
    this_obj = 0.
    for t in xrange(data.periods):
        model = models[t]
        for arc in xrange(data.arcs.size):
            if inc_sol[:t + 1, arc].sum(axis=0) == 0:
                model.modify_arc(data, arc, 'close')
            else:
                this_obj += data.fixed_cost[t, arc]
        model.model.optimize()
        if model.model.status == 2:
            this_obj += model.model.objVal
        else:
            this_obj = 10e12
            break
    print 'Rounded solution objective: \t{}'.format(this_obj)
    if this_obj < inc_obj:
        print 'success!'
    else:
        for t in xrange(data.periods):
            model = models[t].model
            mdl_vars = model.getVars()
            model.setAttr('x', mdl_vars, [1.]*len(mdl_vars))
    return inc_sol, inc_obj

def test():
    root_path = '../../../MPMCFP_DataGen/'
    data_path_c = 'c_Instances_Dec_Fixed_Cost/'
    c_trial, r_trial = 'c33_R_H_10.dow', 'r01.1_R_H_10.dow'
    filename = root_path + data_path_c
    filename += c_trial
    data = read_data(filename)
    models = []
    for t in xrange(data.periods - 1):
        model = Model(data, t)
        delta = np.ones(shape=data.commodities)
        model.solve(data)
        print 'period: {} model status: {}'.format(t, model.status)
        models.append(model)
        # for arc in xrange(data.arcs.size):
        #     model.open_later(data, arc)


if __name__ == '__main__':
    test()
