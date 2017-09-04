from __future__ import division
__author__ = 'ioannis'
# usr/bin/python
# noinspection PyUnresolvedReferences

from gurobipy import tuplelist, GRB, Model, quicksum, Env
from collections import namedtuple
from itertools import product
from sys import argv

DEBUG = False
PRINT_VARS = False
EPSILON = 10e-4

data = namedtuple('data', 'commodities nodes arcs  periods capacity fixed_cost variable_cost demand')


def read_data(filename):
    with open(filename, 'r') as f:
        count, arcs = 0, []
        data.variable_cost, data.capacity, data.variable_cost, data.fixed_cost, data.demand = {}, {}, {}, {}, {}
        line = f.readline().split()
        data.nodes, no_of_arcs, data.commodities, data.periods = \
            xrange(1, int(line[0]) + 1), int(line[1]), xrange(1, int(line[2]) + 1), xrange(1, int(line[3]) + 1)
        for line in f:
            line = line.split()
            arc = int(line[0]), int(line[1])
            if count < no_of_arcs:
                arcs.append(arc)
                data.variable_cost[arc], data.capacity[arc] = \
                    float(line[2]), float(line[3])
                for period in data.periods:
                    data.fixed_cost[(arc, period)] = float(line[3 + period])
                count += 1
            else:
                commodity_no = count - no_of_arcs + 1
                for period in data.periods:
                    data.demand[(commodity_no, arc, period)] = float(line[1 + period])
                count += 1
        data.arcs = tuplelist(arcs)


def make_model(strong_inequalities=False, relax=False, callback=False, hascapacity=1):
    # Relabel data
    commodities = data.commodities
    arcs = data.arcs
    capacity = data.capacity
    variable_cost = data.variable_cost
    fixed_cost = data.fixed_cost
    nodes = data.nodes
    demand = data.demand
    periods = data.periods

    # Create optimization model
    env = Env(logfilename="")
    m = Model('multi-period-netflow', env)

    # Create variables
    flow, arc_open = {}, {}
    for t in periods:
        for i, j in arcs:
            arc_open[i, j, t] = m.addVar(vtype=GRB.BINARY, lb=0.0, ub=1.0, obj=fixed_cost[(i, j), t],
                                        name='open_{0:d}_{1:d}_{2:d}'.format(i, j, t))
            for h in commodities:
                origin, destination = [key_val[1] for key_val in demand.keys() if key_val[0] == h][0]
                upper = capacity[i, j] if has_capacity else demand[(h, (origin, destination), t)]
                flow[h, i, j, t] = m.addVar(obj=variable_cost[i, j], name='flow_{0:d}_{1:d}_{2:d}_{3:d}'.format(h, i, j, t))
    m.update()

    # Arc capacity constraints and unique arc setup constraints
    constrs = []
    for (i, j) in arcs:
        m.addConstr(quicksum(arc_open[i, j, l] for l in range(1, len(data.periods) + 1)) <= 1,
                    'unique_setup{0:d}_{1:d}'.format(i, j))
        for t in periods:
            if not hascapacity:
                capacity[i, j] = sum(demand[i] for i in demand.keys() if i[2] == t)
            m.addConstr(quicksum(flow[h, i, j, t] for h in commodities) <=
                        capacity[i, j] * quicksum(arc_open[i, j, s] for s in xrange(1, t + 1)),
                        'cap_{0:d}_{1:d}_{2:d}'.format(i, j, t))
            if not callback and strong_inequalities:
                for (commodity, (origin, destination), period) in demand:
                    if period == t:
                    	constrs.append(
                        m.addConstr(flow[commodity, i, j, t] <=
                                    demand[commodity, (origin, destination), period] *
                                    quicksum(arc_open[i, j, l] for l in range(1, t + 1)),
                                    name='strong_com{0:d}_{1:d}-{2:d}_per{3:d}'.format(commodity, i, j, t)))

    # Flow conservation constraints
    for (commodity, (origin, destination), period) in demand:
        for j in nodes:
            if j == origin:
                node_demand = demand[commodity, (origin, destination), period]
            elif j == destination:
                node_demand = -demand[commodity, (origin, destination), period]
            else:
                node_demand = 0
            h = commodity
            m.addConstr(
                - quicksum(flow[h, i, j, period] for i, j in arcs.select('*', j)) +
                quicksum(flow[h, j, k, period] for j, k in arcs.select(j, '*')) == node_demand,
                'node_{0:d}_{1:d}_{2:d}'.format(h, j, period)
            )

    m.update()

    # Compute optimal solution
    m.setParam("TimeLimit", 7200)
    # m.params.NodeLimit = 1
    # m.params.cuts = 0
    # m.setParam("Threads", 2)
    m.setAttr('Lazy', constrs, [3]*len(constrs))    
    # m.write("eyes.lp")
    #
    try:
        if strong_inequalities:
            if not relax:
                # m.setParam("NodeLimit", 1000000)
                # m.params.Cuts = 0                
                if callback:
                    print 'callback in action! :)'
                    m.params.preCrush = 1
                    m.update()
                    m._vars = m.getVars()
                    m.optimize(strong_inequalities_callback)
                else:
                    m.optimize(time_callback)
            else:
                m = m.relax()
                m.optimize(time_callback)

        else:
            m.optimize(time_callback)
        if PRINT_VARS:
            for var in m.getVars():
                if str(var.VarName[0]) == 'f' and var.X > 0.0001:
                    name = var.VarName.split('_')
                    print 'arc: \t {} \t commodity: {} \t period: {} \t value: \t {}'.format((
                                int(name[2]), int(name[3])), int(name[1]),
                                int(name[4]), var.x)
            # Grab the positive flows and see how many variables open during the first period
            positive_flows = [var for var in m.getVars() if var.VarName[0] == 'o' and var.X > 0.5]
            first_period_arcs = sum([var.X for var in positive_flows if int(var.VarName.split('_')[3]) == 1])
            print '% of arcs that open in first period: {}%'.format(100*first_period_arcs/len(positive_flows))
            print '% of arcs that are utilized: {}%'.format((100.*len(positive_flows))/len(data.arcs))
            objective = m.getObjective().getValue()
            fixed_cost_percentage = sum([fixed_cost[(i, j), t]*arc_open[i, j, t].X
                                         for i, j in data.arcs for t in data.periods]) / objective
            print 'Fixed cost percentage: {}%'.format(fixed_cost_percentage*100.)
            for var in m.getVars():
                if str(var.VarName[0]) == 'o' and var.X > 0.0001:
                    name = var.VarName.split('_')
                    print 'Arc: \t {} \t Period: {} \t Value: \t {}'.format((int(name[1]), int(name[2])), int(name[3]), var.X)
            # m.write('trial2.lp')
    except:
        if m.status == GRB.status.INFEASIBLE and DEBUG:
            print 'Infeasible model. Computing IIS..'
            m.computeIIS()
            m.write('trial.ilp')


def time_callback(model, where):
    if where != GRB.Callback.POLLING:
        elapsed_time = model.cbGet(GRB.Callback.RUNTIME)
        # if 3600-15 < elapsed_time < 3600 + 15 and where == GRB.Callback.MIP:
        #     message = 'Log: {}\t {} \t {}'.format(
        #         model.cbGet(
        #             GRB.Callback.MIP_OBJBST), 
        #         model.cbGet(
        #             GRB.Callback.MIP_OBJBND), 
        #     elapsed_time)
        #     model.message(message)
        if elapsed_time > model.params.TimeLimit:
            print 'Time limit exceeded caught via callback'
            model.terminate()


def strong_inequalities_callback(model, where):
    if where == GRB.callback.MIPNODE:
        status = model.cbGet(GRB.callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            for (i, j), t in product(data.arcs, data.periods):
                for (commodity, (origin, destination), period) in data.demand:
                    if period == t:
                        flow = model.getVarByName('flow_{}_{}_{}_{}'.format(commodity, i, j, period))
                        flow_val = model.cbGetNodeRel(flow)
                        arc_open, arc_open_val = 0., 0.
                        for l in xrange(1, period + 1):
                            arc_open += model.getVarByName('open_{}_{}_{}'.format(i, j, l))
                            arc_open_val += model.cbGetNodeRel(model.getVarByName('open_{}_{}_{}'.format(i, j, l)))
                        if flow_val > data.demand[commodity, (origin, destination),
                                                  period] * arc_open_val + EPSILON:
                            model.cbCut(flow - (data.demand[commodity, (origin, destination), period]) * arc_open <= 0.)


def main(filename, hascapacity=1):
    read_data(filename)
    make_model(strong_inequalities=True, relax=False, callback=False, hascapacity=hascapacity)


if __name__ == '__main__':
    has_capacity, filename = 1, 'r03.1_R_H_20.dow'
    if len(argv) > 1:
        filename = argv[1]
    if len(argv) > 2:
        has_capacity = int(argv[2])
    main(filename=filename, hascapacity=has_capacity)
