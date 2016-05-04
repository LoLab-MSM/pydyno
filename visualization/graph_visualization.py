#!/usr/bin/env python
import networkx as nx
import pysb
import pysb.bng
from pysb.integrate import odesolve
from collections import OrderedDict
import re
from earm.lopez_embedded import model
import numpy
import matplotlib.cm as cm
import matplotlib.colors as colors
import sympy
import csv
import pandas
import functools
import os


class OrderedGraph(nx.DiGraph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


f = open('/home/oscar/Documents/tropical_project/parameters_5000/pars_embedded_4052.txt')
data = csv.reader(f)
parames = {}
for i in data: parames[i[0]] = float(i[1])

tspan = numpy.linspace(0, 10000, 100)
y = odesolve(model, tspan, parames)


def f2hex(fx):
    norm = colors.Normalize(vmin=0, vmax=1)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('coolwarm'))
    rgb = f2rgb.to_rgba(fx)[:3]
    return '#%02x%02x%02x' % tuple([255 * fc for fc in rgb])


def nodes_colors(model):
    all_rate_colors = {}
    for idx, rxn in enumerate(model.reactions_bidirectional):
        rate = rxn['rate']
        for p in parames:
            rate = rate.subs(p, parames[p])
        args = []  # arguments to put in the lambdify function
        variables = [atom for atom in rate.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
        func = sympy.lambdify(variables, rate, modules=dict(sqrt=numpy.lib.scimath.sqrt))
        for l in variables:
            args.append(y[str(l)])
        react_rate = func(*args)
        rate_data = react_rate / max(react_rate)
        rate_colors = map(f2hex, rate_data)
        all_rate_colors['r%d' % idx] = rate_colors
    all_colors = pandas.DataFrame(all_rate_colors).transpose()

    return all_colors


def reactions_graph(model):
    pysb.bng.generate_equations(model)

    graph = OrderedGraph()
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        slabel = re.sub(r'% ', r'%\\l', str(cp))
        slabel += '\\l'
        color = "#ccffcc"
        # color species with an initial condition differently
        if len([s for s in ic_species if s.is_equivalent_to(cp)]):
            color = "#aaffff"
        graph.add_node(species_node,
                       label=species_node,
                       shape="Mrecord",
                       fillcolor=color, style="filled", color="transparent",
                       fontsize="100",
                       width=".6", height=".6", margin="0.06,0")
    for j, reaction in enumerate(model.reactions):
        reaction_node = 'r%d' % j
        graph.add_node(reaction_node,
                       label=reaction_node,
                       shape="circle",
                       fillcolor="lightgray", style="filled", color="transparent",
                       fontsize="100",
                       width=".6", height=".6", margin="0.06,0")
        reactants = set(reaction['reactants'])
        products = set(reaction['products'])
        modifiers = reactants & products
        reactants = reactants - modifiers
        products = products - modifiers
        attr_reversible = {'penwidth': 10}
        for s in reactants:
            r_link(graph, s, j, **attr_reversible)
        for s in products:
            r_link(graph, s, j, _flip=True, **attr_reversible)
        for s in modifiers:
            r_link(graph, s, j, arrowhead="odiamond")
    return graph


def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 'r%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        print nodes
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


def change_node_colors(node, color, graph):
    n = graph.get_node(node)
    n.attr['fillcolor'] = color
    return

graph_nx = reactions_graph(model)

gra = nx.nx_agraph.to_agraph(graph_nx)
gra.layout(prog='dot', args="-Gsize=30.75,10.75\! -Gratio=fill -Grankdir=LR -Gdpi=200! -Gordering=in")
df = nodes_colors(model)


num_plots = len(tspan)

for kx in range(num_plots):
    map(functools.partial(change_node_colors, graph=gra), list(df.index), list(df.iloc[:, kx]))

    gra.draw('/home/oscar/Documents/tropical_project/tmp/file' + '%03d' % kx + '.png')


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

