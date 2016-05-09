import networkx as nx
import pysb
import pysb.bng
from pysb.integrate import odesolve
from collections import OrderedDict
import re
from earm.lopez_embedded import model
import numpy
import csv
import itertools
import matplotlib.cm as cm
import matplotlib.colors as colors
import sympy
import pandas
import functools


class OrderedGraph(nx.DiGraph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


f = open('/home/oscar/Documents/tropical_project/parameters_5000/pars_embedded_4052.txt')
data = csv.reader(f)
parames = {p[0]: float(p[1]) for p in data}

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


def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 'r%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = list(reversed(nodes))
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)
    if attrs.get('dir'):
        nodes = reversed(nodes)
        graph.add_edge(*nodes, **attrs)


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
                       label=slabel,
                       shape="Mrecord",
                       fillcolor=color, style="filled", color="transparent",
                       fontsize="100",
                       width=".6", height=".6", margin="0.06,0")
    rxn_nodes = [0] * len(model.reactions_bidirectional)
    for j, reaction in enumerate(model.reactions_bidirectional):
        reaction_node = 'r%d' % j
        rxn_nodes[j] = reaction_node
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
        attr_reversible = {'dir': 'both', 'arrowtail': 'empty', 'penwidth': 10} if reaction['reversible'] else {
            'penwidth': 10}
        for s in reactants:
            r_link(graph, s, j, **attr_reversible)
        for s in products:
            r_link(graph, s, j, _flip=True, **attr_reversible)
        for s in modifiers:
            r_link(graph, s, j, arrowhead="odiamond")
    return graph, rxn_nodes


gra, rxn_n = reactions_graph(model)
rxn_nodes_pairs = list(itertools.permutations(rxn_n, 2))

node_attrs = {'color': 'transparent',
              'fillcolor': '#aaffff',
              'fontsize': '100',
              'height': '.6',
              'margin': '0.06,0',
              'shape': 'Mrecord',
              'style': 'filled',
              'width': '.6'}

rxn_graph = nx.DiGraph()
rxn_graph.add_nodes_from(rxn_n, **node_attrs)

for pair in rxn_nodes_pairs:
    if nx.has_path(gra, *pair):
        if len(nx.shortest_path(gra, *pair)) == 3:
            if rxn_graph.has_edge(pair[0], pair[1]):
                rxn_graph[pair[0]][pair[1]]['dir'] = 'both'
                rxn_graph[pair[0]][pair[1]]['penwidth'] = 6
            elif rxn_graph.has_edge(pair[1], pair[0]):
                rxn_graph[pair[1]][pair[0]]['dir'] = 'both'
                rxn_graph[pair[1]][pair[0]]['penwidth'] = 6
            else:
                rxn_graph.add_edge(*pair, **{'penwidth': 6})


rxn_dot = nx.nx_agraph.to_agraph(rxn_graph)

rxn_dot.layout(prog='neato', args="-Goverlap=false -Gstart=50  -Gsplines=true -Gsize=30.75,10.75\! -Gratio=fill -Grankdir=TB -Gdpi=100! -Gordering=in")
df = nodes_colors(model)


def change_node_colors(node, color, graph):
    n = graph.get_node(node)
    n.attr['fillcolor'] = color
    return

num_plots = len(tspan)
for kx in range(num_plots):
    map(functools.partial(change_node_colors, graph=rxn_dot), list(df.index), list(df.iloc[:, kx]))
    rxn_dot.draw('/home/oscar/Documents/tropical_project/reactions_flow/file' + '%03d' % kx + '.png')

# ffmpeg -framerate 1/1 -i file%03d.png -c:v libx264 out.mp4
