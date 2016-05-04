from earm.lopez_embedded import model
import pysb.bng
import pygraphviz
import re
import csv
import numpy
from pysb.integrate import odesolve
import matplotlib.cm as cm
import matplotlib.colors as colors
import sympy
import pandas
import functools

f = open('/home/oscar/Documents/tropical_project/parameters_5000/pars_embedded_5400.txt')
data = csv.reader(f)
parames = {}
for i in data: parames[i[0]] = float(i[1])

tspan = numpy.linspace(0, 10000, 100)
y = odesolve(model, tspan, parames)

def f2hex_edges(fx):
    norm = colors.Normalize(vmin=0, vmax=1)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('coolwarm'))
    rgb = f2rgb.to_rgba(fx)[:3]
    return '#%02x%02x%02x' % tuple([255 * fc for fc in rgb])


class trop_visualization:
    def __initi__(self, model):
        self.model = model
        self.tspan = None
        self.y = None
        self.parameters = None

    def nodes_colors(self, y):
        all_rate_colors = {}
        for idx, rxn in enumerate(self.model.species):
            conc_data = y['__s%d' % idx] / max(y['__s%d' % idx])
            conc_colors = map(f2hex_colors, conc_data)
            all_rate_colors['s%d' % idx] = conc_colors
        all_colors = pandas.DataFrame(all_rate_colors).transpose()

        return all_colors


    def edges_colors(self):
        all_rate_colors = {}
        for idx, rxn in enumerate(self.model.reactions):
            rate = rxn['rate']
            for p in parames:
                rate = rate.subs(p, self.parameters[p])
            args = []  # arguments to put in the lambdify function
            variables = [atom for atom in rate.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            for l in variables:
                args.append(y[str(l)])
            react_rate = func(*args)
            for rctan in rxn['reactants']:
                rate_total = 0
                for r_rxn in self.model.reactions:
                    if rctan in r_rxn['reactants']:
                        rate_total += r_rxn['rate']
                for p in parames:
                    rate_total = rate_total.subs(p, self.parameters[p])
                args = []  # arguments to put in the lambdify function
                variables = [atom for atom in rate_total.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
                func = sympy.lambdify(variables, rate_total, modules=dict(sqrt=numpy.lib.scimath.sqrt))
                for l in variables:
                    args.append(y[str(l)])
                total_react_rate = func(*args)

                rate_data_min = 0
                rate_data_max = 1

                rate_data = react_rate / total_react_rate

                rate_colors = map(functools.partial(f2hex_edges, vmin=rate_data_min, vmax=rate_data_max), rate_data) # map(f2hex_edges, rate_data)
                for pro in rxn['products']:
                    all_rate_colors[('s' + str(rctan), 's' + str(pro))] = rate_colors
        all_colors = pandas.DataFrame(all_rate_colors).transpose()


def f2hex_edges(fx, vmin, vmax):
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('coolwarm'))
    rgb = f2rgb.to_rgba(fx)[:3]
    return '#%02x%02x%02x' % tuple([255 * fc for fc in rgb])


def f2hex_colors(fx):
    norm = colors.Normalize(vmin=0, vmax=1)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('Oranges'))
    rgb = f2rgb.to_rgba(fx)[:3]
    return '#%02x%02x%02x' % tuple([255 * fc for fc in rgb])


def nodes_colors(model):
    all_rate_colors = {}
    for idx, rxn in enumerate(model.species):
        conc_data = y['__s%d' % idx] / max(y['__s%d' % idx])
        conc_colors = map(f2hex_colors, conc_data)
        all_rate_colors['s%d' % idx] = conc_colors
    all_colors = pandas.DataFrame(all_rate_colors).transpose()

    return all_colors


def edges_colors(model):
    all_rate_colors = {}
    for idx, rxn in enumerate(model.reactions):
        rate = rxn['rate']
        for p in parames:
            rate = rate.subs(p, parames[p])
        args = []  # arguments to put in the lambdify function
        variables = [atom for atom in rate.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
        func = sympy.lambdify(variables, rate, modules=dict(sqrt=numpy.lib.scimath.sqrt))
        for l in variables:
            args.append(y[str(l)])
        react_rate = func(*args)
        for rctan in rxn['reactants']:
            rate_total = 0
            for r_rxn in model.reactions:
                if rctan in r_rxn['reactants']:
                    rate_total += r_rxn['rate']
            for p in parames:
                rate_total = rate_total.subs(p, parames[p])
            args = []  # arguments to put in the lambdify function
            variables = [atom for atom in rate_total.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate_total, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            for l in variables:
                args.append(y[str(l)])
            total_react_rate = func(*args)

            rate_data_min = 0
            rate_data_max = 1

            rate_data = react_rate / total_react_rate

            rate_colors = map(functools.partial(f2hex_edges, vmin=rate_data_min, vmax=rate_data_max), rate_data) # map(f2hex_edges, rate_data)
            for pro in rxn['products']:
                all_rate_colors[('s' + str(rctan), 's' + str(pro))] = rate_colors
    all_colors = pandas.DataFrame(all_rate_colors).transpose()

    return all_colors


def species_graph(model):
    pysb.bng.generate_equations(model)

    graph = pygraphviz.AGraph(directed=True, rankdir="LR", labelloc='t', fontsize=50, label='EARM flux')
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        label = re.sub(r'% ', r'%\\l', str(cp))
        label += '\\l'
        rtn_product = False
        rtn = False
        prdt = False

        all_reactants = []
        for rctn in model.reactions:
            if rctn['reverse'][0] is False:
                all_reactants.append(rctn['reactants'])
        all_reactants_flat = list(set([item for sublist in all_reactants for item in sublist]))

        all_products = []
        for rctn in model.reactions:
            if rctn['reverse'][0] is False:
                all_products.append(rctn['products'])
        all_products_flat = list(set([item for sublist in all_products for item in sublist]))

        if i in all_reactants_flat:
            rtn = True
        if i in all_products_flat:
            prdt = True

        if rtn and prdt:
            rtn_product = True

        if rtn_product:
            color = "Cyan"
        elif rtn:
            color = "#FF00F6"
        elif prdt:
            color = "DarkGreen"
        else:
            pass

        graph.add_node(species_node,
                       label=species_node,
                       shape="Mrecord",
                       fillcolor=color, style="filled", color="transparent",
                       fontsize="50",
                       margin="0.06,0")
    for i, reaction in enumerate(model.reactions):
        reactants = set(reaction['reactants'])
        products = set(reaction['products'])
        attr_reversible = {'arrowsize': 2, 'penwidth': 5}
        for s in reactants:
            for p in products:
                r_link(graph, s, p, **attr_reversible)

    return graph


def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


sp_graph = species_graph(model)
sp_graph.layout(prog='dot',
                args="-Gstart=50 -Gesep=1  -Gsplines=true -Gsize=30.75,10.75\! -Gratio=fill -Grankdir=LR -Gdpi=100! -Gordering=in")
sp_graph.add_node('t',
                  label='time',
                  shape='oval',
                  fillcolor='white', style="filled", color="transparent",
                  fontsize="50",
                  margin="0.06,0",
                  pos="20,20!",
                  pin='true')

df_nodes = nodes_colors(model)
df_edges = edges_colors(model)


def change_node_colors(node, color, graph):
    n = graph.get_node(node)
    n.attr['fillcolor'] = color
    return


def change_edge_colors(edge, color, graph):
    n = graph.get_edge(*edge)
    n.attr['color'] = color
    return


for kx, time in enumerate(tspan):
    sp_graph.get_node('t').attr['label'] = 'time:' + ' ' + '%d' % time + ' ' + 'sec'
    # map(functools.partial(change_node_colors, graph=sp_graph), list(df_nodes.index), list(df_nodes.iloc[:, kx]))
    map(functools.partial(change_edge_colors, graph=sp_graph), list(df_edges.index), list(df_edges.iloc[:, kx]))
    sp_graph.draw('/home/oscar/Documents/tropical_project/species_flow_5400/file' + '%03d' % kx + '.png')


#
# sp_graph = species_graph(model)
#
# sp_graph.layout(prog='dot', args="-Gstart=50  -Gsplines=true -Gsize=30.75,10.75\! -Gratio=fill -Grankdir=LR -Gdpi=100! -Gordering=in")
# sp_graph.draw('/home/oscar/Desktop/file_sp.png')
