from __future__ import print_function
import pygraphviz
import re
import numpy
from pysb.integrate import odesolve
import matplotlib.cm as cm
import matplotlib.colors as colors
import sympy
import pandas
import functools
import os
from helper_functions import parse_name
from scipy.optimize import curve_fit
import seaborn as sns
from collections import OrderedDict
import networkx as nx


class OrderedGraph(nx.DiGraph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))


def f2hex_edges(fx, vmin, vmax):
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=sns.diverging_palette(240, 10, as_cmap=True))
    rgb = f2rgb.to_rgba(fx)[:3]
    return '#%02x%02x%02x' % tuple([255 * fc for fc in rgb])


def f2hex_nodes(fx, vmin, vmax, midpoint):
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True))
    rgb = f2rgb.to_rgba(fx)[:3]
    return '#%02x%02x%02x' % tuple([255 * fc for fc in rgb])


def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


def change_node_colors(node, color, graph):
    n = graph.node[node]
    n['fillcolor'] = color
    return


def change_edge_colors(edge, color, graph):
    n = graph.get_edge_data(*edge)
    n.attr['color'] = color
    return


def sig_apop(t, f, td, ts):
    """Return the amount of substrate cleaved at time t.

    Keyword arguments:
    t -- time
    f -- is the fraction cleaved at the end of the reaction
    td -- is the delay period between TRAIL addition and half-maximal substrate cleavage
    ts -- is the switching time between initial and complete effector substrate  cleavage
    """
    return f - f / (1 + numpy.exp((t - td) / (4 * ts)))


class FluxVisualization:
    def __init__(self, model):
        self.model = model
        self.tspan = None
        self.y = None
        self.parameters = None
        self.sp_graph = None
        self.colors_time_edges = None
        self.colors_time_nodes = None

    def visualize(self, fig_path='', tspan=None, parameters=None, verbose=False):
        if verbose:
            print("Solving Simulation")

        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")

        if parameters is not None:
            # accept vector of parameter values as an argument
            if len(parameters) != len(self.model.parameters):
                raise Exception("parameters must be the same length as model.parameters")
            if not isinstance(parameters, numpy.ndarray):
                parameters = numpy.array(parameters)
        else:
            # create parameter vector from the values in the model
            parameters = numpy.array([p.value for p in self.model.parameters])

        new_pars = dict((p.name, parameters[i]) for i, p in enumerate(self.model.parameters))
        self.parameters = new_pars

        self.y = odesolve(self.model, self.tspan, self.parameters)

        if verbose:
            print("Creating graph")
        self.species_graph()

        self.sp_graph.add_node('t',
                               label='time',
                               shape='oval',
                               fillcolor='white', style="filled", color="transparent",
                               fontsize="50",
                               margin="0,0",
                               pos="20,20!")
        nx.write_gexf(self.sp_graph, '/home/oscar/Desktop/gexf_try1.gexf', version="1.2draft")

        # self.edges_colors(self.y)
        # self.nodes_colors(self.y)
        #
        # if os.path.exists(fig_path):
        #     directory = fig_path
        # else:
        #     directory = os.getcwd() + '/visualizations'
        #     os.makedirs(directory)
        #
        # if verbose:
        #     "Generating images"
        # for kx, time in enumerate(self.tspan):
        #     self.sp_graph.get_node('t').attr['label'] = 'time:' + ' ' + '%d' % time + ' ' + 'sec'
        #     map(functools.partial(change_edge_colors, graph=self.sp_graph), list(self.colors_time_edges.index),
        #         list(self.colors_time_edges.iloc[:, kx]))
        #     map(functools.partial(change_node_colors, graph=self.sp_graph), list(self.colors_time_nodes.index),
        #         list(self.colors_time_nodes.iloc[:, kx]))
        #     self.sp_graph.draw(directory + '/file' + '%03d' % kx + '.png')

    def edges_colors(self, y):
        all_rate_colors = {}
        rxns_matrix = numpy.zeros((len(self.model.reactions_bidirectional), len(self.tspan)))
        for idx, reac in enumerate(self.model.reactions_bidirectional):
            rate_reac = reac['rate']
            for p in self.parameters:
                rate_reac = rate_reac.subs(p, self.parameters[p])
            variables = [atom for atom in rate_reac.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
            react_rate = func(*args)
            rxns_matrix[idx] = react_rate

        max_all_times = [max(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]
        min_all_times = [min(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]

        for rxn in self.model.reactions_bidirectional:
            rate = rxn['rate']
            for p in self.parameters:
                rate = rate.subs(p, self.parameters[p])
            variables = [atom for atom in rate.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
            react_rate = func(*args)
            rate_colors = map(f2hex_edges, react_rate, min_all_times, max_all_times)
            for rctan in rxn['reactants']:
                for pro in rxn['products']:
                    all_rate_colors[('s' + str(rctan), 's' + str(pro))] = rate_colors
        all_colors = pandas.DataFrame(all_rate_colors).transpose()
        self.colors_time_edges = all_colors
        return

    def nodes_colors(self, y):
        all_rate_colors = {}
        initial_conditions_values = [ic[1].value for ic in self.model.initial_conditions]
        # cparp_info = curve_fit(sig_apop, self.tspan, y['cPARP'], p0=[100, 100, 100])[0]
        # midpoint = sig_apop(cparp_info[1], cparp_info[0], cparp_info[1], cparp_info[2])
        max_ic = max(initial_conditions_values)

        for idx in range(len(self.model.species)):
            node_colors = map(functools.partial(f2hex_nodes, vmin=0, vmax=max_ic, midpoint=max_ic/2), y['__s%d' % idx])
            all_rate_colors['s%d' % idx] = node_colors
        all_nodes_colors = pandas.DataFrame(all_rate_colors).transpose()
        self.colors_time_nodes = all_nodes_colors
        return

    def species_graph(self):

        graph = OrderedGraph(ankdir="LR")
        for idx, cp in enumerate(self.model.species):
            species_node = 's%d' % idx

            graph.add_node(species_node,
                           label=parse_name(self.model.species[idx]),
                           shape="Mrecord",
                           fillcolor="#ccffcc", style="filled", color="transparent",
                           fontsize="35",
                           margin="0.06,0", start="1990-01-01", end="1993-01-01")

        for reaction in self.model.reactions_bidirectional:
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            attr_reversible = {'dir': 'both', 'arrowtail': 'empty', 'arrowsize': 2, 'penwidth': 5} if reaction[
                'reversible'] else {'arrowsize': 2, 'penwidth': 5}
            for s in reactants:
                for p in products:
                    r_link(graph, s, p, **attr_reversible)
        self.sp_graph = graph
        print(graph.edges(data=True))
        return self.sp_graph


def run_flux_visualization(model, tspan, fig_path='', parameters=None, verbose=False):
    fv = FluxVisualization(model)
    fv.visualize(fig_path, tspan, parameters, verbose)

from earm.lopez_embedded import model

run_flux_visualization(model, [1])