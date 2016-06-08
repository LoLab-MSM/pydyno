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


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))


def f2hex_edges(fx, vmin=-0.99, vmax=0.99):
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=sns.diverging_palette(240, 10, as_cmap=True))
    rgb = [f2rgb.to_rgba(rate)[:3] for rate in fx]
    colors_hex = [0]*(len(rgb))
    for i, color in enumerate(rgb):
        colors_hex[i] = '#%02x%02x%02x' % tuple([255 * fc for fc in color])
    return colors_hex


def f2hex_nodes(fx, vmin, vmax, midpoint):
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True))
    rgb = [f2rgb.to_rgba(rate)[:3] for rate in fx]
    colors_hex = [0]*(len(rgb))
    for i, color in enumerate(rgb):
        colors_hex[i] = '#%02x%02x%02x' % tuple([255 * fc for fc in color])
    return colors_hex


def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


def change_node_colors(node, color, graph):
    n = graph.get_node(node)
    n.attr['fillcolor'] = color
    return


def change_edge_colors(edge, color, graph):
    n = graph.get_edge(*edge)
    n.attr['color'] = color
    return


def change_edge_size(edge, size, graph):
    n = graph.get_edge(*edge)
    n.attr['penwidth'] = size
    return


def magnitude(x):
    return numpy.floor(numpy.log10(x))


def range_normalization(x, min_x, max_x, a=0.1, b=15):
    return a+(x-min_x)*(b-a)/(max_x - min_x)


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
    mach_eps = numpy.finfo(float).eps

    def __init__(self, model):
        self.model = model
        self.tspan = None
        self.y = None
        self.param_dict = None
        self.sp_graph = None
        self.colors_time_edges = None
        self.colors_time_nodes = None
        self.size_time_edges = None

    def visualize(self, fig_path='', tspan=None, param_values=None, verbose=False):
        if verbose:
            print("Solving Simulation")

        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")

        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.model.parameters):
                raise Exception("param_values must be the same length as model.parameters")
            if not isinstance(param_values, numpy.ndarray):
                param_values = numpy.array(param_values)
        else:
            # create parameter vector from the values in the model
            param_values = numpy.array([p.value for p in self.model.parameters])

        self.param_dict = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))

        # self.param_dict['L_0'] = 0
        #
        # y_pre = odesolve(self.model, self.tspan[:200], self.param_dict, verbose=verbose)
        #
        # self.tspan = self.tspan[201:]
        #
        # y_pre[-1][0] = 3000
        # # y_pre[-1][9] = 100
        #
        # y0 = [y_pre['__s%d' % i][-1] for i in range(len(self.model.species))]
        #
        # self.y = odesolve(self.model, self.tspan, param_values=self.param_dict, y0=y0, verbose=verbose)

        self.y = odesolve(self.model, self.tspan, param_values=self.param_dict)

        if verbose:
            print("Creating graph")
        self.species_graph()
        self.sp_graph.layout(prog='dot',
                             args="-Gstart=50 -Gesep=1.5 -Gnodesep=0.8 -Gsep=0.5  -Gsplines=true -Gsize=30.75,10.75\! "
                                  "-Gratio=fill -Grankdir=LR -Gdpi=200! -Gordering=in")
        self.sp_graph.add_node('t',
                               label='time',
                               shape='oval',
                               fillcolor='white', style="filled", color="transparent",
                               fontsize="50",
                               margin="0,0",
                               pos="20,20!")

        self.edges_colors(self.y)
        self.nodes_colors(self.y)

        if os.path.exists(fig_path):
            directory = fig_path
        else:
            directory = os.getcwd() + '/visualizations'
            os.makedirs(directory)

        if verbose:
            "Generating images"
        for kx, time in enumerate(self.tspan):
            self.sp_graph.get_node('t').attr['label'] = 'time:' + ' ' + '%d' % time + ' ' + 'sec'
            map(functools.partial(change_edge_colors, graph=self.sp_graph), list(self.colors_time_edges.index),
                list(self.colors_time_edges.iloc[:, kx]))
            map(functools.partial(change_node_colors, graph=self.sp_graph), list(self.colors_time_nodes.index),
                list(self.colors_time_nodes.iloc[:, kx]))
            map(functools.partial(change_edge_size, graph=self.sp_graph), list(self.size_time_edges.index),
                list(self.size_time_edges.iloc[:, kx]))
            self.sp_graph.draw(directory + '/file' + '%03d' % kx + '.png')

    def edges_colors(self, y):
        all_rate_colors = {}
        all_rate_sizes = {}
        rxns_matrix = numpy.zeros((len(self.model.reactions_bidirectional), len(self.tspan)))
        for idx, reac in enumerate(self.model.reactions_bidirectional):
            rate_reac = reac['rate']
            for p in self.param_dict:
                rate_reac = rate_reac.subs(p, self.param_dict[p])
            variables = [atom for atom in rate_reac.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
            react_rate = func(*args)
            rxns_matrix[idx] = react_rate

        max_all_times = [max(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]
        min_all_times = [min(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]

        max_abs_flux = numpy.array([max(i, abs(j)) for i, j in zip(max_all_times, min_all_times)])
        max_flux_diff_area = max_abs_flux / numpy.power(numpy.array([10] * len(max_abs_flux)), magnitude(max_abs_flux))

        global_min_rate = -6
        global_max_rate = magnitude(numpy.abs(max(max_all_times)))
        print(global_min_rate, global_max_rate)

        for rxn in self.model.reactions_bidirectional:
            rate = rxn['rate']
            for p in self.param_dict:
                rate = rate.subs(p, self.param_dict[p])
            variables = [atom for atom in rate.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
            react_rate = func(*args) + self.mach_eps
            react_rate_magnitudes = magnitude(numpy.abs(react_rate))
            react_rate_diff_area = react_rate / numpy.power(numpy.array([10] * len(react_rate)), react_rate_magnitudes)
            rate_colors = f2hex_edges(react_rate_diff_area)
            rate_sizes = range_normalization(react_rate_magnitudes, min_x=global_min_rate, max_x=global_max_rate)
            for rctan in rxn['reactants']:
                for pro in rxn['products']:
                    all_rate_colors[('s' + str(rctan), 's' + str(pro))] = rate_colors
                    all_rate_sizes[('s' + str(rctan), 's' + str(pro))] = rate_sizes
        all_colors = pandas.DataFrame(all_rate_colors).transpose()
        all_sizes = pandas.DataFrame(all_rate_sizes).transpose()

        self.size_time_edges = all_sizes
        self.colors_time_edges = all_colors
        return

    def nodes_colors(self, y):
        all_rate_colors = {}
        initial_conditions_values = [ic[1].value for ic in self.model.initial_conditions]
        # cparp_info = curve_fit(sig_apop, self.tspan, y['cPARP'], p0=[100, 100, 100])[0]
        # midpoint = sig_apop(cparp_info[1], cparp_info[0], cparp_info[1], cparp_info[2])
        max_ic = max(initial_conditions_values)

        for idx in range(len(self.model.species)):
            node_colors = f2hex_nodes(y['__s%d' % idx], vmin=0, vmax=max_ic, midpoint=max_ic/2)
            all_rate_colors['s%d' % idx] = node_colors
        all_nodes_colors = pandas.DataFrame(all_rate_colors).transpose()
        self.colors_time_nodes = all_nodes_colors
        return

    def species_graph(self):

        graph = pygraphviz.AGraph(start=20, directed=True, rankdir="LR", labelloc='t', fontsize=50, label='EARM flux',
                                  bgcolor='gray')
        for idx, cp in enumerate(self.model.species):
            species_node = 's%d' % idx

            graph.add_node(species_node,
                           label=parse_name(self.model.species[idx]),
                           shape="Mrecord",
                           fillcolor="#ccffcc", style="filled", color="transparent",
                           fontsize="35",
                           margin="0.06,0")

        for reaction in self.model.reactions_bidirectional:
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            # attr_reversible = {'arrowsize': 2, 'penwidth': 5}
            attr_reversible = {'dir': 'both', 'arrowtail': 'empty', 'arrowsize': 1, 'penwidth': 5} if reaction[
                'reversible'] else {'arrowsize': 1, 'penwidth': 5}
            for s in reactants:
                for p in products:
                    r_link(graph, s, p, **attr_reversible)
        self.sp_graph = graph
        return self.sp_graph


def run_flux_visualization(model, tspan, fig_path='', parameters=None, verbose=False):
    fv = FluxVisualization(model)
    fv.visualize(fig_path, tspan, parameters, verbose)