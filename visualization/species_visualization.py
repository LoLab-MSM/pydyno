from earm.lopez_embedded import model
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


def f2hex_edges(fx):
    norm = colors.Normalize(vmin=0, vmax=1)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('coolwarm'))
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
    n = graph.get_node(node)
    n.attr['fillcolor'] = color
    return


def change_edge_colors(edge, color, graph):
    n = graph.get_edge(*edge)
    n.attr['color'] = color
    return


class FluxVisualization:
    def __init__(self, model):
        self.model = model
        self.tspan = None
        self.y = None
        self.parameters = None
        self.sp_graph = None
        self.colors_time_edges = None

    def visualize(self, fig_path='', tspan=None, parameters=None, verbose=False):
        if verbose:
            print "Solving Simulation"

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
            print "Creating graph"
        self.species_graph()
        self.sp_graph.layout(prog='dot',
                             args="-Gstart=50 -Gesep=1  -Gsplines=true -Gsize=30.75,10.75\! "
                                  "-Gratio=fill -Grankdir=LR -Gdpi=100! -Gordering=in")
        self.sp_graph.add_node('t',
                               label='time',
                               shape='oval',
                               fillcolor='white', style="filled", color="transparent",
                               fontsize="50",
                               margin="0.06,0",
                               pos="20,20!",
                               pin='true')

        self.edges_colors(self.y)

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
            self.sp_graph.draw(directory + '/file' + '%03d' % kx + '.png')

    def edges_colors(self, y):
        all_rate_colors = {}
        for idx, rxn in enumerate(self.model.reactions):
            rate = rxn['rate']
            for p in self.parameters:
                rate = rate.subs(p, self.parameters[p])
            variables = [atom for atom in rate.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
            react_rate = func(*args)
            for rctan in rxn['reactants']:
                rate_total = 0
                for r_rxn in self.model.reactions:
                    if rctan in r_rxn['reactants']:
                        rate_total += r_rxn['rate']
                for p in self.parameters:
                    rate_total = rate_total.subs(p, self.parameters[p])
                variables = [atom for atom in rate_total.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
                func = sympy.lambdify(variables, rate_total, modules=dict(sqrt=numpy.lib.scimath.sqrt))
                args = [y[str(l)] for l in variables]   # arguments to put in the lambdify function
                total_react_rate = func(*args)

                rate_data = react_rate / total_react_rate

                rate_colors = map(f2hex_edges, rate_data)
                for pro in rxn['products']:
                    all_rate_colors[('s' + str(rctan), 's' + str(pro))] = rate_colors
        all_colors = pandas.DataFrame(all_rate_colors).transpose()
        self.colors_time_edges = all_colors
        return self.colors_time_edges

    def species_graph(self):

        graph = pygraphviz.AGraph(directed=True, rankdir="LR", labelloc='t', fontsize=50, label='EARM flux')
        for idx, cp in enumerate(self.model.species):
            species_node = 's%d' % idx
            label = re.sub(r'% ', r'%\\l', str(cp))
            label += '\\l'

            rtn_product = False
            rtn = False
            prdt = False

            all_reactants = []
            for rctn in self.model.reactions:
                if rctn['reverse'][0] is False:
                    all_reactants.append(rctn['reactants'])
            all_reactants_flat = list(set([item for sublist in all_reactants for item in sublist]))

            all_products = []
            for rctn in self.model.reactions:
                if rctn['reverse'][0] is False:
                    all_products.append(rctn['products'])
            all_products_flat = list(set([item for sublist in all_products for item in sublist]))

            if idx in all_reactants_flat:
                rtn = True
            if idx in all_products_flat:
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

        for reaction in model.reactions:
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            attr_reversible = {'arrowsize': 2, 'penwidth': 5}
            for s in reactants:
                for p in products:
                    r_link(graph, s, p, **attr_reversible)
        self.sp_graph = graph
        return self.sp_graph


def run_flux_visualization(model, tspan, fig_path='', parameters=None, verbose=False):
    fv = FluxVisualization(model)
    fv.visualize(fig_path, tspan, parameters, verbose)
