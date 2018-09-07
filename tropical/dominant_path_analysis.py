import numpy as np
import networkx as nx
from pysb.bng import generate_equations
import re
import pandas as pd
from math import log10
import sympy
import tropical.util as hf
from collections import defaultdict
from anytree import Node, RenderTree, findall
from anytree.exporter import DotExporter, DictExporter
try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    Pool = None

try:
    import h5py
except ImportError:
    h5py = None


class DomPath(object):
    """

    Parameters
    ----------
    model: PySB model
        Model to analyze
    tspan: vector-like
        Time of the simulation
    dom_om: float
        Order of magnitude to consider dominancy
    target
    depth
    """
    def __init__(self, model, tspan, dom_om, target, depth):
        self.model = model
        self.tspan = tspan
        self.dom_om = dom_om
        self.target = target
        self.depth = depth
        generate_equations(self.model)

    def create_bipartite_graph(self):
        """
        Creates bipartite graph of the pysb model
        Returns
        -------

        """
        graph = nx.DiGraph(name=self.model.name)
        ic_species = [cp for cp, parameter in self.model.initial_conditions]
        for i, cp in enumerate(self.model.species):
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
                           fontsize="12",
                           margin="0.06,0")
        for i, reaction in enumerate(self.model.reactions_bidirectional):
            reaction_node = 'r%d' % i
            graph.add_node(reaction_node,
                           label=reaction_node,
                           shape="circle",
                           fillcolor="lightgray", style="filled", color="transparent",
                           fontsize="12",
                           width=".3", height=".3", margin="0.06,0")
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            modifiers = reactants & products
            reactants = reactants - modifiers
            products = products - modifiers
            attr_reversible = {'dir': 'both', 'arrowtail': 'empty'} if reaction['reversible'] else {}
            for s in reactants:
                self.r_link(graph, s, i, **attr_reversible)
            for s in products:
                self.r_link(graph, s, i, _flip=True, **attr_reversible)
            for s in modifiers:
                self.r_link(graph, s, i, arrowhead="odiamond")
        return graph

    @staticmethod
    def r_link(graph, s, r, **attrs):
        nodes = ('s%d' % s, 'r%d' % r)
        if attrs.get('_flip'):
            del attrs['_flip']
            nodes = reversed(nodes)
        attrs.setdefault('arrowhead', 'normal')
        graph.add_edge(*nodes, **attrs)

    def get_reaction_flux_df(self, trajectories, parameters):
        """
        Creates a data frame with the reaction rates values at each time point
        Parameters
        ----------
        trajectories: vector-like
            Species trajectories used to calculate the reaction rates
        parameters: vector-like
            Model parameters. Parameters must have the same order as the model

        Returns
        -------

        """
        param_values = parameters
        rxns_names = ['r{0}'.format(rxn) for rxn in range(len(self.model.reactions_bidirectional))]
        rxns_df = pd.DataFrame(columns=self.tspan, index=rxns_names)
        param_dict = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))

        for idx, reac in enumerate(self.model.reactions_bidirectional):
            rate_reac = reac['rate']
            # Getting species and parameters from the reaction rate
            variables = [atom for atom in rate_reac.atoms(sympy.Symbol)]
            args = [0] * len(variables)  # arguments to put in the lambdify function
            for idx2, va in enumerate(variables):
                # Getting species index
                if str(va).startswith('__'):
                    sp_idx = int(''.join(filter(str.isdigit, str(va))))
                    args[idx2] = trajectories[:, sp_idx]
                else:
                    args[idx2] = param_dict[va.name]
            func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=np.lib.scimath.sqrt))
            react_rate = func(*args)
            rxns_df.loc['r{0}'.format(idx)] = react_rate
        rxns_df['Total'] = rxns_df.sum(axis=1)
        return rxns_df

    @staticmethod
    def get_reaction_incoming_species(network, r):
        """
        Gets all the edges that are coming from species nodes and are going to a reaction node r
        Parameters
        ----------
        network: nx.Digraph
            Networkx directed network
        r: str
            Node name

        Returns
        -------
        Species that are involved in reaction r
        """
        in_edges = network.in_edges(r)
        sp_nodes = [n[0] for n in in_edges]
        return sp_nodes

    def get_dominant_paths(self, trajectories, parameters):
        """
        Traceback a dominant path from a defined target
        Parameters
        ----------
        target : str
            Node label from network, Node from which the pathway starts
        depth : int
            The depth of the pathway

        Returns
        -------

        """
        network = self.create_bipartite_graph()
        reaction_flux_df = self.get_reaction_flux_df(trajectories, parameters)

        path_rlabels = {}
        path_sp_labels = {}
        signature = [0] * len(self.tspan[1:])
        prev_neg_rr = []
        # First we iterate over time points
        for label, t in enumerate(self.tspan[1:]):
            # Get reaction rates that are negative to see which edges have to be reversed
            neg_rr = reaction_flux_df.index[reaction_flux_df[t] < 0].tolist()
            if not neg_rr or prev_neg_rr == neg_rr:
                pass
            else:
                # Compare the negative indices from the current iteration to the previous one
                # and flip the edges of the ones that have changed
                rr_changes = list(set(neg_rr).symmetric_difference(set(prev_neg_rr)))

                for r_node in rr_changes:
                    # remove in and out edges of the node to add them in the reversed direction
                    in_edges = network.in_edges(r_node)
                    out_edges = network.out_edges(r_node)
                    edges_to_remove = list(in_edges) + list(out_edges)
                    network.remove_edges_from(edges_to_remove)
                    edges_to_add = [edge[::-1] for edge in edges_to_remove]
                    network.add_edges_from(edges_to_add)

                prev_neg_rr = neg_rr

            dom_nodes = {self.target: [[self.target]]}
            all_rdom_nodes = [0] * self.depth
            t_paths = [0] * self.depth
            # Iterate over the depth
            for d in range(self.depth):
                all_dom_nodes = {}
                dom_r3 = []
                for node_doms in dom_nodes.values():
                    # Looping over dominant nodes i.e [[s1, s2], [s4, s8]]
                    dom_r2 = []
                    for nodes in node_doms:
                        # Looping over one of the dominants i.e. [s1, s2]
                        dom_r1 = []
                        for node in nodes:
                            # node = s1
                            # Obtaining the incoming edges of the species node
                            in_edges = network.in_edges(node)
                            if not in_edges:
                                continue
                            # Obtaining the reaction rate value of the rate node that connects to
                            # the species node
                            fluxes_in = {edge: log10(reaction_flux_df.loc[edge[0], t])
                                         for edge in in_edges if reaction_flux_df.loc[edge[0], t] > 0}
                            if not fluxes_in:
                                continue

                            max_val = np.amax(list(fluxes_in.values()))
                            # Obtaining dominant species and reactions nodes
                            dom_r_nodes = [n[0] for n, i in fluxes_in.items() if i > (max_val - self.dom_om)]
                            dom_sp_nodes = [self.get_reaction_incoming_species(network, reaction_nodes)
                                            for reaction_nodes in dom_r_nodes]
                            # Get the species nodes from the reaction nodes to keep back tracking the pathway
                            all_dom_nodes[node] = dom_sp_nodes
                            # all_rdom_nodes.append(dom_r_nodes)
                            dom_r1.append(dom_r_nodes)

                        dom_nodes = all_dom_nodes
                        dom_r2.append(dom_r1)
                    dom_r3.append(dom_r2)

                all_rdom_nodes[d] = dom_r3
                t_paths[d] = dom_nodes

            all_rdom_noodes_str = str(all_rdom_nodes)
            # sp_paths = []
            # This is to create a tree with the information of the dominant species
            root = Node(self.target, order=0)
            for idx, ds in enumerate(t_paths):
                for pa, v in ds.items():
                    sps = np.concatenate(v)
                    for sp in sps:
                        p = findall(root, filter_=lambda n: n.name == pa and n.order == idx)
                        for m in p:
                            Node(sp, parent=m, order=idx+1)
                        # sp_paths.append((sp, idx+1))
            # sp_paths.insert(0, (self.target, 0))

            rdom_label = list_to_int(find_numbers(all_rdom_noodes_str))
            path_rlabels[rdom_label] = DictExporter().export(root)
            signature[label] = rdom_label
            path_sp_labels[rdom_label] = t_paths
        # TODO figure a way to return this.
        # print(path_sp_labels)
        #     if t == 40.0:
        #         print(t_paths)
        #         print(RenderTree(root))
        #         DotExporter(root, graph='strict digraph', options=["rankdir=RL;"], nodenamefunc=self.nodenamefunc,
        #                     edgeattrfunc=edgeattrfunc).to_picture("s27_40reverse.pdf")
        return signature, path_rlabels

    def nodenamefunc(self, node):
        node_idx = list(find_numbers(node.name))[0]
        node_sp = self.model.species[node_idx]
        node_name = hf.parse_name(node_sp)
        return node_name


def edgeattrfunc(node, child):
    return 'dir="back"'


def run_dompath_single(model, simulations, dom_om, target, depth):
    trajectories, parameters, nsims, tspan = hf.get_simulations(simulations)
    dompath = DomPath(model, tspan, dom_om, target, depth)
    if nsims == 1:
        signatures = dompath.get_dominant_paths(trajectories, parameters[0])
        return signatures
    elif nsims > 1:
        all_signatures = [0] * nsims
        all_labels = [0] * nsims
        for i in range(nsims):
            all_signatures[i], all_labels[i] = dompath.get_dominant_paths(trajectories[i], parameters[i])
        all_labels = merge_dicts(all_labels)
        all_signatures = np.array(all_signatures)
        signatures_labels = {'signatures': all_signatures, 'labels': all_labels}
        return signatures_labels


def run_dompath_multi(model, simulations, dom_om, target, depth, cpu_cores=1):
    if Pool is None:
        raise Exception('Plese install the pathos package for this feature')
    trajectories, parameters, nsims, tspan = hf.get_simulations(simulations)
    dompath = DomPath(model, tspan, dom_om, target, depth)
    if nsims == 1:
        trajectories = [trajectories]
    p = Pool(cpu_cores)
    res = p.amap(dompath.get_dominant_paths, trajectories, parameters)
    signatures_labels = res.get()
    signatures = [0] * len(signatures_labels)
    labels = [0] * len(signatures_labels)
    for idx, sl in enumerate(signatures_labels):
        signatures[idx] = sl[0]
        labels[idx] = sl[1]
    all_labels = merge_dicts(labels)
    signatures = np.array(signatures)
    signatures_labels = {'signatures': signatures, 'labels': all_labels}
    return signatures_labels


def find_numbers(dom_r_str):
    n = map(int, re.findall('\d+', dom_r_str))
    return n


def list_to_int(nums):
        return int(''.join(map(str, nums)))


def merge_dicts(dicts):
    super_dict = defaultdict(set)
    for d in dicts:
        for k, v in d.items():  # use d.iteritems() in python 2
            super_dict[k].add(v)
    return super_dict
