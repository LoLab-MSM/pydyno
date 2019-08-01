import numpy as np
import networkx as nx
from pysb.bng import generate_equations
import re
import pandas as pd
from math import log10
import sympy
import pydyno.util as hf
from pydyno.sequences import Sequences
from collections import OrderedDict
from anytree import Node, findall
from anytree.exporter import DictExporter
from collections import ChainMap
import time

try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    Pool = None

try:
    import h5py
except ImportError:
    h5py = None

# Types of analysis that have been implemented. For `production` the analysis consists in
# finding the dominant path that is producing a species defined by the target parameter.
# For consumption the analysis consists in finding the dominant path that is consuming a
# species defined by the target.
TYPE_ANALYSIS_DICT = {'production': ['in_edges', 0],
                      'consumption': ['out_edges', 1]}


class DomPath(object):
    """
    Class to discretize the simulated trajectory of a model species
    Parameters
    ----------
    model: PySB model
        Model to analyze
    simulations: PySB SimulationResult object or str
        simulations used to perform the analysis. If str it should be the
        path to a simulation result in hdf5 format
    type_analysis: str
        Type of analysis to perform. It can be `production` or `consumption`
    dom_om: float
        Order of magnitude to consider dominancy
    target: str
        Species target. It has to be in a format `s1` where the number
        represents the species index
    depth: int
        Depth of the traceback starting from target
    """

    def __init__(self, model, simulations, type_analysis, dom_om, target, depth):
        self._model = model
        self._trajectories, self._parameters, self._nsims, self._tspan = hf.get_simulations(simulations)
        self._par_name_idx = {j.name: i for i, j in enumerate(self.model.parameters)}
        self._type_analysis = type_analysis
        self._dom_om = dom_om
        self._target = target
        self._depth = depth
        generate_equations(self.model)

    @property
    def model(self):
        return self._model

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def parameters(self):
        return self._parameters

    @property
    def nsims(self):
        return self._nsims

    @property
    def tspan(self):
        return self._tspan

    @property
    def par_name_idx(self):
        return self._par_name_idx

    @property
    def type_analysis(self):
        return self._type_analysis

    @property
    def dom_om(self):
        return self._dom_om

    @property
    def target(self):
        return self._target

    @property
    def depth(self):
        return self._depth

    def create_bipartite_graph(self):
        """
        Creates bipartite graph with species and reaction nodes of the pysb model
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
    def species_connected_to_node(network, r, type_edge, idx_r):
        """
        Obtains the species connected to a node.
        Parameters
        ----------
        network: nx.Digraph
            Networkx directed network
        r: str
            Node name
        type_edge: str
            it can be `in_edges` or `out_edges`
        idx_r: int
            Index of the reaction node in the edge returned by the in_edges or out_edges function

        Returns
        -------
        If `type_edge` == in_edges and `r` == 0 this function returns the species that are being consumed
        by the reaction node r.
        If `type_edge` == out_edges and `r` == 1 this function returns the species that are being produced
        by the reaction node r.
        """
        in_edges = getattr(network, type_edge)(r)
        sp_nodes = [n[idx_r] for n in in_edges]
        # Sort the incoming nodes to get the same results in each simulation
        return _natural_sort(sp_nodes)

    def dominant_paths(self, trajectories, parameters):
        """
        Traceback a dominant path from a user defined target
        Parameters
        ----------
        trajectories : PySB SimulationResult object
            Simulation result to use to obtain the dominant_paths
        parameters : vector-like
            Parameter set used for the trajectories simulation

        Returns
        -------

        """
        network = self.create_bipartite_graph()
        reaction_flux_df = self.get_reaction_flux_df(trajectories, parameters)

        path_rlabels = {}
        # path_sp_labels = {}
        signature = [0] * len(self.tspan[1:])
        prev_neg_rr = []
        # First we iterate over time points
        for t_idx, t in enumerate(self.tspan[1:]):
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
                all_dom_nodes = OrderedDict()
                dom_r3 = []
                for node_doms in dom_nodes.values():
                    # Looping over dominant nodes i.e [[s1, s2], [s4, s8]]
                    dom_r2 = []
                    for nodes in node_doms:
                        # Looping over one of the dominants i.e. [s1, s2]
                        dom_r1 = []
                        for node in nodes:
                            # node = s1
                            # Obtaining the edges connected to the species node. It would be
                            # in_edges if type_analysis is `production` and out_edges if
                            # type_analysis is `consumption`
                            type_edge, idx_r = TYPE_ANALYSIS_DICT[self.type_analysis]
                            connected_edges = getattr(network, type_edge)(node)
                            if not connected_edges:
                                continue
                            # Obtaining the reaction rate value of the rate node that connects to
                            # the species node
                            fluxes_in = {edge: log10(abs(reaction_flux_df.loc[edge[idx_r], t]))
                                         for edge in connected_edges if reaction_flux_df.loc[edge[idx_r], t] != 0}

                            if not fluxes_in:
                                continue

                            max_val = np.amax(list(fluxes_in.values()))
                            # Obtaining dominant species and reactions nodes
                            dom_r_nodes = [n[idx_r] for n, i in fluxes_in.items() if i > (max_val - self.dom_om)]
                            # Sort the dominant r nodes to get the same results in each simulation
                            dom_r_nodes = _natural_sort(dom_r_nodes)
                            dom_sp_nodes = [self.species_connected_to_node(network, reaction_nodes, type_edge, idx_r)
                                            for reaction_nodes in dom_r_nodes]
                            # Get the species nodes from the reaction nodes to keep back tracking the pathway
                            all_dom_nodes[node] = dom_sp_nodes
                            # all_rdom_nodes.append(dom_r_nodes)
                            dom_r1.append(sorted(dom_r_nodes))

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
                            Node(sp, parent=m, order=idx + 1)
                        # sp_paths.append((sp, idx+1))
            # sp_paths.insert(0, (self.target, 0))

            # if not dominant path define a label 1
            if not list(_find_numbers(all_rdom_noodes_str)):
                rdom_label = -1
            else:
                rdom_label = _list_to_int(_find_numbers(all_rdom_noodes_str))
            path_rlabels[rdom_label] = DictExporter().export(root)
            signature[t_idx] = rdom_label
            # path_sp_labels[rdom_label] = t_paths
        return signature, path_rlabels

    def get_path_signatures(self, num_processors=1, sample_simulations=None, verbose=False):
        """

        Parameters
        ----------
        num_processors : int
            Number of cores to use in the function
        sample_simulations : int
            Number of simulations to use for the analysis
        verbose : bool
            Print the number of processes left

        Returns
        -------

        """
        if sample_simulations:
            if isinstance(sample_simulations, int):
                trajectories = self.trajectories[:sample_simulations]
                parameters = self.parameters[:sample_simulations]
                nsims = sample_simulations
            elif isinstance(sample_simulations, list):
                trajectories = self.trajectories[sample_simulations]
                parameters = self.parameters[sample_simulations]
                nsims = len(sample_simulations)
            else:
                raise TypeError('Sample method not supported')
        else:
            trajectories = self.trajectories
            parameters = self.parameters
            nsims = self.nsims

        if num_processors == 1 or nsims == 1:
            if nsims == 1:
                # This assumes that the pysb simulation used the squeeze_output
                # which is the default
                if sample_simulations:
                    trajectories = trajectories[0]
                    parameters = parameters[0]
                else:
                    parameters = parameters[0]
                signatures, labels = self.dominant_paths(trajectories, parameters)
                signatures_df, new_paths = _reencode_signatures_paths(signatures, labels, self.tspan)
                # signatures_labels = {'signatures': signatures, 'labels': labels}
                return Sequences(signatures_df, self.target), new_paths
            else:
                all_signatures = [0] * nsims
                all_labels = [0] * nsims
                for idx in range(nsims):
                    all_signatures[idx], all_labels[idx] = self.dominant_paths(trajectories[idx], parameters[idx])
                signatures_df, new_paths = _reencode_signatures_paths(all_signatures, all_labels, self.tspan)
                return Sequences(signatures_df, self.target), new_paths
        else:
            if Pool is None:
                raise Exception('Please install the pathos package for this feature')
            # if self.nsims == 1:
            #     self.trajectories = [self.trajectories]
            #     self.parameters = [self.parameters]

            p = Pool(num_processors)
            res = p.amap(self.dominant_paths, trajectories, parameters)
            if verbose:
                while not res.ready():
                    print('We\'re not done yet, {0} tasks to go!'.format(res._number_left))
                    time.sleep(60)
            signatures_labels = res.get()
            signatures = [0] * len(signatures_labels)
            labels = [0] * len(signatures_labels)
            for idx, sl in enumerate(signatures_labels):
                signatures[idx] = sl[0]
                labels[idx] = sl[1]
            signatures_df, new_paths = _reencode_signatures_paths(signatures, labels, self.tspan)
            # signatures_labels = {'signatures': signatures, 'labels': all_labels}
            return Sequences(signatures_df, self.target), new_paths


def _reencode_signatures_paths(signatures, labels, tspan):
    if isinstance(labels, list):
        all_labels = dict(ChainMap(*labels))
        signatures = np.array(signatures)
    else:
        all_labels = labels
        signatures = np.array(signatures, ndmin=2)
    unique_signatures = np.unique(signatures)
    new_labels = {va: i for i, va in enumerate(unique_signatures)}
    new_paths = {new_labels[key]: value for key, value in all_labels.items()}
    del all_labels
    signatures_df = _signatures_to_dataframe(signatures, tspan)
    signatures_df = signatures_df.applymap(lambda x: new_labels[x])
    return signatures_df, new_paths


def _signatures_to_dataframe(signatures, tspan):
    def time_values(t):
        # We add 1 because the first time point is ignored as there could
        # be equilibration issues
        return tspan[t + 1]

    s = pd.DataFrame(signatures)
    s.rename(time_values, axis='columns', inplace=True)
    return s


def _find_numbers(dom_r_str):
    n = map(int, re.findall(r'\d+', dom_r_str))
    return n


def _list_to_int(nums):
    return int(''.join(map(str, nums)))


def _natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
