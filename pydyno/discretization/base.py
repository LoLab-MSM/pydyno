from abc import ABC, abstractmethod
import re
import json
import hashlib
from math import log10
from collections import OrderedDict, ChainMap
import numpy as np
import pandas as pd
from anytree.exporter import DictExporter

# Types of analysis that have been implemented. For `production` the analysis consists in
# finding the dominant path that is producing a species defined by the target parameter.
# For consumption the analysis consists in finding the dominant path that is consuming a
# species defined by the target.
TYPE_ANALYSIS_DICT = {'production': ['in_edges', 0],
                      'consumption': ['out_edges', 1]}


class DomPath(ABC):
    """
    An abstract base class to discretize simulated trajectories of a model species
    Parameters
    ----------
    model: PySB model
        Model to analyze
    """
    @abstractmethod
    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    @abstractmethod
    def create_bipartite_graph(self):
        """
        Creates bipartite graph with species and reaction nodes of the pysb model

        Returns
        -------
        nx.DiGraph
        """

        return None

    @abstractmethod
    def get_path_signatures(self, target, type_analysis, depth, dom_om,
                            num_processors=1, sample_simulations=None):
        """

        Parameters
        ----------
        target: str
            Species target. It has to be in a format `s1` where the number
            represents the species index
        type_analysis: str
            Type of analysis to perform. It can be `production` or `consumption`
        depth: int
            Depth of the traceback starting from target
        dom_om: float
            Order of magnitude to consider dominancy
        num_processors : int
            Number of cores to use in the function
        sample_simulations : int
            Number of simulations to use for the analysis
        verbose : bool
            Print the number of processes left

        Returns
        -------
        pydyno.SeqAnalysis
            Sequences of the discretized signatures
        """

        return None


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


def _natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def _dominant_paths(reaction_flux_df, network, tspan,
                    target, type_analysis, depth, dom_om):
    """
    Traceback a dominant path from a user defined target
    Parameters
    ----------
    network: nx.DiGraph
        Network obtained from model
    reaction_flux_df: pd.DataFrame
        Pandas dataframe with the reaction rates obtained from a simulation of the model
    target: str
        Species target. It has to be in a format `s1` where the number
        represents the species index
    type_analysis: str
        Type of analysis to perform. It can be `production` or `consumption`
    depth: int
        Depth of the traceback starting from target
    dom_om: float
        Order of magnitude to consider dominancy

    Returns
    -------

    """
    type_edge, node_from_edge = TYPE_ANALYSIS_DICT[type_analysis]
    path_rlabels = {}
    # path_sp_labels = {}
    signature = [0] * len(tspan[1:])
    prev_neg_rr = []
    # First we iterate over time points
    for t_idx, t in enumerate(tspan[1:]):
        # Get reaction rates that are negative to see which edges have to be reversed
        neg_rr = reaction_flux_df.index[reaction_flux_df[t] < 0].tolist()
        if not neg_rr or prev_neg_rr == neg_rr:
            pass
        else:
            _flip_network_edges(network, neg_rr, prev_neg_rr)

            prev_neg_rr = neg_rr

        # Starting with the target
        dom_nodes = {target: [[target]]}
        t_paths = [0] * depth
        # Iterate over the depth
        for d in range(depth):
            all_dom_nodes = OrderedDict()
            for node_doms in dom_nodes.values():
                # Looping over dominant nodes e.g [[s1, s2], [s4, s8]]
                for nodes in node_doms:
                    # Looping over one of the dominants e.g. [s1, s2]
                    for node in nodes:
                        dom_r_nodes = _dominant_connected_reactions(network, node, t,
                                                                    reaction_flux_df, dom_om, type_edge, node_from_edge)
                        if dom_r_nodes is None:
                            continue

                        dom_sp_nodes = []
                        for reaction_nodes in dom_r_nodes:
                            sp_nodes = _species_connected_to_node(network, reaction_nodes, type_edge, node_from_edge)
                            if sp_nodes and sp_nodes not in dom_sp_nodes:
                                dom_sp_nodes.append(sp_nodes)

                        # Get the species nodes from the reaction nodes to keep backtracking the pathway
                        if dom_sp_nodes:
                            all_dom_nodes[node] = dom_sp_nodes
                        # all_rdom_nodes.append(dom_r_nodes)

                    dom_nodes = all_dom_nodes
            t_paths[d] = dom_nodes

        dom_path_label = hashlib.sha1(json.dumps(t_paths, sort_keys=True).encode()).hexdigest()
        # This is to create a tree with the information of the dominant species
        path_rlabels[dom_path_label] = t_paths
        signature[t_idx] = dom_path_label
        # path_sp_labels[rdom_label] = t_paths
    return signature, path_rlabels


def _dominant_connected_reactions(network, species_node, t, reaction_flux_df, dom_om, type_edge, node_from_edge):
    """
    Obtains the dominant reaction nodes connected to species_node based on the reaction flux and the
    dominant threshold

    Parameters
    ----------
    network : nx.DiGraph
        Network used
    species_node : str
        Node label
    t : float
        Time point
    reaction_flux_df : pd.DataFrame
        Pandas dataframe that contains the reaction rates from a simulation
    dom_om : float
        Order of magnitude to consider dominancy
    type_edge : str
        Type of edges connected to node. It can be `in_edges` or `out_edges`
    node_from_edge : idx
        Node to obtain from an edge. It can be the source node or the target node

    Returns
    -------
    list
        List of dominant reaction nodes
    """
    # node = s1
    # Obtaining the edges connected to the species node. It would be
    # in_edges if type_analysis is `production` and out_edges if
    # type_analysis is `consumption`

    connected_edges = getattr(network, type_edge)(species_node)
    if not connected_edges:
        return None
    # Obtaining the reaction rate value of the rate node that connects to
    # the species node
    fluxes_in = {edge: log10(abs(reaction_flux_df.loc[edge[node_from_edge], t]))
                 for edge in connected_edges if reaction_flux_df.loc[edge[node_from_edge], t] != 0}
    if not fluxes_in:
        return None

    max_val = np.amax(list(fluxes_in.values()))
    # Obtaining dominant species and reactions nodes
    dom_r_nodes = [n[node_from_edge] for n, i in fluxes_in.items() if i > (max_val - dom_om)]
    # Sort the dominant r nodes to get the same results in each simulation
    dom_r_nodes = _natural_sort(dom_r_nodes)
    return dom_r_nodes


def _flip_network_edges(network, neg_rr, prev_neg_rr):
    """
    Flip network edges that represent bidirectional reaction rates. When a network is created edges's
    direction goes from reactant to product species. For bidirectional reactions the net flux sometimes
    can go from product (bound species) to reactant species. Hence, to account for this change in flux
    directionality, this function flip the required edges

    Parameters
    ----------
    network : nx.DiGraph
        Bipartite network obtained from a model
    neg_rr : list
        List of reaction labels that have negative reaction rates at the current time point
    prev_neg_rr : list
        List of reaction labels that had negative reaction rates at the previous time point

    Returns
    -------

    """
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


def _species_connected_to_node(network, r, type_edge, idx_r):
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

