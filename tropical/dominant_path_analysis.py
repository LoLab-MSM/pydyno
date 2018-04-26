from pysb.simulator import ScipyOdeSimulator
import numpy as np
import networkx as nx
from pysb.bng import generate_equations
import re
import pandas as pd
from math import log10
import sympy
from itertools import chain

def create_bipartite_graph(model):
    generate_equations(model)
    graph = nx.DiGraph(name=model.name)
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
                       fontsize="12",
                       margin="0.06,0")
    for i, reaction in enumerate(model.reactions_bidirectional):
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
            r_link(graph, s, i, **attr_reversible)
        for s in products:
            r_link(graph, s, i, _flip=True, **attr_reversible)
        for s in modifiers:
            r_link(graph, s, i, arrowhead="odiamond")
    return graph


def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 'r%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


def get_reaction_flux_df(model, tspan, param_values=None):
    if param_values is not None:
        # accept vector of parameter values as an argument
        if len(param_values) != len(model.parameters):
            raise Exception("param_values must be the same length as model.parameters")
        if not isinstance(param_values, np.ndarray):
            param_values = np.array(param_values)
    else:
        # create parameter vector from the values in the model
        param_values = np.array([p.value for p in model.parameters])
    rxns_names = ['r{0}'.format(rxn) for rxn in range(len(model.reactions_bidirectional))]
    rxns_df = pd.DataFrame(columns=tspan, index=rxns_names)
    param_dict = dict((p.name, param_values[i]) for i, p in enumerate(model.parameters))
    sim_result = ScipyOdeSimulator(model, tspan=tspan, param_values=param_dict).run()
    y_df = sim_result.all
    for idx, reac in enumerate(model.reactions_bidirectional):
        rate_reac = reac['rate']
        for p in param_dict:
            rate_reac = rate_reac.subs(p, param_dict[p])
        variables = [atom for atom in rate_reac.atoms(sympy.Symbol)]
        args = [0] * len(variables)  # arguments to put in the lambdify function
        for idx2, va in enumerate(variables):
            if str(va).startswith('__'):
                args[idx2] = y_df[str(va)]
            else:
                args[idx2] = param_dict[va.name]
        func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=np.lib.scimath.sqrt))
        react_rate = func(*args)
        # edge_name = get_edges_name(reac['reactants'], reac['products'])
        # rxns_df.rename(index={int(idx): edge_name}, inplace=True)
        rxns_df.loc['r{0}'.format(idx)] = react_rate
    rxns_df['Total'] = rxns_df.sum(axis=1)
    return rxns_df


def get_reaction_incoming_species(network, r):
    in_edges = network.in_edges(r)
    sp_nodes = [n[0] for n in in_edges]
    return sp_nodes


def get_dominant_paths(model, tspan, param_values, network, target, depth):
    """

    Parameters
    ----------
    model : PySB model
    tspan : vector-like, time span for simulation
    param_values : vector-like, parameters used for the model
    network : nx.DiGraph, bipartite graph to use for obatining the paths
    target : Node label from network, Node from which the pathway starts
    depth : int, The depth of the pathway

    Returns
    -------

    """
    dom_om = 0.5 # Order of magnitude to consider dominancy
    reaction_flux_df = get_reaction_flux_df(model, tspan, param_values)

    path_labels = {0:[]}
    signature = [0] * len(tspan[1:])
    prev_neg_rr = []
    for label, t in enumerate(tspan[1:]):
        # Get reaction rates that are negative
        neg_rr = reaction_flux_df.index[reaction_flux_df[t] < 0].tolist()
        if not neg_rr or prev_neg_rr == neg_rr:
            pass
        else:
            rr_changes = list(set(neg_rr).symmetric_difference(set(prev_neg_rr)))

            # Now we are going to flip the edges whose reaction rate value is negative
            for r_node in rr_changes:
                in_edges = network.in_edges(r_node)
                out_edges = network.out_edges(r_node)
                edges_to_remove = list(in_edges) + list(out_edges)
                network.remove_edges_from(edges_to_remove)
                edges_to_add = [edge[::-1] for edge in edges_to_remove]
                network.add_edges_from(edges_to_add)

            prev_neg_rr = neg_rr

        dom_nodes = {target: [[target]]}
        all_rdom_noodes = []
        t_paths = [0] * depth
        for d in range(depth):
            all_dom_nodes = {}
            for node_doms in dom_nodes.values():
                flat_node_doms = [item for items in node_doms for item in items]
                for node in flat_node_doms:
                    in_edges = network.in_edges(node)
                    # for edge in in_edges: print (node, edge)
                    fluxes_in = {edge: log10(abs(reaction_flux_df.loc[edge[0], t])) for edge in in_edges}
                    max_val = np.amax(fluxes_in.values())
                    dom_r_nodes = [n[0] for n, i in fluxes_in.items() if i > (max_val - dom_om) and max_val > -5]
                    dom_sp_nodes = [get_reaction_incoming_species(network, reaction_nodes) for reaction_nodes in dom_r_nodes]
                        # [n[0] for reaction_nodes in dom_r_nodes for n in network.in_edges(reaction_nodes)]
                    # Get the species nodes from the reaction nodes to keep back tracking the pathway
                    all_dom_nodes[node] = dom_sp_nodes
                    all_rdom_noodes.append(dom_r_nodes)
                dom_nodes = all_dom_nodes

            t_paths[d] = dom_nodes
        check_paths = [all_rdom_noodes == path for path in path_labels.values()]
        if not any(check_paths):
            path_labels[label] = all_rdom_noodes
            signature[label] = label
        else:
            signature[label] = np.array(path_labels.keys())[check_paths][0]

    print(path_labels.keys())
    print(signature)



        # Add the dominants at each node
        # Add a depth argument to define how many steps should be taken to obtain the path
