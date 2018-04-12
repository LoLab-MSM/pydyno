import itertools
import pandas as pd
import sympy
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from pysb.pattern import SpeciesPatternMatcher
from pysb.bng import generate_equations
from pysb import ANY
import networkx as nx
import re


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


def get_graph_pathways(graph, source, target):
    return nx.all_simple_paths(graph, source, target)


def remove_duplicated_paths(paths):
    # To remove the duplicated paths we take only the reactions from the path and take
    # the paths of reactions that are note duplicated
    paths = np.array(list(paths))
    reaction_paths = np.array([get_r_nodes(path) for path in paths])
    unique_paths_idxs = np.unique(reaction_paths, return_index=True)
    return paths[unique_paths_idxs]


def get_edges_name(reactants, products):
    edge_str = ''
    for r in reactants:
        for p in products:
            edge = (r, p)
            edge_str += str(edge)
    return edge_str


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


def get_sp_nodes(path):
    sp_nodes = [node for node in path if node.startswith('s')]
    return sp_nodes


def get_r_nodes(path):
    r_nodes = [node for node in path if node.startswith('r')]
    return r_nodes

def get_paths_flux(model, tspan, param_values, network, source, target):

    reaction_flux_df = get_reaction_flux_df(model, tspan, param_values)
    paths_flux = {}

    bla = []
    for t in tspan:
        # Get reaction rates that are negative
        neg_rr = reaction_flux_df.index[reaction_flux_df[t] < 0].tolist()
        if not neg_rr or bla == neg_rr:
            continue
        else:
            rr_changes = list(set(neg_rr).symmetric_difference(set(bla)))

            # Now we are going to flip the edges whose reaction rate value is negative
            for edge in rr_changes:
                in_edges = network.in_edges(edge)
                out_edges = network.out_edges(edge)
                edges_to_remove = list(in_edges) + list(out_edges)
                network.remove_edges_from(edges_to_remove)
                edges_to_add = [edge[::-1] for edge in edges_to_remove]
                network.add_edges_from(edges_to_add)

            bla = neg_rr
        updated_paths = list(nx.all_simple_paths(network, source, target))
        inst_flux_df = pd.DataFrame(columns=tspan, index=range(len(updated_paths)))
        for path_idx, path in enumerate(updated_paths):
            sp_nodes = get_sp_nodes(path)
            r_nodes = get_r_nodes(path)
            sp_nodes.reverse()
            r_nodes.reverse()
            print(r_nodes)

            path_percentage = 1
            for sp_node, r_node in zip(sp_nodes[:-1], r_nodes):
                in_edges = network.in_edges(sp_node)
                # print (path_idx, r_node, r_node_edges_correction)

                total_flux_in_l = np.array([reaction_flux_df.loc[edge[0], t] for edge in in_edges])
                total_flux_in = np.sum(np.abs(total_flux_in_l))
                # total_flux_in = np.sum(total_flux_in_l[np.where(total_flux_in_l > 0)])
                # print (path_idx, total_flux_in_l)
                # print (path_idx, reaction_flux_df.loc[r_node, t])
                flux_in = np.sum(np.abs(reaction_flux_df.loc[r_node, t]))

                if total_flux_in < np.finfo(float).eps:
                    percentage_in = 0
                else:
                    percentage_in = flux_in / total_flux_in

                # if path_idx == 2 or path_idx==3:
                #     print (r_node, r_nodes, in_edges)
                #     print ('total', total_flux_in_l)
                #     print('flux in',flux_in)
                #     print ('perc', percentage_in)

                node_percentage = percentage_in

                path_percentage *= node_percentage

            inst_flux_df.loc[path_idx][t] = path_percentage
        print (t, inst_flux_df[t].values.sum())

    # for path_idx, path in enumerate(paths):
    #
    #     sp_nodes = get_sp_nodes(path)
    #     r_nodes = get_r_nodes(path)
    #
    #     for t in tspan:
    #
    #         path_percentage = 1
    #         for sp_node, r_node in zip(sp_nodes[:-1], r_nodes):
    #             in_edges = network.in_edges(sp_node)
    #
    #             total_flux_in_l = np.array([reaction_flux_df.loc[edge[0], tspan[5]] for edge in in_edges])
    #             total_flux_in = np.sum(total_flux_in_l[np.where(total_flux_in_l > 0)])
    #
    #             flux_in = np.sum(reaction_flux_df.loc[r_node, t])
    #
    #             if total_flux_in < np.finfo(float).eps:
    #                 percentage_in = 0
    #             else:
    #                 percentage_in = flux_in / total_flux_in
    #
    #             node_percentage = percentage_in
    #
    #             path_percentage *= node_percentage
    #
    #         inst_flux_df.loc[path_idx][t] = path_percentage

        # path_edges = zip(path[:-1],path[1:])
        # if total_flux:
        #     path_flux = 0
        #     for edge in path_edges:
        #         flux = np.sum(reaction_flux_df.filter(like=str(edge), axis=0)['Total'].values[0])
        #         path_flux += flux
        #     paths_flux[path_idx] = path_flux
        #
        # else:
        #     cum_path_flux = 0
        #     for t in tspan:
        #         path_flux = 0
        #         last_edge = path_edges[-1]
        #         last_edge_flux = np.sum(reaction_flux_df.filter(like=str(last_edge), axis=0)[t].values[0])
        #         # inst_flux_df[path_idx][t] = last_edge_flux
        #         cum_path_flux += last_edge_flux
        #         inst_flux_df.loc[path_idx][t] = cum_path_flux
        #         # for edge in path_edges:
        #         #     flux = np.sum(reaction_flux_df.filter(like=str(edge), axis=0)[t].values[0])
        #         #     path_flux += flux
        #         # cum_path_flux += path_flux
        #         # inst_flux_df.loc[path_idx][t] = cum_path_flux

    return inst_flux_df


def check_path(model, path):
    spm = SpeciesPatternMatcher(model)
    jnk3 = model.monomers['JNK3']
    all_jnk3 = spm.match(jnk3(tyro='P', threo='P'))
    all_bound_ppjnk3 = spm.match(jnk3(b=ANY, tyro='P', threo='P'))
    all_bound_ppjnk3 = [model.get_species_index(sp) for sp in all_bound_ppjnk3]
    all_jnk3 = [model.get_species_index(sp) for sp in all_jnk3]
    mkk4_jnk3 = list(itertools.product([1], all_jnk3))
    mkk7_jnk3 = list(itertools.product([2], all_jnk3))
    # complexes_to_ppjnk3 = list(itertools.product(all_bound_ppjnk3, [27]))
    good_path = True
    for comb in range(len(mkk4_jnk3)):
        if set(mkk4_jnk3[comb]).issubset(path) or set(mkk7_jnk3[comb]).issubset(path):
            good_path = False
            break
    if good_path:
        for sp in all_bound_ppjnk3:
            if sp in path:
                complex_idx = path.index(sp)
                if path[complex_idx+1] != 27:
                    good_path = False
                    break

    return good_path


# from tropical.examples.double_enzymatic.mm_two_paths_model import model
#
# tspan = np.linspace(0,100,100)
# pars = np.load('/Users/dionisio/PycharmProjects/DynSign/tropical/examples/double_enzymatic/calibrated_pars.npy')
# bla = get_paths_flux(model, tspan, pars[0], [[0, 3, 5], [0, 4, 5]])

