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

def get_paths_flux(model, tspan, param_values, paths, network, total_flux=True):

    reaction_flux_df = get_reaction_flux_df(model, tspan, param_values)
    inst_flux_df = pd.DataFrame(columns=tspan, index=range(len(paths)))
    paths_flux = {}
    for path_idx, path in enumerate(paths):
        # checked_path = check_path(model, path)
        # if not checked_path:
        #     print ('path {0} is not allowed'.format(path_idx))
        #     continue
        if total_flux:
            path_flux = 1
            sp_nodes = get_sp_nodes(path)
            r_nodes = get_r_nodes(path)
            node_flux = 1
            for sp_node, r_node in zip(sp_nodes[1:-1], r_nodes[:-1]):
                in_edges = network.in_edges(sp_node)
                out_edges = network.out_edges(sp_node)
                total_flux_in_l = np.array([reaction_flux_df.loc[edge[0], 'Total'] for edge in in_edges])
                total_flux_in = np.sum(total_flux_in_l[np.where(total_flux_in_l > 0)])
                flux_in_negative = np.sum(total_flux_in_l[np.where(total_flux_in_l < 0)])
                total_flux_out_l = np.array([reaction_flux_df.loc[edge[1], 'Total'] for edge in out_edges])
                total_flux_out = np.sum(total_flux_out_l[np.where(total_flux_out_l > 0)])
                flux_out_negative = np.sum(total_flux_out_l[np.where(total_flux_out_l < 0)])
                total_flux_in += flux_out_negative * -1
                total_flux_out += flux_in_negative * -1
                # print ('in', sp_node, flux_in_negative)
                # print('out', sp_node, flux_out_negative)

                flux_in = np.sum(reaction_flux_df.loc[r_node, 'Total'])
                next_r_node = r_nodes.index(r_node) + 1
                flux_out = np.sum(reaction_flux_df.loc[r_nodes[next_r_node], 'Total'])
                if total_flux_out < np.finfo(float).eps:
                    percentage_out = 0
                else:
                    percentage_out = flux_out / total_flux_out

                if total_flux_in < np.finfo(float).eps:
                    percentage_in = 0
                else:
                    percentage_in = flux_in / total_flux_in

                node_flux = percentage_in * total_flux_out * percentage_out

                path_flux *= node_flux
            if path_flux <= 0:
                continue
            paths_flux[path_idx] = path_flux
        else:
            cum_path_flux = 0
            sp_nodes = get_sp_nodes(path)
            r_nodes = get_r_nodes(path)
            for t in tspan:
                path_flux = 1
                for sp_node, r_node in zip(sp_nodes[1:-1], r_nodes[:-1]):
                    in_edges = network.in_edges(sp_node)
                    out_edges = network.out_edges(sp_node)

                    total_flux_in = np.sum(
                        [reaction_flux_df.loc[edge[0], t] for edge in in_edges])
                    total_flux_out = np.sum(
                        [reaction_flux_df.loc[edge[1], t] for edge in out_edges])
                    flux_in = np.sum(
                        reaction_flux_df.loc[r_node, t])
                    next_r_node = r_nodes.index(r_node) + 1
                    flux_out = np.sum(
                        reaction_flux_df.loc[r_nodes[next_r_node], t])

                    if total_flux_out < np.finfo(float).eps:
                        percentage_out = 0
                    else:
                        percentage_out = flux_out / total_flux_out

                    if total_flux_in < np.finfo(float).eps:
                        percentage_in = 0
                    else:
                        percentage_in = flux_in / total_flux_in

                    node_flux = percentage_in * flux_out * percentage_out
                    # print (total_flux_in)
                    # print (total_flux_out)
                    # print (flux_in)
                    # print (flux_out)

                    path_flux *= node_flux
                cum_path_flux += path_flux

                inst_flux_df.loc[path_idx][t] = path_flux

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
    if total_flux:
        return paths_flux
    else:
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

