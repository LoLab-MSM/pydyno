import itertools
import pandas as pd
import sympy
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from pysb.pattern import SpeciesPatternMatcher
from pysb import ANY


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

    rxns_df = pd.DataFrame(columns=tspan, index=range(len(model.reactions)))
    param_dict = dict((p.name, param_values[i]) for i, p in enumerate(model.parameters))
    sim_result = ScipyOdeSimulator(model, tspan=tspan, param_values=param_dict).run()
    y_df = sim_result.all
    for idx, reac in enumerate(model.reactions):
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
        edge_name = get_edges_name(reac['reactants'], reac['products'])
        rxns_df.rename(index={int(idx): edge_name}, inplace=True)
        rxns_df.loc[edge_name] = react_rate
    rxns_df['Total'] = rxns_df.sum(axis=1)
    return rxns_df


def get_paths_flux(model, tspan, param_values, paths, network, total_flux=True):

    reaction_flux_df = get_reaction_flux_df(model, tspan, param_values)
    inst_flux_df = pd.DataFrame(columns=tspan, index=range(len(paths)))
    paths_flux = {}
    for path_idx, path in enumerate(paths):
        checked_path = check_path(model, path)
        if not checked_path:
            print ('path {0} is not allowed'.format(path_idx))
            continue
        if total_flux:
            path_flux = 1
            for node in range(1, len(path)-1):
                in_edges = network.in_edges(path[node])
                out_edges = network.out_edges(path[node])
                total_flux_in = np.sum([reaction_flux_df.filter(like=str(edge), axis=0)['Total'].values[0] for edge in in_edges])
                total_flux_out = np.sum([reaction_flux_df.filter(like=str(edge), axis=0)['Total'].values[0] for edge in out_edges])
                flux_in = np.sum(reaction_flux_df.filter(like=str((path[node-1], path[node])), axis=0)['Total'].values[0])
                flux_out = np.sum(reaction_flux_df.filter(like=str((path[node], path[node+1])), axis=0)['Total'].values[0])
                percentage_in = flux_in / total_flux_in
                percentage_out = flux_out / total_flux_out
                node_flux = percentage_in*flux_out*percentage_out

                path_flux *= node_flux
            paths_flux[path_idx] = path_flux
        else:
            cum_path_flux = 0
            for t in tspan:
                path_flux = 1
                for node in range(1, len(path) - 1):
                    in_edges = network.in_edges(path[node])
                    out_edges = network.out_edges(path[node])
                    total_flux_in = np.sum(
                        [reaction_flux_df.filter(like=str(edge), axis=0)[t].values[0] for edge in in_edges])
                    total_flux_out = np.sum(
                        [reaction_flux_df.filter(like=str(edge), axis=0)[t].values[0] for edge in out_edges])
                    flux_in = np.sum(
                        reaction_flux_df.filter(like=str((path[node - 1], path[node])), axis=0)[t].values[0])
                    flux_out = np.sum(
                        reaction_flux_df.filter(like=str((path[node], path[node + 1])), axis=0)[t].values[0])

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

