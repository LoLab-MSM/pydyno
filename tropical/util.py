import os
import pandas as pd
import re
import numpy as np
from collections import OrderedDict
from pysb.bng import generate_equations
import pysb
from pysb.simulator import ScipyOdeSimulator
from itertools import compress
import matplotlib.pyplot as plt
import sympy

def listdir_fullpath(d):
    """

    Parameters
    ----------
    d : str
        path to directory

    Returns
    -------
    a list of paths of the files in directory
    """
    return [os.path.join(d, f) for f in os.listdir(d)]


def list_pars_infile(f, new_path=None):
    """

    Parameters
    ----------
    f : str
        File that contain paths to parameter values
    new_path : str
        New path to assign to the parameter sets

    Returns
    -------
    list of parameter paths
    """
    par_sets = pd.read_csv(f, names=['parameters'])['parameters'].tolist()
    if new_path:
        par_sets = [w.replace(w.rsplit('/', 1)[0], new_path) for w in par_sets]
    return par_sets


def read_pars(par_path):
    """

    Parameters
    ----------
    par_path : str
        Path to parameter file

    Returns
    -------
    a list of parameter values
    """
    if par_path.endswith('.txt') or par_path.endswith('.csv'):
        data = np.genfromtxt(par_path, delimiter=',', dtype=None)
        if len(data.dtype) == 0:
            pars = data
        elif len(data.dtype) == 2:
            pars = data['f1']
        else:
            raise Exception('structure of the file is not supported')
    elif par_path.endswith('.npy'):
        pars = np.load(par_path)
    else:
        raise ValueError('format not supported')

    return pars


def read_all_pars(pars_path, new_path=None):
    """

    Parameters
    ----------
    pars_path : str
        Parameter file or directory path
    new_path : np.ndarray
        Array with all the parameters

    Returns
    -------

    """
    if isinstance(pars_path, list):
        par_sets = pars_path
    elif os.path.isfile(pars_path):
        par_sets = list_pars_infile(pars_path, new_path)
    elif os.path.isdir(pars_path):
        par_sets = listdir_fullpath(pars_path)
    else:
        raise Exception("Not valid parameter file or path")
    pars_0 = read_pars(par_sets[0])
    all_pars = np.zeros((len(par_sets), len(pars_0)))
    all_pars[0] = pars_0
    for idx in range(1, len(par_sets)):
        all_pars[idx] = read_pars(par_sets[idx])
    return all_pars


def parse_name(spec):
    """

    Parameters
    ----------
    spec : pysb.ComplexPattern
        Species's whose name is going to be parsed

    Returns
    -------
    A shorter version of the species name
    """
    m = spec.monomer_patterns
    lis_m = []
    name_counts = OrderedDict()
    parsed_name = ''
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"(?<=\').+(?=\')", str(m[i]))

        if not tmp_2:
            lis_m.append(tmp_1[0])
        else:
            lis_m.append(''.join([tmp_1[0], tmp_2[0]]))

    for name in lis_m:
        name_counts[name] = lis_m.count(name)

    for sp, counts in name_counts.items():
        if counts == 1:
            parsed_name += sp + '_'
        else:
            parsed_name += str(counts) + sp + '_'
    return parsed_name[:len(parsed_name) - 1]


def column(matrix, i):
    """

    Parameters
    ----------
    matrix : np.ndarray
        Array to get the column from
    i : int
        Column index to get from array

    Returns
    -------

    """
    return np.array([row[i] for row in matrix])


def sig_apop(t, f, td, ts):
    """

    Parameters
    ----------
    t : list-like
        Time variable in the function
    f : float
        is the fraction cleaved at the end of the reaction
    td : float
        is the delay period between TRAIL addition and half-maximal substrate cleavage
    ts : float
        is the switching time between initial and complete effector substrate  cleavage

    Returns
    -------

    """
    return f - f / (1 + np.exp((t - td) / (4 * ts)))


def pre_equilibration(model, time_search, parameters, tolerance=1e-6):
    """

    Parameters
    ----------
    model : pysb.Model
        pysb model
    time_search : np.array
        Time span array used to find the equilibrium
    parameters :  dict or np.array
        Model parameters used to find the equilibrium, it can be an array with all model parameters
        (this array must have the same order as model.parameters) or it can be a dictionary where the
        keys are the parameter names that want to be changed and the values are the new parameter
        values.
    tolerance : float
        Tolerance to define when the equilibrium has been reached

    Returns
    -------

    """
    # Solve system for the time span provided
    solver = ScipyOdeSimulator(model, tspan=time_search, param_values=parameters).run()
    y = solver.species.T
    dt = time_search[1] - time_search[0]

    time_to_equilibration = [0, 0]
    for idx, sp in enumerate(y):
        sp_eq = False
        derivative = np.diff(sp) / dt
        derivative_range = ((derivative < tolerance) & (derivative > -tolerance))
        # Indexes of values less than tolerance and greater than -tolerance
        derivative_range_idxs = list(compress(range(len(derivative_range)), derivative_range))
        for i in derivative_range_idxs:
            # Check if derivative is close to zero in the time points ahead
            if (derivative[i + 3] < tolerance) | (derivative[i + 3] > -tolerance):
                sp_eq = True
                if time_search[i] > time_to_equilibration[0]:
                    time_to_equilibration[0] = time_search[i]
                    time_to_equilibration[1] = i
            if not sp_eq:
                raise Exception('Equilibrium can not be reached within the time_search input')
            if sp_eq:
                break
        else:
            raise Exception('Species s{0} has not reached equilibrium'.format(idx))

    conc_eq = y[:, time_to_equilibration[1]]
    return time_to_equilibration, conc_eq


def find_nonimportant_nodes(model):
    """
    This function looks a the bidirectional reactions and finds the nodes that only have one incoming and outgoing
    reaction (edge)

    Parameters
    ----------
    model : pysb.Model
        PySB model to use

    Returns
    -------
    a list of non-important nodes
    """
    if not model.odes:
        pysb.bng.generate_equations(model)

    # gets the reactant and product species in the reactions
    rcts_sp = sum([i['reactants'] for i in model.reactions_bidirectional], ())
    pdts_sp = sum([i['products'] for i in model.reactions_bidirectional], ())
    # find the reactants and products that are only used once
    non_imp_rcts = set([x for x in range(len(model.species)) if rcts_sp.count(x) < 2])
    non_imp_pdts = set([x for x in range(len(model.species)) if pdts_sp.count(x) < 2])
    non_imp_nodes = set.intersection(non_imp_pdts, non_imp_rcts)
    passengers = non_imp_nodes
    return passengers


def visualization(model, tspan, y, sp_to_vis, all_signatures, all_comb, param_values):
    mach_eps = np.finfo(float).eps
    species_ready = list(set(sp_to_vis).intersection(all_signatures.keys()))
    par_name_idx = {j.name: i for i, j in enumerate(model.parameters)}
    if not species_ready:
        raise Exception('None of the input species is a driver')

    for sp in species_ready:

        # Setting up figure
        plt.figure(1)
        plt.subplot(313)

        signature = all_signatures[sp][1]

        # if not signature:
        #     continue

        # mon_val = OrderedDict()
        # merged_mon_comb = self.merge_dicts(*self.all_comb[sp].values())
        # merged_mon_comb.update({'ND': 'N'})
        #
        # for idx, mon in enumerate(list(set(signature))):
        #     mon_val[merged_mon_comb[mon]] = idx
        #
        # mon_rep = [0] * len(signature)
        # for i, m in enumerate(signature):
        #     mon_rep[i] = mon_val[merged_mon_comb[m]]
        # mon_rep = [mon_val[self.all_comb[sp][m]] for m in signature]
        plt.scatter(tspan, signature)
        plt.yticks(list(set(signature)))
        plt.ylabel('Dominant terms', fontsize=14)
        plt.xlabel('Time(s)', fontsize=14)
        plt.xlim(0, tspan[-1])
        # plt.ylim(0, max(y_pos))
        plt.subplot(312)
        for val, rr in all_comb[sp]['reactants'][1].items():
            mon = rr[0]
            var_to_study = [atom for atom in mon.atoms(sympy.Symbol)]
            arg_f1 = [0] * len(var_to_study)
            for idx, va in enumerate(var_to_study):
                if str(va).startswith('__'):
                    sp_idx = int(''.join(filter(str.isdigit, str(va))))
                    arg_f1[idx] = np.maximum(mach_eps, y[:, sp_idx])
                else:
                    arg_f1[idx] = param_values[par_name_idx[va.name]]

            f1 = sympy.lambdify(var_to_study, mon)
            mon_values = f1(*arg_f1)
            mon_name = str(val)
            plt.plot(tspan, mon_values, label=mon_name)
        plt.ylabel('Rate(m/sec)', fontsize=14)
        plt.legend(bbox_to_anchor=(-0.15, 0.85), loc='upper right', ncol=3)
        plt.xlim(0, tspan[-1])

        plt.subplot(311)
        plt.plot(tspan, y[:, sp], label=parse_name(model.species[sp]))
        plt.ylabel('Molecules', fontsize=14)
        plt.xlim(0, tspan[-1])
        plt.legend(bbox_to_anchor=(-0.15, 0.85), loc='upper right', ncol=1)
        plt.suptitle('Tropicalization' + ' ' + str(model.species[sp]), y=1.08)

        plt.tight_layout()
        plt.savefig('s%d' % sp + '.png', bbox_inches='tight', dpi=400)
        plt.clf()