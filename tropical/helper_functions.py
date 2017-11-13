import os
import pandas as pd
import re
import numpy as np
from collections import OrderedDict
from pysb.bng import generate_equations
import pysb
from scipy.optimize import curve_fit
from pysb.simulator import ScipyOdeSimulator
from itertools import compress


def listdir_fullpath(d):
    """
    Gets a list of path files in directory
    :param d: path to directory
    :return: a list of path of files in directory
    """
    return [os.path.join(d, f) for f in os.listdir(d)]


def list_pars_infile(f, new_path=None):
    """

    :param f: File that contain paths to parameter set values
    :param new_path: parameter paths of f may be different to where they are in the local computer
    :return: list of parameter paths
    """
    par_sets = pd.read_csv(f, names=['parameters'])['parameters'].tolist()
    if new_path:
        par_sets = [w.replace(w.rsplit('/', 1)[0], new_path) for w in par_sets]
    return par_sets


def read_pars(par_path):
    """
    Reads parameter file
    :param par_path: path to parameter file
    :return: Return a list of parameter values from csv file
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


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def read_all_pars(pars_path, new_path=None):
    """
    Reads all pars in file or directory
    :param new_path:
    :param pars_path: Parameter file or directory path
    :return: np.ndarray with all the parameters
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

    # all_pars = pd.DataFrame()
    # for i, pat in enumerate(par_sets):
    #     var = pd.read_table(pat, sep=',', names=['parameters', 'val'])
    #     all_pars['par%d' % i] = var.val
    # all_pars.set_index(var.parameters, inplace=True)
    # all_pars_t = all_pars.transpose()
    # return all_pars_t


def parse_name(spec):
    """
    Parses name of species
    :param spec: species name to parse
    :return: parsed name of species
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
    """Return the i column of a matrix

    Keyword arguments:
    matrix -- matrix to get the column from
    i -- column to get fro the matrix
    """
    return np.array([row[i] for row in matrix])


def _find_nearest_zero(array):
    """
    Finds array value closer to zero
    :param array: Array
    :return: Value of array closer to zero
    """
    idx = np.nanargmin(np.abs(array))
    return array[idx]


def sig_apop(t, f, td, ts):
    """Return the amount of substrate cleaved at time t.

    Keyword arguments:
    t -- time
    f -- is the fraction cleaved at the end of the reaction
    td -- is the delay period between TRAIL addition and half-maximal substrate cleavage
    ts -- is the switching time between initial and complete effector substrate  cleavage
    """
    return f - f / (1 + np.exp((t - td) / (4 * ts)))


def check_param_values(model, param_values):
    if param_values is not None:
        if type(param_values) is str:
            pars_to_check = read_pars(param_values)
        else:
            pars_to_check = param_values
        # accept vector of parameter values as an argument
        if len(pars_to_check) != len(model.parameters):
            raise Exception("param_values must be the same length as model.parameters")
        # convert model parameters into dictionary
        pars_checked = dict((p.name, pars_to_check[i]) for i, p in enumerate(model.parameters))
    else:
        # create parameter vector from the values in the model
        pars_checked = dict((p.name, p.value) for i, p in enumerate(model.parameters))
    return pars_checked


def pre_equilibration(model, time_search, ligand_par_name, ligand_idx, ligand_value=None, parameters=None, tolerance=1e-6):
    """

    :param model: PySB model
    :param ligand_par_name: Species whose value want to be changed.
    :param time_search: time span array to be used to find the equilibrium
    :param tolerance: (tolerance, -tolerance) Range within equilibrium is considered as reached
    :param ligand_value: Initial condition of ligand (usually zero)
    :param parameters: Model parameters, must have same order as model.parameters
    :return:
    """
    param_dict = parameters
    # Check if ligand name to be used for pre equilibration is provided
    if not isinstance(ligand_par_name, str):
        raise Exception('ligand must be a string with the parameter name')

    ligand_prev = param_dict[ligand_par_name]

    if ligand_value is not None:
        param_dict[ligand_par_name] = ligand_value
    else:
        param_dict[ligand_par_name] = 0

    # Solve system for the time span provided
    solver = ScipyOdeSimulator(model, tspan=time_search, param_values=param_dict).run()
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

    y[ligand_idx, time_to_equilibration[1]] = ligand_prev
    conc_eq = y[:, time_to_equilibration[1]]
    return time_to_equilibration, conc_eq


def truncate_signature_death(simulations, signatures, species_to_fit):
    if type(simulations) == str:
        all_simulations = np.load(simulations)
    else:
        all_simulations = simulations
    if type(signatures) == str:
       snts = pd.read_csv(signatures, header=0, index_col=0)
    else:
        snts = signatures

    tspan = snts.columns.values
    tspan = tspan.astype(np.float)

    for i, y in enumerate(all_simulations):
        try:
            td = curve_fit(f=sig_apop, xdata=tspan, ydata=y[species_to_fit], p0=[100, 100, 100])[0][1]
        except:
            print "Trajectory {0} can't be fitted".format(i)
        idx_death = min(enumerate(tspan), key=lambda x: abs(x[1]-td))[0]
        snts.loc[i][idx_death:len(tspan)] = 100

    snts.to_csv(signatures)



def sps_signature_to_df(signatures, dir_path, global_signature=False, col_index=None, row_index=None):
    """

    :param signatures: array or path to npy file to convert to data frame
    :param dir_path: Path to the directory where the data frames are saved
    :param col_index: usually tspan
    :param row_index: usually the name or idx of parameter set
    :return:
    """

    if dir_path is not None:
        path = dir_path
    else:
        raise Exception("'dir_path' must be given.")

    if isinstance(signatures, str):
        tropical_data = np.load(signatures)
    else:
        tropical_data = signatures

    if col_index is not None:
        cindex = col_index
    else:
        cindex = range(len(tropical_data[0].values()[0]))

    if row_index is not None:
        rindex = row_index
    else:
        rindex = range(len(tropical_data))

    if global_signature:
        pd.DataFrame(np.array(signatures), index=rindex, columns=cindex).to_csv(
            path + '/data_frame_global_signature' + '.csv')
    else:
        drivers_all_ic = [set(dr.keys()) for dr in tropical_data]
        drivers_over_pars = set.intersection(*drivers_all_ic)
        drivers_to_df = {}
        for sp in drivers_over_pars:
            tmp = [0] * len(drivers_all_ic)
            for idx, tro in enumerate(tropical_data):
                tmp[idx] = tro[sp]
            drivers_to_df[sp] = tmp

        for sp in drivers_to_df.keys():
            pd.DataFrame(np.array(drivers_to_df[sp]), index=rindex, columns=cindex).to_csv(
                path + '/data_frame%d' % sp + '.csv')
    return


def get_species_initial(model, sp):
    if not model.species:
        generate_equations(model)

    if type(sp) == int:
        species = model.species[sp]
    elif type(sp) == pysb.ComplexPattern:
        species = sp
    else:
        raise Exception('species type is invalid')

    initial_value = 0
    for i in model.initial_conditions:
        if species.is_equivalent_to(i[0]):
            initial_value = i[1].value
    return initial_value


def find_nonimportant_nodes(model):
    """
    This function looks a the bidirectional reactions and finds the nodes that only have one incoming and outgoing
    reaction (edge)
    :return: a list of non-important nodes
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


