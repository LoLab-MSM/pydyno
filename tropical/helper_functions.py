import os
import csv
import pandas as pd
import re
import numpy as np
from collections import OrderedDict


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
    f = open(par_path)
    data = csv.reader(f)
    param = [float(d[1]) for d in data]
    return param


def read_all_pars(pars_path, new_path=None):
    """
    Reads all pars in file or directory
    :param new_path:
    :param pars_path: Parameter file or directory path
    :return: DataFrame with all the parameters
    """
    if type(pars_path) is list:
        par_sets = pars_path
    elif os.path.isfile(pars_path):
        par_sets = list_pars_infile(pars_path, new_path)
    elif os.path.isdir(pars_path):
        par_sets = listdir_fullpath(pars_path)
    else:
        raise Exception("Not valid parameter file or path")

    all_pars = pd.DataFrame()
    for i, pat in enumerate(par_sets):
        var = pd.read_table(pat, sep=',', names=['parameters', 'val'])
        all_pars['par%d' % i] = var.val
    all_pars.set_index(var.parameters, inplace=True)
    all_pars_t = all_pars.transpose()
    return all_pars_t


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


def array_to_dataframe(array, dir_path, col_index=None, row_index=None):
    """

    :param array: array or path to npy file to convert to data frame
    :param dir_path: Path to the directory where the data frames are saved
    :param col_index: usually tspan
    :param row_index: usually the name or idx of parameter set
    :return:
    """

    if dir_path is not None:
        path = dir_path
    else:
        raise Exception("'dir_path' must be given.")

    if isinstance(array, str):
        tropical_data = np.load(array)
    else:
        tropical_data = array

    if col_index is not None:
        cindex = col_index
    else:
        cindex = range(len(tropical_data[0].values()[0]))

    if row_index is not None:
        rindex = row_index
    else:
        rindex = range(len(tropical_data))

    drivers_all = [set(dr.keys()) for dr in tropical_data]
    drivers_over_pars = set.intersection(*drivers_all)
    drivers_to_df = {}
    for sp in drivers_over_pars:
        tmp = [0] * len(drivers_all)
        for idx, tro in enumerate(tropical_data):
            tmp[idx] = tro[sp]
        drivers_to_df[sp] = tmp

    for sp in drivers_to_df.keys():
        pd.DataFrame(np.array(drivers_to_df[sp]), index=rindex, columns=cindex).to_csv(
            path + '/data_frame%d' % sp + '.csv')
    return
