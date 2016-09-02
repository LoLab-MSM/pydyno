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


def read_all_pars(pars_path):
    """
    Reads all pars in file or directory
    :param pars_path: Parameter file or directory path
    :return: DataFrame with all the parameters
    """
    if os.path.isfile(pars_path):
        par_sets = pd.read_csv(pars_path, names=['parameters'])['parameters'].tolist()
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
    return parsed_name[:len(parsed_name)-1]


def _find_nearest_zero(array):
    """
    Finds array value closer to zero
    :param array: Array
    :return: Value of array closer to zero
    """
    idx = np.nanargmin(np.abs(array))
    return array[idx]
