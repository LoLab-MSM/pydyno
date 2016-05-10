import os
import csv
import pandas as pd
import re
import numpy as np

def listdir_fullpath(d):
    """Return a list of path of files in directory

       Keyword arguments:
       d -- path to directory
    """
    return [os.path.join(d, f) for f in os.listdir(d)]


def read_pars(par_path):
    """Return a list of parameter values from csv file

       keyword arguments:
       par_path -- path to parameter file
    """
    f = open(par_path)
    data = csv.reader(f)
    param = [float(d[1]) for d in data]
    return param


def read_all_pars(pars_path):
    par_sets = listdir_fullpath(pars_path)
    all_pars = pd.DataFrame()
    for i, pat in enumerate(par_sets):
        var = pd.read_table(pat, sep=',', names=['parameters', 'val'])
        all_pars['par%d' % i] = var.val
    all_pars.set_index(var.parameters, inplace=True)
    all_pars_t = all_pars.transpose()
    return all_pars_t


def _parse_name(spec):
    """Returns parsed name of species

        keyword arguments:
        spec -- species name to parse
    """
    m = spec.monomer_patterns
    lis_m = []
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"(?<=\').+(?=\')", str(m[i]))
        if not tmp_2:
            lis_m.append(tmp_1[0])
        else:
            lis_m.append(''.join([tmp_1[0], tmp_2[0]]))
    return '_'.join(lis_m)


def _find_nearest_zero(array):
    idx = np.nanargmin(np.abs(array))
    return array[idx]
