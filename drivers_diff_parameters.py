from __future__ import print_function
from tropicalize import run_tropical
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import traceback
import sys
import functools
import pandas
import helper_functions as hf


def parameter_signatures(par, model, tspan):
    """

    :param par: parameter file path or vector of parameters
    :param model: PySB model
    :param tspan: time span
    :return: tropical signatures of input parameters
    """

    try:
        if isinstance(par, str):
            parames = hf.read_pars(par)
        else:
            parames = par
        drivers = run_tropical(model, tspan, parameters=parames, sp_visualize=None)
        return drivers
    except:
        print(par)
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def compare_all_drivers_signatures(model, tspan, parameters, to_data_frame=False, dir_path=None):
    """

    :param model: PySB model
    :param tspan: time span
    :param parameters: list of parameter files path
    :param to_data_frame: optional, True if a data frame of species that are drivers with all parameter sets is wanted
    :param dir_path: Path to directory where data frames are saved if to_data_frame=True
    :return: Save a file with all the tropical signatures from the different parameter sets that equally fit the data.
    If to_date_frame == True. It generates data frames of species that are drivers with all the parameter sets
    """

    p = Pool(cpu_count() - 1)
    all_drivers = p.map(functools.partial(parameter_signatures, model=model, tspan=tspan), parameters)
    np.save("drivers_CORM", np.array(all_drivers))

    if to_data_frame:
        array_to_dataframe(all_drivers, dir_path, tspan[1:], parameters)
    return


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
        cindex = range(tropical_data.shape[1])

    if row_index is not None:
        rindex = row_index
    else:
        rindex = range(tropical_data.shape[0])

    drivers_all = [set(dr.keys()) for dr in tropical_data]
    drivers_over_pars = set.intersection(*drivers_all)
    drivers_to_df = {}
    for sp in drivers_over_pars:
        tmp = [0] * len(drivers_all)
        for idx, tro in enumerate(tropical_data):
            tmp[idx] = tro[sp]
        drivers_to_df[sp] = tmp

    for sp in drivers_to_df.keys():
        pandas.DataFrame(np.array(drivers_to_df[sp]),
                         index=rindex,
                         columns=cindex).to_csv(path + '/data_frame%d' % sp + '.csv')
    return

# t = np.linspace(0, 20000, 100)
# pars = hf.listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_5')
# compare_all_drivers_signatures(model, t, pars, to_data_frame=True)

# all_drivers = np.load('/home/oscar/Documents/tropical_projetct/drivers_all_parameters5000.npy')
# drivers_all = {idx: dr.keys() for idx, dr in enumerate(all_drivers)}
#
# for i in drivers_all:
#     for j in drivers_all:
#         if set(drivers_all[i]) == set(drivers_all[j]):
#             if i != j:
#                 print (i, j)
