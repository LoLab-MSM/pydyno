from __future__ import print_function
from earm.lopez_embedded import model
from tropicalize import run_tropical
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import traceback
import sys
import functools
import pandas
import helper_functions as hf


def all_parameters_signatures(par, model, tspan):
    """Return tropical signature of parameter set

        Keyword arguments:
        par -- parameter file path
        model -- PySB model
        tspan -- time span
    """
    try:
        parames = hf.read_pars(par)
        drivers = run_tropical(model, tspan, parameters=parames, sp_visualize=None)
        return drivers
    except:
        print(par)
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def compare_all_drivers_signatures(model, tspan, parameters_path, to_data_frame=False):
    """Save a file with all the tropical signatures from the different parameter sets that equally fit the data.
    If to_date_frame == True. It generates data frames of species that are drivers with all the parameter sets

        Keyword arguments:
        model -- PySB model
        tspan -- time span
        parameters_path -- list of parameter files path
        to_data_frame -- optional, True if a data frame of species that are drivers with all parameter sets is wanted
    """
    p = Pool(cpu_count() - 1)
    all_drivers = p.map(functools.partial(all_parameters_signatures, model=model, tspan=tspan), parameters_path)
    np.save("/home/oscar/tropical_project_new/drivers_all_parameters5", np.array(all_drivers))

    if to_data_frame:
        tropical_data = all_drivers
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
                             index=parameters_path,
                             columns=tspan[1:]).to_csv(
                '/home/oscar/tropical_project_new/data_frames5/data_frame%d' % sp + '.csv')
    return


t = np.linspace(0, 20000, 100)
pars = hf.listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_5')
compare_all_drivers_signatures(model, t, pars, to_data_frame=True)
