from __future__ import print_function
from earm.lopez_embedded import model
from tropicalize import run_tropical
from multiprocessing import Pool
from multiprocessing import cpu_count
from numpy.random import lognormal
import numpy as np
import traceback
import sys
import functools
import pandas
import helper_functions as hf
import os
import csv


def sample_lognormal(parameter_ic, size, cv=0.25):
    """

    :param parameter_ic: PySB model parameter
    :param size:
    :param cv:
    :return:
    """
    mean = np.log(parameter_ic.value)
    cv = cv
    if parameter_ic.name == 'C3_0':
        cv = 0.282
    elif parameter_ic.name == 'XIAP_0' or parameter_ic == 'Bid_0':
        cv = 0.288
    elif parameter_ic.name == 'Bax_0':
        cv = 0.271
    else:
        parameter_ic.name == 'Bcl2_0'

    sd = cv
    return lognormal(mean, sd, size)

parameters_ic = {idx: p for idx, p in enumerate(model.parameters) if p in model.parameters_initial_conditions()[1:]}

samples = 100

pso_pars = hf.listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_5000')

all_pars_ic = np.zeros((len(pso_pars), len(model.parameters)))

for idx, pa in enumerate(pso_pars):
    pso_pars[idx] = hf.read_pars(pa)


repeated_parameter_values = np.repeat(pso_pars, samples, axis=0)

for idx, par in parameters_ic.items():
    repeated_parameter_values[:, idx] = sample_lognormal(par, size=samples*len(pso_pars))


def parameter_signatures(par, model, tspan):
    """

    :param par: parameter file path or vector of parameters
    :param model: PySB model
    :param tspan: time span
    :return: tropical signatures of input parameters
    """

    try:
        if os.path.isfile(par):
            parames = hf.read_pars(par)
        else:
            parames = par
        drivers = run_tropical(model, tspan, parameters=parames, sp_visualize=None)
        return drivers
    except:
        print(par)
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def compare_all_drivers_signatures(model, tspan, parameters_path, to_data_frame=False):
    """

    :param model: PySB model
    :param tspan: time span
    :param parameters_path: list of parameter files path
    :param to_data_frame: optional, True if a data frame of species that are drivers with all parameter sets is wanted
    :return: Save a file with all the tropical signatures from the different parameter sets that equally fit the data.
    If to_date_frame == True. It generates data frames of species that are drivers with all the parameter sets
    """

    p = Pool(cpu_count() - 1)
    all_drivers = p.map(functools.partial(parameter_signatures, model=model, tspan=tspan), parameters_path)
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
