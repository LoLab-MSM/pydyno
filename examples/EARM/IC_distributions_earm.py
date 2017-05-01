from __future__ import print_function
import math
import numpy as np
from numpy.random import lognormal
from earm.lopez_embedded import model
from tropical.max_plus_global import run_tropical_multiprocessing
import os
import tropical.helper_functions as hf


def normal_mu_sigma(log_mean, cv):
    """

    :param log_mean:
    :param cv:
    :return:
    """
    sigma_normal = math.sqrt(math.log((cv ** 2)+1))
    mu_normal = math.log(log_mean)  # - 0.5*(sigma_normal ** 2)
    return mu_normal, sigma_normal


def sample_lognormal(parameter_ic, size, cv=0.25):
    """

    :param parameter_ic: PySB model parameter
    :param size:
    :param cv:
    :return:
    """
    # mean = np.log(parameter_ic.value)
    cv = cv
    if parameter_ic.name == 'C3_0':
        cv = 0.282
    elif parameter_ic.name == 'XIAP_0' or parameter_ic == 'Bid_0':
        cv = 0.288
    elif parameter_ic.name == 'Bax_0':
        cv = 0.271
    elif parameter_ic.name == 'Bcl2_0':
        cv = 0.294

    mean_normal, mu_normal = normal_mu_sigma(parameter_ic.value, cv)

    return lognormal(mean_normal, mu_normal, size)

parameters_ic = {idx: p for idx, p in enumerate(model.parameters) if p in model.parameters_initial_conditions()[1:]}
samples = 10

directory = os.path.dirname(__file__)
parameters_path = os.path.join(directory, "parameters_5000")
all_parameters = hf.listdir_fullpath(parameters_path)
parameters = hf.read_pars(all_parameters[541])

repeated_parameter_values = np.tile(parameters, (samples, 1))
for idx, par in parameters_ic.items():
    repeated_parameter_values[:, idx] = sample_lognormal(par, size=samples)

t = np.linspace(0, 20000, 100)
a = run_tropical_multiprocessing(model, t, repeated_parameter_values, type_sign='consumption',
                                 find_passengers_by='imp_nodes', to_data_frame=True, dir_path=directory)
