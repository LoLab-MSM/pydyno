from __future__ import print_function
import math
import numpy as np
import pickle
from numpy.random import lognormal
from earm2_flat import model
from tropical.dynamic_signatures_range import run_tropical_multi
from pysb.simulator.scipyode import ScipyOdeSimulator


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
samples = 4


parameters = np.load('calibrated_6572pars.npy')
par_clus1 = parameters[0]

repeated_parameter_values = np.tile(par_clus1, (samples, 1))
for idx, par in parameters_ic.items():
    repeated_parameter_values[:, idx] = sample_lognormal(par, size=samples)
np.save('earm_diff_IC_par0.npy', repeated_parameter_values)

t = np.linspace(0, 20000, 100)
sims = ScipyOdeSimulator(model=model, tspan=t, param_values=repeated_parameter_values).run()
signatures = run_tropical_multi(model=model, simulations=sims, cpu_cores=25)

with open('earm_signatures_ic_par0.pickle', 'wb') as handle:
    pickle.dump(signatures, handle, protocol=pickle.HIGHEST_PROTOCOL)