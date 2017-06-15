import matplotlib
from earm.lopez_embedded import model
import numpy as np
from numpy.random import lognormal
from scipy.optimize import curve_fit
from pysb.integrate import odesolve
import matplotlib.pyplot as plt
import csv
import os

def sig_apop(t, f, td, ts):
    """Return the amount of substrate cleaved at time t.

    Keyword arguments:
    t -- time
    f -- is the fraction cleaved at the end of the reaction
    td -- is the delay period between TRAIL addition and half-maximal substrate cleavage
    ts -- is the switching time between initial and complete effector substrate  cleavage
    """
    return f - f / (1 + np.exp((t - td) / (4 * ts)))


def sample_lognormal(parameter_ic, size, cv=0.25):
    """

    :param parameter_ic:
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

samples = 10

directory = os.path.dirname(__file__)
pars_path = os.path.join(directory, 'parameters_5000')
f = open(pars_path+'/pars_embedded_5400.txt')
data = csv.reader(f)
parames = [float(i[1]) for i in data]
# parames[0] = 10

param_values = np.array(parames).reshape(1, len(parames))
repeated_parameter_values = np.repeat(param_values, samples, axis=0)

for idx, par in parameters_ic.items():
    repeated_parameter_values[:, idx] = sample_lognormal(par, size=samples)

tspan = np.linspace(0, 20000, 100)

td_list = [0]*samples

for pmter in range(repeated_parameter_values.shape[0]):
    # repeated_parameter_values[pmter, 63] = 0
    y = odesolve(model, tspan, repeated_parameter_values[pmter])
    try:
        cparp_info = curve_fit(sig_apop, tspan, y['cPARP']/repeated_parameter_values[pmter, 23], p0=[100, 100, 100], maxfev = 800)[0]
    except:
        pass

    print pmter
    td_list[pmter] = cparp_info[1]

td = np.array(td_list)/3600
plt.hist(td)
plt.show()
